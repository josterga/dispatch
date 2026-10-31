"""Digest chat: an interactive agent over the user's digests, read in layers.

  Layer 0  catalog of every dispatch (in the system prompt)          digests.catalog()
  Layer 1  read_digest: one run's digest, latest or up to 5 back       db.recent_outputs()
  Layer 1b read_section: one named section from every latest digest  (cross-digest questions)
  Layer 2  run_sources: what that run read                             digests.run_sources()
  lateral  search_digests: keyword search over two weeks of digests    db.search_outputs()
  Layer 3  web_search / web_fetch, only for turns with Research on     tools.dispatch_tool()
           research_run: a full research pipeline run, fresh (no digest context), Research on only
  propose_agent: drafts an agent (+ optional schedule) for the user to confirm in the UI; never writes one itself

A thread is a sessions row (source 'chat'); turns are messages, tool steps are tool_calls.
A turn runs as a JobManager job (job_manager.submit_chat), so it shares the single model slot.
"""
import json
import os
import re
import time
from typing import Callable, Optional

import db
import digests
from agent_config import DomainConfig
from pipeline import run_agent
from prompts import PROMPT_CHAT, PROMPT_CHAT_OFFER, PROMPT_CHAT_RESEARCH
from tools import TOOLS, dispatch_tool

MAX_ITERATIONS = 6          # tool rounds per turn before the forced final answer
DIGEST_MAX_CHARS = 8000     # one read_digest result
FOCUS_MAX_CHARS = 8000      # the digest a thread was opened from, pre-loaded into the system prompt
HISTORY_TURNS = 6           # earlier Q&A pairs sent with each turn
MAX_SOURCES = 40
SECTION_MAX_CHARS = 1500    # one section in read_section
TURN_READ_BUDGET = 40000    # tool-output chars per answer; past it read_digest returns an outline (keeps the 32k context safe)
RESEARCH_MAX_CHARS = 6000   # one research_run result
RESEARCH_OUTPUT = """Markdown, under 500 words. Only use the research provided; never invent facts, names, numbers, dates or URLs.

## Answer
The direct answer in 2-4 sentences, with sources.

## Details
Up to 8 supporting points, one line each with (source: URL).

## Open questions
One line on anything the research could not confirm. Skip if none."""
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}


DIGEST_TOOLS = [
    _fn("read_digest", "Read one dispatch's digest. runs_back=0 is the latest run, 1 the run before, up to 5.",
        {"dispatch_id": {"type": "integer", "description": "From the catalog: [dispatch N]"},
         "runs_back": {"type": "integer", "description": "0 = latest (default)"}}, ["dispatch_id"]),
    _fn("read_section", "Read one named section (e.g. 'Act on', 'Top stories', 'Urgent') from the latest digest of every "
        "dispatch that has it. Best for questions across all digests.",
        {"heading": {"type": "string", "description": "Section heading, or part of it"}}, ["heading"]),
    _fn("run_sources", "List the sources (titles, URLs, snippets) a run read before writing its digest.",
        {"run_id": {"type": "integer", "description": "From the catalog or read_digest"}}, ["run_id"]),
    _fn("search_digests", "Keyword search across all digests from the last 14 days (every word must appear). "
        "Returns dispatch, run, section heading and the matching line.",
        {"query": {"type": "string", "description": "1-4 distinctive words"},
         "days": {"type": "integer", "description": "How far back, default 14"}}, ["query"]),
]
RESEARCH_TOOL = _fn(
    "research_run", "Run a full research pass on a question: plans several searches, reads the results and writes a "
    "sourced brief. Slower (a few minutes) but more thorough than web_search. Use it when the user asks to research "
    "something, or for a broad or new topic their digests don't cover; use web_search for a quick lookup.",
    {"question": {"type": "string", "description": "What to research, as a full sentence"},
     "freshness": {"type": "string", "enum": ["d", "w", "m", "y"], "description": "Only results from the last day/week/month/year"}},
    ["question"])
PROPOSE_TOOL = _fn(
    "propose_agent", "Draft a new dispatch agent (a recurring research brief) from the conversation and show it to the "
    "user, who reviews, edits and creates it. It is not created until they confirm. Use it when the user wants to "
    "track, watch or follow a topic regularly, or asks for a new agent or dispatch.",
    {"name": {"type": "string", "description": "Slug: lowercase letters, digits, dashes (e.g. bay-housing)"},
     "label": {"type": "string", "description": "Display name, e.g. 'Bay Area housing'"},
     "description": {"type": "string", "description": "One line: what it covers"},
     "goal_template": {"type": "string", "description": "What to find, written to the researcher: topics, angles, what matters to the user"},
     "output_template": {"type": "string", "description": "Markdown format: word limit and ## sections, each with what goes in it and (source: URL)"},
     "sources": {"type": "array", "items": {"type": "string"}, "description": "Domains to favour, e.g. sfchronicle.com"},
     "freshness": {"type": "string", "enum": ["d", "w", "m", "y"], "description": "d for daily briefs, w for weekly"},
     "planner_query_count": {"type": "integer", "description": "Searches per run, 3-8 (default 5)"},
     "schedule": {"type": "string", "description": "Optional cron in the user's time zone, e.g. '30 7 * * *' (7:30 daily), '0 9 * * 1' (Mondays 9:00)"},
     "prompt": {"type": "string", "description": "Optional extra instruction for each scheduled run"}},
    ["name", "label", "goal_template", "output_template"])
_LOCAL = {t["function"]["name"] for t in DIGEST_TOOLS}
_WEB = {t["function"]["name"] for t in TOOLS}


def _labels() -> dict[int, str]:
    agents = {a["name"]: a.get("label") or a["name"] for a in db.get_agents()}
    return {d["id"]: d.get("label") or agents.get(d["agent"]) or d["agent"] for d in db.get_dispatches()}


def _int(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def read_digest(dispatch_id, runs_back=0) -> dict:
    dispatch_id, runs_back = _int(dispatch_id), max(0, min(5, _int(runs_back)))
    label = _labels().get(dispatch_id)
    if not label:
        return {"error": f"No dispatch {dispatch_id}; use an id from the catalog."}
    runs = db.recent_outputs(dispatch_id, runs_back + 1)
    if len(runs) <= runs_back:
        return {"error": f"{label} has only {len(runs)} run(s) with a digest."}
    r = runs[runs_back]
    text = r["content"]
    return {"dispatch_id": dispatch_id, "dispatch": label, "run_id": r["id"], "runs_back": runs_back,
            "when": digests._local(r["started_at"]),
            "digest": text[:DIGEST_MAX_CHARS] + ("\n[… digest cut here]" if len(text) > DIGEST_MAX_CHARS else "")}


def _sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) for each markdown section; a section ends at the next heading of the same or higher level."""
    lines, out = text.splitlines(), []
    for i, l in enumerate(lines):
        m = re.match(r"(#{1,6})\s+(.*)", l)
        if not m:
            continue
        level, body = len(m.group(1)), []
        for nxt in lines[i + 1:]:
            n = re.match(r"(#{1,6})\s", nxt)
            if n and len(n.group(1)) <= level:
                break
            body.append(nxt)
        out.append((m.group(2).strip(), "\n".join(body).strip()))
    return out


def read_section(heading) -> dict:
    want = str(heading or "").strip().lower()
    if not want:
        return {"error": "Give a section heading."}
    labels, found = _labels(), []
    for d in db.get_dispatches(enabled_only=True):
        runs = db.recent_outputs(d["id"], 1)
        if not runs:
            continue
        for h, body in _sections(runs[0]["content"]):
            if want in h.lower():
                found.append({"dispatch_id": d["id"], "dispatch": labels.get(d["id"]), "run_id": runs[0]["id"],
                              "when": digests._local(runs[0]["started_at"]), "heading": h,
                              "text": (body[:SECTION_MAX_CHARS] + (" […]" if len(body) > SECTION_MAX_CHARS else "")) or "(empty)"})
                break
    return {"heading": heading, "sections": found} if found else {"heading": heading, "sections": [], "note": "No latest digest has a section with that heading."}


def run_sources(run_id) -> dict:
    run_id = _int(run_id)
    dispatch_id = db.session_dispatch_id(run_id)
    if not dispatch_id:
        return {"error": f"No dispatch run {run_id}."}
    out = digests.run_sources(run_id, snippet=240)
    out["dispatch"] = _labels().get(dispatch_id)
    if len(out["sources"]) > MAX_SOURCES:
        out["note"] = (out.get("note", "") + f" Showing {MAX_SOURCES} of {len(out['sources'])}.").strip()
        out["sources"] = out["sources"][:MAX_SOURCES]
    return out


def search_digests(query, days=14) -> dict:
    query = str(query or "").strip()
    rows = db.search_outputs(query, days=max(1, min(60, _int(days, 14))), limit=10)
    labels, words = _labels(), [w for w in query.lower().split() if w]
    hits = []
    for r in rows:
        heading, line = "", ""
        for l in r["content"].splitlines():
            if re.match(r"#{1,3}\s", l):
                heading = l.lstrip("#").strip()
            elif any(w in l.lower() for w in words):
                line = l.strip()
                break
        hits.append({"dispatch_id": r["dispatch_id"], "dispatch": labels.get(r["dispatch_id"]), "run_id": r["id"],
                     "when": digests._local(r["started_at"]), "section": heading, "line": line[:300]})
    return {"query": query, "hits": hits} if hits else {"query": query, "hits": [], "note": "No digest mentions all of these words."}


def research_run(client, question, freshness=None, cancel_event=None) -> dict:
    """A fresh pipeline run on the question, inline in this turn (the chat already holds the model slot)."""
    import job_manager
    question = str(question or "").strip()
    if not question:
        return {"error": "Give a question to research."}
    config = DomainConfig(
        agent_name="adhoc", goal_template=question, output_template=RESEARCH_OUTPUT,
        tool_bindings=["web_search"], planner_query_count=4, max_validator_loops=1,
        freshness=freshness if freshness in ("d", "w", "m", "y") else None, recall_runs=0, dedupe_today=False,
    )
    run_id, answer = job_manager.run_research_session(client, config, question, cancel_event)
    if not answer:
        return {"run_id": run_id, "error": "The research run produced no result.", "output": db.get_output(run_id)}
    cut = answer[:RESEARCH_MAX_CHARS] + ("\n[… cut]" if len(answer) > RESEARCH_MAX_CHARS else "")
    return {"run_id": run_id, "brief": cut,
            "note": "Fresh research run (not from the digests). Cite its URLs; the user can open the run to see its sources."}


def propose_agent(chat_id: int, args: dict) -> dict:
    """Validate a drafted agent and store it as a 'proposal' message for the UI to confirm. Never creates anything."""
    from job_manager import cron_trigger
    name = str(args.get("name") or "").strip().lower()
    problems = []
    if not _SLUG_RE.match(name):
        problems.append("name must be a slug: 2-41 lowercase letters, digits or dashes")
    elif db.get_agent(name):
        problems.append(f"an agent named '{name}' already exists: pick another name")
    for f in ("label", "goal_template", "output_template"):
        if not str(args.get(f) or "").strip():
            problems.append(f"{f} is required")
    schedule = str(args.get("schedule") or "").strip()
    if schedule:
        try:
            cron_trigger(schedule, db.local_timezone())
        except Exception:
            problems.append(f"schedule '{schedule}' is not a valid 5-field cron")
    if problems:
        return {"error": "Fix and call propose_agent again: " + "; ".join(problems)}
    freshness = args.get("freshness") if args.get("freshness") in ("d", "w", "m", "y") else None
    proposal = {
        "name": name, "label": str(args["label"]).strip(), "description": str(args.get("description") or "").strip(),
        "goal_template": str(args["goal_template"]).strip(), "output_template": str(args["output_template"]).strip(),
        "sources": [str(x).strip() for x in (args.get("sources") or []) if str(x).strip()][:15],
        "freshness": freshness,
        "planner_query_count": max(3, min(8, _int(args.get("planner_query_count"), 5))),
        "schedule": schedule or None, "prompt": str(args.get("prompt") or "").strip(),
    }
    db.log_message(chat_id, "proposal", json.dumps(proposal))
    return {"shown": True, "note": "The draft is shown to the user as a card they can edit, then create (and schedule). "
            "It does not exist yet. Briefly say what you drafted and that they can adjust it before creating it."}


def step_label(name: str, args: dict, labels: Optional[dict] = None) -> str:
    """Short human label for a tool step ("read World, 1 run back")."""
    labels = labels if labels is not None else _labels()
    if name == "read_digest":
        rb = _int(args.get("runs_back"))
        return f"read {labels.get(_int(args.get('dispatch_id')), 'dispatch ' + str(args.get('dispatch_id')))}" + (f" ({rb} back)" if rb else "")
    if name == "read_section":
        return f"section \u201c{args.get('heading')}\u201d across digests"
    if name == "run_sources":
        return f"sources of run {args.get('run_id')}"
    if name == "search_digests":
        return f"searched digests: {args.get('query')}"
    if name == "web_search":
        return f"web: {args.get('query')}"
    if name == "web_fetch":
        return f"page: {args.get('url')}"
    if name == "research_run":
        return f"research run: {args.get('question')}"
    if name == "propose_agent":
        return f"drafted agent: {args.get('label') or args.get('name')}"
    return name


def _router(research: bool, on_step: Optional[Callable[[str], None]], client=None, cancel_event=None):
    labels = _labels()
    used = {"chars": 0}

    def route(name: str, args: dict, session_id: int) -> dict:
        args = args if isinstance(args, dict) else {}
        if on_step:
            on_step(step_label(name, args, labels))
        if name in _WEB or name == "research_run":
            if not research:
                return {"error": "Research is off for this message. Tell the user they can turn it on to search the web."}
            if name in _WEB:
                return dispatch_tool(name, args, session_id)
        t0 = time.monotonic()
        try:
            if name == "research_run":
                if on_step:
                    on_step("running research (a few minutes)")
                result = research_run(client, args.get("question"), args.get("freshness"), cancel_event)
            elif name == "propose_agent":
                result = propose_agent(session_id, args)
            elif name == "read_digest":
                result = read_digest(args.get("dispatch_id"), args.get("runs_back", 0))
                if used["chars"] > TURN_READ_BUDGET and "digest" in result:
                    result["digest"] = digests.outline(result["digest"], max_chars=1500)
                    result["note"] = "Shortened to an outline: this answer's reading budget is used up. Answer with what you have."
            elif name == "read_section":
                result = read_section(args.get("heading"))
            elif name == "run_sources":
                result = run_sources(args.get("run_id"))
            elif name == "search_digests":
                result = search_digests(args.get("query"), args.get("days", 14))
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        used["chars"] += len(json.dumps(result))
        db.log_tool_call(session_id, name, args, result, int((time.monotonic() - t0) * 1000))
        return result
    return route


def _history(chat: dict) -> list[dict]:
    """Earlier turns as plain messages; each answer carries a one-line note of what it opened, not the tool output."""
    msgs, steps = chat["messages"], chat["steps"]
    labels = _labels()
    turns, q, proposed = [], None, []
    for m in msgs:
        if m["role"] == "user":
            q, proposed = m, []
        elif m["role"] == "proposal" and q:
            try:
                p = json.loads(m["content"])
                created = " (created)" if db.get_agent(p.get("name", "")) else " (not created yet)"
                proposed.append(f"{p.get('label')} [{p.get('name')}]{created}")
            except (ValueError, TypeError):
                pass
        elif m["role"] == "assistant" and q:
            opened = [step_label(s["tool_name"], s["args"], labels) for s in steps if q["created_at"] <= s["created_at"] <= m["created_at"]]
            note = f"\n\n[Opened for this answer: {'; '.join(opened)}]" if opened else ""
            if proposed:
                note += f"\n[Drafted agents: {'; '.join(proposed)}]"
            turns.append([{"role": "user", "content": q["content"]}, {"role": "assistant", "content": m["content"] + note}])
            q = None
    return [m for t in turns[-HISTORY_TURNS:] for m in t]


def _focus_block(chat: dict) -> str:
    sid = chat.get("focus_session_id")
    text = db.get_output(sid) if sid else None
    if not text:
        return ""
    label = _labels().get(chat.get("focus_dispatch_id"), "a dispatch")
    cut = text[:FOCUS_MAX_CHARS] + ("\n[… cut]" if len(text) > FOCUS_MAX_CHARS else "")
    return (f"\nFOCUS: the user opened this chat from the {label} digest (dispatch {chat.get('focus_dispatch_id')}, run {sid}). "
            f"\"This\", \"it\" and \"here\" refer to it. Its text:\n<digest>\n{cut}\n</digest>\n")


def run_turn(client, chat_id: int, question: str, research: bool = False,
             cancel_event=None, on_step: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Answer one question in a thread; logs the question, tool steps and answer. Returns the answer or None."""
    chat = db.get_chat(chat_id)
    if not chat:
        raise ValueError(f"no chat {chat_id}")
    history = _history(chat)
    db.log_message(chat_id, "user", question)
    system = PROMPT_CHAT.format(
        research=PROMPT_CHAT_RESEARCH if research else "",
        offer="" if research else PROMPT_CHAT_OFFER,
        profile=f"\nABOUT THE USER: {os.getenv('USER_PROFILE')}\n" if os.getenv("USER_PROFILE") else "",
        focus=_focus_block(chat),
        catalog=digests.catalog(),
    )
    tools = DIGEST_TOOLS + [PROPOSE_TOOL] + (TOOLS + [RESEARCH_TOOL] if research else [])
    if on_step:
        on_step("thinking")
    answer = run_agent(client, history + [{"role": "user", "content": question}], chat_id,
                       tools=tools, system_prompt=system, cancel_event=cancel_event, think=False,
                       max_iterations=MAX_ITERATIONS, phase="chat", tool_router=_router(research, on_step, client, cancel_event))
    if not answer:
        db.log_message(chat_id, "error", "Cancelled." if cancel_event and cancel_event.is_set() else "No answer: the model call failed.")
    return answer
