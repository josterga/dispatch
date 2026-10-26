"""Digest chat: an interactive agent over the user's digests, read in layers.

  Layer 0  catalog of every dispatch (in the system prompt)          digests.catalog()
  Layer 1  read_digest: one run's digest, latest or up to 5 back       db.recent_outputs()
  Layer 1b read_section: one named section from every latest digest  (cross-digest questions)
  Layer 2  run_sources: what that run read                             digests.run_sources()
  lateral  search_digests: keyword search over two weeks of digests    db.search_outputs()
  Layer 3  web_search / web_fetch, only for turns with Research on     tools.dispatch_tool()

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
    return name


def _router(research: bool, on_step: Optional[Callable[[str], None]]):
    labels = _labels()
    used = {"chars": 0}

    def route(name: str, args: dict, session_id: int) -> dict:
        args = args if isinstance(args, dict) else {}
        if on_step:
            on_step(step_label(name, args, labels))
        if name in _WEB:
            if not research:
                return {"error": "Research is off for this message. Tell the user they can turn it on to search the web."}
            return dispatch_tool(name, args, session_id)
        t0 = time.monotonic()
        try:
            if name == "read_digest":
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
    turns, q = [], None
    for m in msgs:
        if m["role"] == "user":
            q = m
        elif m["role"] == "assistant" and q:
            opened = [step_label(s["tool_name"], s["args"], labels) for s in steps if q["created_at"] <= s["created_at"] <= m["created_at"]]
            note = f"\n\n[Opened for this answer: {'; '.join(opened)}]" if opened else ""
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
    tools = DIGEST_TOOLS + (TOOLS if research else [])
    if on_step:
        on_step("thinking")
    answer = run_agent(client, history + [{"role": "user", "content": question}], chat_id,
                       tools=tools, system_prompt=system, cancel_event=cancel_event, think=False,
                       max_iterations=MAX_ITERATIONS, phase="chat", tool_router=_router(research, on_step))
    if not answer:
        db.log_message(chat_id, "error", "Cancelled." if cancel_event and cancel_event.is_set() else "No answer: the model call failed.")
    return answer
