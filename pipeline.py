import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests
from providers import LLMClient, MODEL, PROVIDER
from dotenv import load_dotenv

import db
from agent_config import SubQueryConfig, DomainConfig
from prompts import (
    PROMPT_PLANNER, PROMPT_RANKER, PROMPT_VALIDATOR,
    PROMPT_PAGE_SUMMARIZER, PROMPT_SYNTHESIZER,
    PROMPT_AGGREGATOR, _FRESHNESS_LABELS,
)
from tools import TOOLS, call_web_search, call_web_fetch, dispatch_tool

load_dotenv(override=True)

# ── Context budget — controls content limits for local vs cloud providers ─────
# CONTEXT_BUDGET=cloud (default): no limits, full content passed to model
# CONTEXT_BUDGET=local: moderate limits for Ollama on capable hardware
# CONTEXT_BUDGET=minimal: aggressive limits for Ollama on low-powered hardware
CONTEXT_BUDGET = os.getenv("CONTEXT_BUDGET", "cloud")

_BUDGET_SETTINGS: dict[str, dict] = {
    "cloud":   {"TOOL_RESULT_MAX_CHARS": None, "SYNTHESIZER_BUFFER_LIMIT": None, "SEARCH_RESULTS_PER_PROVIDER": 8},
    "local":   {"TOOL_RESULT_MAX_CHARS": 12000, "SYNTHESIZER_BUFFER_LIMIT": 800,  "SEARCH_RESULTS_PER_PROVIDER": 6},
    "minimal": {"TOOL_RESULT_MAX_CHARS": 6000,  "SYNTHESIZER_BUFFER_LIMIT": 400,  "SEARCH_RESULTS_PER_PROVIDER": 4},
}
_budget = _BUDGET_SETTINGS.get(CONTEXT_BUDGET, _BUDGET_SETTINGS["cloud"])

LOCATION = os.getenv("LOCATION", "")
MAX_TOOL_ITERATIONS     = int(os.getenv("MAX_ITERATIONS", "20"))
NUM_CTX                 = int(os.getenv("NUM_CTX", os.getenv("OLLAMA_NUM_CTX", "0"))) or None
TEMPERATURE             = float(os.getenv("TEMPERATURE", "0.2"))
OLLAMA_TIMEOUT          = int(os.getenv("TIMEOUT", "300"))
_SYNTHESIZER_CTX        = int(os.getenv("SYNTHESIZER_CTX", "0")) or None
_SYNTHESIZER_MAX_CHARS  = int(os.getenv("SYNTHESIZER_MAX_CHARS", "80000"))
_SYNTHESIZER_MAX_TOKENS = int(os.getenv("SYNTHESIZER_MAX_TOKENS", "0")) or None  # output cap; 0 = unlimited
_PARALLEL_SUB_PIPELINES = int(os.getenv("PARALLEL_SUB_PIPELINES", "1"))  # 0 = sequential, N = max workers

TOOL_RESULT_MAX_CHARS: Optional[int]     = _budget["TOOL_RESULT_MAX_CHARS"]
_SYNTHESIZER_BUFFER_LIMIT: Optional[int] = _budget["SYNTHESIZER_BUFFER_LIMIT"]
SEARCH_RESULTS_PER_PROVIDER: int         = _budget["SEARCH_RESULTS_PER_PROVIDER"]
PLANNER_QUERY_COUNT: int                 = int(os.getenv("PLANNER_QUERY_COUNT", "5"))

print(f"[config] provider={PROVIDER} budget={CONTEXT_BUDGET} model={MODEL} num_ctx={NUM_CTX} timeout={OLLAMA_TIMEOUT}s max_iter={MAX_TOOL_ITERATIONS}")

def _context_preamble() -> str:
    now = datetime.now()
    tz = time.strftime("%Z")
    date_str = now.strftime(f"%A, %B {now.day} %Y, %I:%M %p {tz}").replace(" 0", " ")
    lines = [f"Current date and time: {date_str}"]
    if LOCATION:
        lines.append(f"User location: {LOCATION}")
    return "\n".join(lines) + "\n\n"


def run_agent(
    client: LLMClient,
    messages: list,
    session_id: int,
    tools: Optional[list] = None,
    system_prompt: Optional[str] = None,
    cancel_event=None,
    think: bool = True,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    model: Optional[str] = None,
    log_response: bool = True,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
) -> Optional[str]:
    """Run the agentic loop. Returns the final answer, or None if max iterations hit or cancelled."""
    if tools is None:
        tools = TOOLS
    preamble = _context_preamble()
    if system_prompt:
        full_system = preamble + system_prompt
    else:
        full_system = preamble.strip()
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": full_system})
    # Native Ollama `think` (a "/no_think" prompt prefix only works for Qwen3; gemma4 ignores it).
    think_opt = None if think else False
    for _ in range(max_iterations):
        if cancel_event and cancel_event.is_set():
            return None
        try:
            response = client.chat(
                model=model or MODEL,
                messages=messages,
                tools=tools,
                temperature=TEMPERATURE,
                max_tokens=num_predict,
                num_ctx=num_ctx or NUM_CTX,
                think=think_opt,
            )
        except Exception as e:
            print(f"[agent] ERROR: model call failed ({type(e).__name__}): {e}")
            return None
        msg = response.message
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg._anthropic_blocks is not None:
            assistant_entry["_anthropic_blocks"] = msg._anthropic_blocks
        elif msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.tool_use_id or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.dumps(tc.function.arguments),
                    },
                }
                for i, tc in enumerate(msg.tool_calls)
            ]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            if log_response:
                db.log_message(session_id, "assistant", msg.content)
            if not msg.content:
                print(f"[agent] WARNING: model returned empty content (content={repr(msg.content)})")
            return msg.content

        for tc in msg.tool_calls:
            result = dispatch_tool(tc.function.name, tc.function.arguments, session_id)
            tool_content = json.dumps(result)
            if TOOL_RESULT_MAX_CHARS is not None and len(tool_content) > TOOL_RESULT_MAX_CHARS:
                tool_content = tool_content[:TOOL_RESULT_MAX_CHARS] + "…[truncated]"
            tool_entry: dict = {"role": "tool", "content": tool_content}
            if tc.tool_use_id:
                tool_entry["tool_use_id"] = tc.tool_use_id
            messages.append(tool_entry)

    print(f"[agent] WARNING: max_iterations ({max_iterations}) reached — forcing final response")
    messages.append({
        "role": "user",
        "content": "You've used your full tool budget. Summarize what you've found and give your best answer now. Do not call any more tools.",
    })
    try:
        resp = client.chat(
            model=model or MODEL,
            messages=messages,
            tools=[],
            temperature=TEMPERATURE,
            num_ctx=num_ctx or NUM_CTX,
            think=think_opt,
        )
        final = resp.message.content
        if final:
            if log_response:
                db.log_message(session_id, "assistant", final)
            return final
    except Exception as e:
        print(f"[agent] ERROR: forced final response failed ({type(e).__name__}): {e}")
    return None


def run_micro_agent(
    client: LLMClient,
    session_id: int,
    system_prompt: str,
    user_content: str,
    tools: Optional[list] = None,
    max_iterations: int = 1,
    cancel_event=None,
    model: Optional[str] = None,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
) -> Optional[str]:
    """Single-purpose agent call with a capped tool budget and no chain-of-thought overhead."""
    messages = [{"role": "user", "content": user_content}]
    return run_agent(
        client, messages, session_id,
        tools=tools or [],
        system_prompt=system_prompt,
        cancel_event=cancel_event,
        think=False,
        max_iterations=max_iterations,
        model=model,
        log_response=False,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )


def _parse_json_output(text: Optional[str], fallback: dict) -> dict:
    """Safely parse JSON object from a micro-agent response, stripping markdown fences if present."""
    if not text:
        return fallback
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(stripped)
        if isinstance(result, (dict, list)):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    # Extract the first [...] array from the text
    start = stripped.find('[')
    end = stripped.rfind(']')
    if start != -1 and end > start:
        try:
            result = json.loads(stripped[start:end + 1])
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    # Extract the first {...} object from the text
    start = stripped.find('{')
    end = stripped.rfind('}')
    if start != -1 and end > start:
        try:
            result = json.loads(stripped[start:end + 1])
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return fallback


class AgentPipeline:
    """
    Orchestrates a multi-phase micro-agent pipeline:
    Router → Planner → Searchers → Fetcher → Validator(loop) → Synthesizer

    The orchestrator (this class) owns all state. Each micro-agent gets a
    fresh context with no accumulated tool history.
    """

    def __init__(
        self,
        client: LLMClient,
        session_id: int,
        config: DomainConfig,
        cancel_event=None,
    ):
        self.client = client
        self.session_id = session_id
        self.config = config
        self.cancel_event = cancel_event
        self.research_buffer: list[dict] = []
        self.freshness: Optional[str] = config.freshness

    def run(self, query: str) -> Optional[str]:
        goal = self.config.goal_template.replace("{query}", query)

        print(f"[pipeline] model={MODEL}")

        # ── 1. Planner ────────────────────────────────────────────────────────
        if self.config.direct_sources:
            today = datetime.now().strftime("%Y-%m-%d")
            queries = [u.replace("{today}", today) for u in self.config.direct_sources]
            print(f"[pipeline] planner: {len(queries)} direct_sources (skipped planner)")
        else:
            print(f"[pipeline] planner: generating {self.config.planner_query_count} queries...")
            sources_hint = ""
            if self.config.sources:
                sources_hint = "\nKnown authoritative sources to target: " + ", ".join(self.config.sources)
            plan_prompt = PROMPT_PLANNER.format(
                n=self.config.planner_query_count,
                goal=goal,
                sources_hint=sources_hint,
            )
            _t0 = time.monotonic()
            queries_json = run_micro_agent(
                self.client, self.session_id,
                plan_prompt,
                goal,
                tools=[],
                max_iterations=1,
                cancel_event=self.cancel_event,
            )
            print(f"[pipeline] planner: {int((time.monotonic()-_t0)*1000)}ms")
            if queries_json is None:
                print("[pipeline] planner: WARNING: model returned no output — no queries will be generated")
            queries = _parse_json_output(queries_json, [])
            if isinstance(queries, dict):
                queries = list(queries.values())
            queries = [q for q in queries if isinstance(q, str)]
            print(f"[pipeline] planner: {len(queries)} queries generated")

        if not queries:
            print("[pipeline] planner: no queries generated, skipping research phases")

        # ── 2. Searcher loop ──────────────────────────────────────────────────
        if queries:
            self._run_searches(queries)

        # ── 2b. Link validator ────────────────────────────────────────────────
        if queries and self.config.validate_links:
            self._run_link_validator()

        # ── 2c. Fetcher (optional) ────────────────────────────────────────────
        if queries and self.config.fetch_content:
            self._run_fetcher(goal)
            self._run_page_summarizer(goal)

        # ── 4. Validator loop ─────────────────────────────────────────────────
        if queries:
            freshness_label = _FRESHNESS_LABELS.get(self.freshness or "", "any date")
            for loop_i in range(self.config.max_validator_loops):
                print(f"[pipeline] validator: loop {loop_i + 1}/{self.config.max_validator_loops}")
                _t0 = time.monotonic()
                val_out = run_micro_agent(
                    self.client, self.session_id,
                    PROMPT_VALIDATOR.format(goal=goal, freshness_label=freshness_label),
                    self._compact_buffer(),
                    tools=[],
                    max_iterations=1,
                    cancel_event=self.cancel_event,
                )
                print(f"[pipeline] validator: {int((time.monotonic()-_t0)*1000)}ms")
                if val_out is None:
                    print("[pipeline] validator: WARNING: model returned no output — assuming sufficient")
                verdict = _parse_json_output(val_out, {"sufficient": True})
                print(f"[pipeline] validator: sufficient={verdict.get('sufficient')}, notes={verdict.get('notes', '')[:80]}")
                if verdict.get("sufficient", True):
                    break
                retry_queries = [q for q in verdict.get("retry_queries", []) if isinstance(q, str)]
                if not retry_queries:
                    break
                print(f"[pipeline] validator: retrying with {len(retry_queries)} additional queries: {retry_queries}")
                self._run_searches(retry_queries)
                if self.config.validate_links:
                    self._run_link_validator()
                if self.config.fetch_content:
                    self._run_fetcher(goal)
                    self._run_page_summarizer(goal)

        # ── 5. Synthesizer ────────────────────────────────────────────────────
        print("[pipeline] synthesizer: formatting final output...")
        if not self.research_buffer:
            print("[pipeline] synthesizer: no research collected — aborting to prevent hallucination")
            return None
        synth_prompt = PROMPT_SYNTHESIZER.format(output_template=self.config.output_template)
        has_summaries = any(e.get("phase") == "summaries" for e in self.research_buffer)
        has_fetch = any(e.get("phase") == "fetch" for e in self.research_buffer)
        search_limit = 0 if (has_summaries or has_fetch) else _SYNTHESIZER_BUFFER_LIMIT
        synth_input = self._compact_buffer(search_content_limit=search_limit)
        print(f"[pipeline] synthesizer: input={len(synth_input)} chars")
        if len(synth_input) > _SYNTHESIZER_MAX_CHARS and search_limit != 0:
            for _cap in (200, 100, 50):
                synth_input = self._compact_buffer(search_content_limit=_cap)
                print(f"[pipeline] synthesizer: trimmed to {len(synth_input)} chars (cap={_cap})")
                if len(synth_input) <= _SYNTHESIZER_MAX_CHARS:
                    break
            else:
                synth_input = synth_input[:_SYNTHESIZER_MAX_CHARS] + "\n[TRUNCATED: input exceeded synthesizer limit]"
                print(f"[pipeline] synthesizer: hard-truncated to {_SYNTHESIZER_MAX_CHARS} chars")
        _t0 = time.monotonic()
        result = run_micro_agent(
            self.client, self.session_id,
            synth_prompt,
            synth_input,
            tools=[],
            max_iterations=1,
            cancel_event=self.cancel_event,
            num_ctx=_SYNTHESIZER_CTX,
            num_predict=_SYNTHESIZER_MAX_TOKENS,
        )
        print(f"[pipeline] synthesizer: {int((time.monotonic()-_t0)*1000)}ms")
        if not result:
            print("[pipeline] synthesizer: WARNING: model returned no output")
            return None
        return result

    def _run_searches(self, queries: list[str]) -> None:
        """Fire all search queries in parallel — pure HTTP, no Ollama inference."""
        if not queries:
            return
        print(f"[pipeline] searcher: firing {len(queries)} queries in parallel")

        def _search(q: str) -> tuple[str, dict]:
            allowed = self.config.tool_bindings  # empty list = allow all tools
            if q.startswith("http://") or q.startswith("https://"):
                if allowed and "web_fetch" not in allowed:
                    return q, {}
                return q, call_web_fetch(q)
            if allowed and "web_search" not in allowed:
                return q, {}
            return q, call_web_search(q, freshness=self.freshness)

        all_results = []
        all_source_errors: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(len(queries), 8)) as pool:
            futures = {pool.submit(_search, q): q for q in queries}
            for future in as_completed(futures):
                q, result = future.result()
                if result and not result.get("error"):
                    if result.get("source_errors"):
                        all_source_errors[q] = result["source_errors"]
                    if "results" in result:
                        trimmed = [
                            {
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "content": r.get("content", "")[:1500],
                            }
                            for r in (result.get("results") or [])
                        ]
                    else:
                        content = result.get("content", "")
                        trimmed = [{"title": q, "url": q, "content": content[:10000]}] if content else []
                    if trimmed:
                        all_results.append({"query": q, "results": trimmed})

        if all_results:
            entry: dict = {"phase": "search", "content": all_results}
            if all_source_errors:
                entry["source_errors"] = all_source_errors
            self.research_buffer.append(entry)

    def _run_link_validator(self) -> None:
        """HEAD-check all search result URLs in parallel; drop dead ones from the buffer."""
        all_urls: set[str] = set()
        for entry in self.research_buffer:
            if entry.get("phase") == "search":
                for item in entry.get("content", []):
                    for r in item.get("results", []):
                        url = r.get("url", "")
                        if url:
                            all_urls.add(url)

        if not all_urls:
            return

        print(f"[pipeline] link-validator: checking {len(all_urls)} URLs in parallel")
        TIMEOUT = 3

        def _check(url: str) -> tuple[str, bool]:
            try:
                resp = requests.head(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; local-web-search/1.0)"},
                    timeout=TIMEOUT,
                    allow_redirects=True,
                )
                if resp.status_code == 405:
                    return url, True  # HEAD not allowed but URL exists
                if resp.status_code == 403:
                    # Some sites block HEAD but serve GET — treat as alive
                    return url, True
                return url, resp.status_code < 400
            except Exception:
                return url, False

        dead: set[str] = set()
        with ThreadPoolExecutor(max_workers=min(len(all_urls), 16)) as pool:
            futures = {pool.submit(_check, u): u for u in all_urls}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    _, ok = future.result()
                    if not ok:
                        dead.add(url)
                except Exception:
                    dead.add(url)

        if not dead:
            print("[pipeline] link-validator: all URLs live")
            return

        print(f"[pipeline] link-validator: removing {len(dead)} dead URLs: {dead}")
        for entry in self.research_buffer:
            if entry.get("phase") != "search":
                continue
            for item in entry.get("content", []):
                item["results"] = [r for r in item.get("results", []) if r.get("url", "") not in dead]

    def _run_fetcher(self, goal: str) -> None:
        """Rank URLs with one Ollama call, then fetch the top picks in parallel."""
        url_list = self._extract_titles_and_urls()
        if not url_list:
            print("[pipeline] fetcher: no URLs to fetch")
            return

        print(f"[pipeline] ranker: selecting from {len(url_list)} candidate URLs")
        rank_out = run_micro_agent(
            self.client, self.session_id,
            PROMPT_RANKER.format(goal=goal),
            json.dumps(url_list),
            tools=[],
            max_iterations=1,
            cancel_event=self.cancel_event,
        )
        ranked_urls = _parse_json_output(rank_out, [])
        if isinstance(ranked_urls, dict):
            ranked_urls = list(ranked_urls.values())
        ranked_urls = list(dict.fromkeys(u for u in ranked_urls if isinstance(u, str)))[:5]

        if rank_out is None:
            print("[pipeline] ranker: WARNING: model call failed — skipping fetch phase")
            return
        if not ranked_urls:
            print("[pipeline] ranker: no URLs selected")
            return

        print(f"[pipeline] fetcher: fetching {len(ranked_urls)} URLs in parallel")

        def _fetch(url: str) -> tuple[str, dict]:
            print(f"[web_fetch]  Fetching: {url}")
            return url, call_web_fetch(url)

        FETCH_TIMEOUT = 15
        fetched_pages = []
        with ThreadPoolExecutor(max_workers=len(ranked_urls)) as pool:
            futures = {pool.submit(_fetch, u): u for u in ranked_urls}
            for future in as_completed(futures, timeout=FETCH_TIMEOUT + 5):
                url = futures[future]
                try:
                    _, result = future.result(timeout=FETCH_TIMEOUT)
                    if result and not result.get("error"):
                        content = result.get("content", "") or result.get("text", "")
                        fetched_pages.append({"url": url, "content": content[:4000]})
                except Exception as e:
                    print(f"[pipeline] fetcher: skipping {url} ({type(e).__name__}: {e})")

        if fetched_pages:
            self.research_buffer.append({"phase": "fetch", "content": fetched_pages})

    def _run_page_summarizer(self, goal: str) -> None:
        """Distill each fetched page into goal-relevant facts via parallel fast-model calls."""
        pages = [
            p
            for e in self.research_buffer if e.get("phase") == "fetch"
            for p in e.get("content", [])
        ]
        if not pages:
            return

        print(f"[pipeline] page-summarizer: summarizing {len(pages)} pages in parallel")

        def _summarize(page: dict) -> Optional[dict]:
            url = page.get("url", "")
            content = page.get("content", "")
            out = run_micro_agent(
                self.client, self.session_id,
                PROMPT_PAGE_SUMMARIZER.format(goal=goal, url=url),
                content,
                tools=[],
                max_iterations=1,
                cancel_event=self.cancel_event,
            )
            parsed = _parse_json_output(out, {"url": url, "relevant": False, "facts": []})
            parsed["url"] = url
            return parsed if parsed.get("relevant") and parsed.get("facts") else None

        with ThreadPoolExecutor(max_workers=len(pages)) as pool:
            results = list(pool.map(_summarize, pages))

        summaries = [r for r in results if r]
        print(f"[pipeline] page-summarizer: {len(summaries)}/{len(pages)} pages relevant")
        if summaries:
            self.research_buffer.append({"phase": "summaries", "content": summaries})

    def _extract_titles_and_urls(self) -> list[dict]:
        """Pull {title, url} pairs from structured search results in the buffer."""
        pairs = []
        seen: set[str] = set()
        for entry in self.research_buffer:
            if entry.get("phase") != "search":
                continue
            for item in entry.get("content", []):
                for r in item.get("results", []):
                    url = r.get("url", "")
                    title = r.get("title", url)
                    if url and url not in seen:
                        seen.add(url)
                        content = r.get("content", "")
                        pairs.append({"title": title, "url": url, "snippet": content[:500]})
        return pairs[:30]

    def _compact_buffer(self, search_content_limit: Optional[int] = 300, fetch_content_limit: Optional[int] = None) -> str:
        """Return a size-capped JSON representation of the buffer for micro-agent input.

        When a summaries phase is present it supersedes the raw fetch phase.
        """
        has_summaries = any(e.get("phase") == "summaries" for e in self.research_buffer)
        compact = []
        all_source_errors: dict[str, dict] = {}
        for entry in self.research_buffer:
            phase = entry.get("phase")
            if phase == "search":
                compacted_items = []
                for item in entry.get("content", []):
                    compacted_results = [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            **({"content": r.get("content", "") if search_content_limit is None else r.get("content", "")[:search_content_limit]} if search_content_limit != 0 else {}),
                        }
                        for r in item.get("results", [])
                    ]
                    compacted_items.append({"query": item.get("query", ""), "results": compacted_results})
                compact.append({"phase": "search", "content": compacted_items})
                if entry.get("source_errors"):
                    all_source_errors.update(entry["source_errors"])
            elif phase == "fetch":
                if has_summaries:
                    continue  # summaries phase supersedes raw fetch
                if fetch_content_limit is not None:
                    capped = [
                        {"url": p.get("url", ""), "content": p.get("content", "")[:fetch_content_limit]}
                        for p in entry.get("content", [])
                    ]
                    compact.append({"phase": "fetch", "content": capped})
                else:
                    compact.append(entry)
            elif phase == "summaries":
                compact.append(entry)
            else:
                compact.append(entry)

        output = json.dumps(compact, indent=2)
        if all_source_errors:
            error_counts: dict[str, list[str]] = {}
            for errs in all_source_errors.values():
                for source, msg in errs.items():
                    error_counts.setdefault(source, []).append(msg)
            total_queries = len(all_source_errors)
            lines = [f"\n[SOURCE ERRORS] The following search sources failed for some queries:"]
            for source, msgs in sorted(error_counts.items()):
                lines.append(f"- {source}: failed on {len(msgs)}/{total_queries} queries ({msgs[0]})")
            output += "\n" + "\n".join(lines)
        return output

    def _buffer_summary(self) -> str:
        return json.dumps(self.research_buffer, indent=2)


def create_pipeline(
    client: LLMClient,
    session_id: int,
    config: DomainConfig,
    cancel_event=None,
) -> "AgentPipeline | MultiAgentPipeline":
    """Return MultiAgentPipeline when sub_queries are defined, AgentPipeline otherwise."""
    if config.sub_queries:
        return MultiAgentPipeline(client, session_id, config, cancel_event)
    return AgentPipeline(client, session_id, config, cancel_event)


class MultiAgentPipeline:
    """
    Runs one AgentPipeline per SubQueryConfig in parallel, then aggregates
    the labeled sub-results into a final response with a single LLM call.

    Sub-pipelines share the parent session_id so all tool calls appear in
    the same session. Recall is handled once at the top level.
    """

    def __init__(
        self,
        client: LLMClient,
        session_id: int,
        config: DomainConfig,
        cancel_event=None,
    ):
        self.client = client
        self.session_id = session_id
        self.config = config
        self.cancel_event = cancel_event

    def run(self, query: str) -> Optional[str]:
        prior_lines: list[str] = []
        sub_results: dict[str, str] = {}

        def _run_sub(sq: SubQueryConfig) -> tuple[str, Optional[str]]:
            print(f"[multi-pipeline] starting sub-pipeline: {sq.label}")
            resolved_goal = sq.goal.replace("{query}", query)
            sub_config = DomainConfig(
                agent_name=f"{self.config.agent_name}:{sq.label}",
                goal_template=resolved_goal,
                output_template=(
                    f"List the top 5 findings for: {sq.label}. "
                    "One item per line: headline — source URL. No extra commentary."
                ),
                sources=sq.sources,
                max_validator_loops=self.config.max_validator_loops,
                planner_query_count=min(sq.query_count, PLANNER_QUERY_COUNT),
                validate_links=self.config.validate_links,
                freshness=sq.freshness or self.config.freshness,
            )
            sub_pipeline = AgentPipeline(self.client, self.session_id, sub_config, self.cancel_event)
            result = sub_pipeline.run(resolved_goal)
            print(f"[multi-pipeline] sub-pipeline done: {sq.label}")
            return sq.label, result

        if _PARALLEL_SUB_PIPELINES == 0:
            print(f"[multi-pipeline] running {len(self.config.sub_queries)} sub-pipelines sequentially")
            for sq in self.config.sub_queries:
                label, result = _run_sub(sq)
                if result:
                    sub_results[label] = result
        else:
            workers = _PARALLEL_SUB_PIPELINES if _PARALLEL_SUB_PIPELINES > 1 else len(self.config.sub_queries)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_sub, sq): sq for sq in self.config.sub_queries}
                for future in as_completed(futures):
                    label, result = future.result()
                    if result:
                        sub_results[label] = result

        if not sub_results:
            return None

        print(f"[multi-pipeline] aggregator: combining {len(sub_results)} sub-results...")
        labeled_parts = "\n\n".join(
            f"=== {label} ===\n{result}" for label, result in sub_results.items()
        )
        agg_input = "\n\n".join(prior_lines + [labeled_parts])

        return run_micro_agent(
            self.client, self.session_id,
            PROMPT_AGGREGATOR.format(output_template=self.config.output_template),
            agg_input,
            tools=[],
            max_iterations=1,
            cancel_event=self.cancel_event,
        )
