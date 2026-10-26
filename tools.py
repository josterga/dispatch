import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
import trafilatura
from dotenv import load_dotenv
from ddgs import DDGS

import db

load_dotenv(override=True)

SEARCH_RESULTS_PER_PROVIDER = int(os.getenv("OLLAMA_SEARCH_RESULTS_PER_PROVIDER", "5"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information about a topic. "
                "Use this when you need up-to-date facts, news, or any "
                "information you are uncertain about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch and read the full content of a specific web page. "
                "Use this after web_search to get detailed information from "
                "a particular URL that appeared in search results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the page to fetch",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

_ollama_failure_until: float = 0.0

def _ollama_search(query: str) -> tuple[Optional[list], Optional[str]]:
    global _ollama_failure_until
    try:
        r = requests.post(
            "https://ollama.com/api/web_search",
            json={"query": query, "max_results": SEARCH_RESULTS_PER_PROVIDER},
            headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}", "Content-Type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        if not r.text.strip():
            print("[ollama_search] empty body — cooling down for 60s")
            _ollama_failure_until = time.time() + 60
            return None, "empty response"
        data = r.json()
        results = data.get("results", [])
        print(f"[ollama_search] {len(results)} results")
        return results, None
    except requests.exceptions.HTTPError as e:
        msg = "Invalid OLLAMA_API_KEY" if e.response.status_code == 401 else str(e)
        print(f"[ollama_search] ERROR: {msg}")
        _ollama_failure_until = time.time() + 60
        return None, msg
    except requests.exceptions.Timeout:
        print("[ollama_search] ERROR: timed out — cooling down for 60s")
        _ollama_failure_until = time.time() + 60
        return None, "timed out"
    except Exception as e:
        print(f"[ollama_search] ERROR: {e}")
        _ollama_failure_until = time.time() + 60
        return None, str(e)


_BRAVE_FRESHNESS = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}


def _brave_search(query: str, freshness: Optional[str] = None) -> tuple[Optional[list], Optional[str]]:
    brave_key = os.getenv("BRAVE_API_KEY", "")
    if not brave_key:
        return None, None
    try:
        params: dict = {"q": query, "count": SEARCH_RESULTS_PER_PROVIDER}
        if freshness and freshness in _BRAVE_FRESHNESS:
            params["freshness"] = _BRAVE_FRESHNESS[freshness]
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers={"Accept": "application/json", "X-Subscription-Token": brave_key},
            timeout=10,
        )
        r.raise_for_status()
        hits = r.json().get("web", {}).get("results", [])
        results = [{"title": h.get("title", ""), "url": h.get("url", ""), "content": h.get("description", "")} for h in hits]
        print(f"[brave_search] {len(results)} results")
        return results, None
    except Exception as e:
        print(f"[brave_search] ERROR: {e}")
        return None, str(e)


_SEARXNG_TIME_RANGE = {"d": "day", "w": "week", "m": "month", "y": "year"}


def _searxng_search(query: str, freshness: Optional[str] = None) -> tuple[Optional[list], Optional[str]]:
    searxng_url = os.getenv("SEARXNG_URL", "").rstrip("/")
    if not searxng_url:
        return None, None
    try:
        params: dict = {"q": query, "format": "json"}
        if freshness and freshness in _SEARXNG_TIME_RANGE:
            params["time_range"] = _SEARXNG_TIME_RANGE[freshness]
        r = requests.get(f"{searxng_url}/search", params=params, timeout=15)
        r.raise_for_status()
        hits = r.json().get("results", [])
        results = [{"title": h.get("title", ""), "url": h.get("url", ""), "content": h.get("content", "")} for h in hits][:SEARCH_RESULTS_PER_PROVIDER]
        print(f"[searxng_search] {len(results)} results")
        return results, None
    except Exception as e:
        print(f"[searxng_search] ERROR: {e}")
        return None, str(e)


def _ddg_search(query: str, freshness: Optional[str] = None) -> tuple[Optional[list], Optional[str]]:
    try:
        kwargs: dict = {"max_results": SEARCH_RESULTS_PER_PROVIDER}
        if freshness in ("d", "w", "m", "y"):
            kwargs["timelimit"] = freshness
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, **kwargs))
        results = [{"title": h.get("title", ""), "url": h.get("href", ""), "content": h.get("body", "")} for h in hits]
        print(f"[ddg_search] {len(results)} results")
        return results, None
    except Exception as e:
        print(f"[ddg_search] ERROR: {e}")
        return None, str(e)


def _timed(fn, *args) -> tuple[Optional[list], Optional[str], int]:
    t0 = time.monotonic()
    results, err = fn(*args)
    return results, err, int((time.monotonic() - t0) * 1000)


def call_web_search(query: str, freshness: Optional[str] = None) -> dict:
    """Search all configured providers in parallel and merge by URL.

    The result's "meta" (per-provider status/timing/count, every merged hit with its rank in each
    provider) is for run metrics only: callers pop it before handing results to a model.
    """
    print(f'[web_search] Searching for: "{query}"' + (f" [freshness={freshness}]" if freshness else ""))
    t0 = time.monotonic()
    ollama_ready = time.time() >= _ollama_failure_until

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "searxng": pool.submit(_timed, _searxng_search, query, freshness),
            "brave": pool.submit(_timed, _brave_search, query, freshness),
        }
        if ollama_ready:
            futures["ollama"] = pool.submit(_timed, _ollama_search, query)

    answers = {name: f.result() for name, f in futures.items()}
    answered = any(res for res, _, _ in answers.values())

    # DuckDuckGo scraping is often throttled ("No results found"): only use it when nothing else answered.
    if not answered:
        answers["ddg"] = _timed(_ddg_search, query, freshness)

    order = ("searxng", "ollama", "brave", "ddg")
    meta_providers: dict[str, dict] = {}
    for name in order:
        if name not in answers:
            status = "cooldown" if name == "ollama" else "not_needed" if name == "ddg" else "off"
            meta_providers[name] = {"status": status}
            continue
        res, err, ms = answers[name]
        if err:
            meta_providers[name] = {"status": "error", "ms": ms, "error": err[:300]}
        elif res is None:
            meta_providers[name] = {"status": "off"}
        else:
            meta_providers[name] = {"status": "ok", "ms": ms, "count": len(res)}

    seen: dict[str, dict] = {}
    merged = []
    raw = 0
    for name in order:
        res = answers.get(name, (None, None, 0))[0] or []
        for rank, result in enumerate(res, 1):
            url = result.get("url", "")
            if not url:
                continue
            raw += 1
            if url in seen:
                seen[url]["by"][name] = rank
                continue
            seen[url] = {"url": url, "title": (result.get("title") or "")[:200], "by": {name: rank}}
            merged.append(result)
    meta = {
        "providers": meta_providers,
        "hits": [dict(seen[r["url"]], rank=i) for i, r in enumerate(merged, 1)],
        "raw": raw,
        "dupes": raw - len(merged),
        "ddg_fallback": "ddg" in answers,
        "ms": int((time.monotonic() - t0) * 1000),
        "freshness": freshness,
    }

    if not merged:
        return {"error": "All search providers failed", "meta": meta}

    counts = ", ".join(f"{n}={len(answers.get(n, (None,))[0] or [])}" for n in order)
    print(f"[web_search] {len(merged)} merged results ({counts})")

    source_errors = {n: p["error"] for n, p in meta_providers.items() if p.get("error")}
    return {"results": merged, **({"source_errors": source_errors} if source_errors else {}), "meta": meta}


THIN_CONTENT_CHARS = 3000  # below this the page is likely a JS shell or boilerplate (flagged in metrics)


def _is_local_url(url: str) -> bool:
    """localhost / private / link-local hosts (e.g. the intel inbox): recorded in fetch metrics."""
    import ipaddress
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def call_web_fetch(url: str) -> dict:
    """Fetch a page as text, locally (no cloud reader). The result's "meta" (HTTP status, sizes, timings) is for
    run metrics only: callers pop it before handing the content to a model."""
    print(f"[web_fetch]  Fetching: {url}")
    t0 = time.monotonic()
    local = _is_local_url(url)
    meta: dict = {"local": local}
    result = _fetch(url, local, meta)
    content = result.get("content") or ""
    meta["chars"] = len(content)
    meta["ms"] = int((time.monotonic() - t0) * 1000)
    result["meta"] = meta
    return result


def _html_text(html: str, url: str, meta: dict) -> str:
    """Main content via trafilatura (drops nav/footer boilerplate); regex tag strip if it finds nothing."""
    try:
        text = trafilatura.extract(html, url=url, favor_recall=True, include_comments=False) or ""
    except Exception as e:
        print(f"[web_fetch]  trafilatura ERROR: {e}")
        text = ""
    if text.strip():
        meta["extract"] = "trafilatura"
        return text.strip()
    meta["extract"] = "regex"
    text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch(url: str, local: bool, meta: dict) -> dict:
    meta["via"] = "http"
    try:
        t0 = time.monotonic()
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; local-web-search/1.0)"},
            timeout=12,
            allow_redirects=True,
        )
        meta.update(status=r.status_code, bytes=len(r.content), ctype=r.headers.get("content-type", ""),
                    http_ms=int((time.monotonic() - t0) * 1000), redirects=len(r.history) or None)
        if r.url != url:
            meta["final_url"] = r.url
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "json" in ctype:
            return {"content": r.text}
        if ctype.startswith("text/plain"):
            # Keep line structure (e.g. the local intel inbox's item blocks).
            text = r.text.strip()
        else:
            text = _html_text(r.text, url, meta)
        print(f"[web_fetch]  content: {len(text)} chars, snippet: {text[:200]!r}")
        if len(text) < THIN_CONTENT_CHARS and not local:
            meta["thin"] = True
        return {"content": text}
    except requests.exceptions.HTTPError as e:
        print(f"[web_fetch] ERROR: {e}")
        meta["err"] = str(e)[:300]
        return {"error": str(e)}
    except requests.exceptions.Timeout:
        print("[web_fetch] ERROR: Request timed out")
        meta["err"] = "timeout"
        return {"error": "Web fetch timed out"}
    except Exception as e:
        print(f"[web_fetch] ERROR: {e}")
        meta["err"] = f"{type(e).__name__}: {e}"[:300]
        return {"error": str(e)}


def dispatch_tool(name: str, args: dict, session_id: int) -> dict:
    t0 = time.monotonic()
    if name == "web_search":
        result = call_web_search(args["query"])
    elif name == "web_fetch":
        result = call_web_fetch(args["url"])
    else:
        print(f"[tool] Unknown tool: {name}")
        result = {"error": f"Unknown tool: {name}"}
    duration_ms = int((time.monotonic() - t0) * 1000)
    meta = result.pop("meta", None)
    db.log_tool_call(session_id, name, args, result, duration_ms)
    if meta is not None:
        import metrics
        if name == "web_search":
            metrics.get(session_id).event("search", None, args["query"], start=t0, ok="results" in result,
                                          error=result.get("error"), round=0, via="agent_tool", **meta)
        else:
            metrics.get(session_id).event("fetch", None, args["url"], start=t0, ok="error" not in result,
                                          error=result.get("error"), source="agent_tool", **meta)
    return result
