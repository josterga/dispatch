import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from dotenv import load_dotenv
from ddgs import DDGS

import db

load_dotenv(override=True)

SEARCH_RESULTS_PER_PROVIDER = int(os.getenv("OLLAMA_SEARCH_RESULTS_PER_PROVIDER", "5"))
WEB_FETCH_PROVIDER = os.getenv("WEB_FETCH_PROVIDER", "http")  # "http" or "jina"

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


def call_web_search(query: str, freshness: Optional[str] = None) -> dict:
    print(f'[web_search] Searching for: "{query}"' + (f" [freshness={freshness}]" if freshness else ""))
    ollama_ready = time.time() >= _ollama_failure_until

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "searxng": pool.submit(_searxng_search, query, freshness),
            "brave": pool.submit(_brave_search, query, freshness),
            "jina":  pool.submit(_jina_search, query, freshness),
        }
        if ollama_ready:
            futures["ollama"] = pool.submit(_ollama_search, query)

    ollama_results, ollama_err = futures["ollama"].result() if "ollama" in futures else (None, None)
    searxng_results, searxng_err = futures["searxng"].result()
    brave_results,  brave_err  = futures["brave"].result()
    jina_results,   jina_err   = futures["jina"].result()

    # DuckDuckGo scraping is often throttled ("No results found"): only use it when nothing else answered.
    ddg_results, ddg_err = None, None
    if not (searxng_results or ollama_results or brave_results or jina_results):
        ddg_results, ddg_err = _ddg_search(query, freshness)

    seen_urls = set()
    merged = []
    for result in (searxng_results or []) + (ollama_results or []) + (brave_results or []) + (jina_results or []) + (ddg_results or []):
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged.append(result)

    if not merged:
        return {"error": "All search providers failed"}

    print(f"[web_search] {len(merged)} merged results (searxng={len(searxng_results or [])}, ollama={len(ollama_results or [])}, brave={len(brave_results or [])}, jina={len(jina_results or [])}, ddg={len(ddg_results or [])})")

    source_errors = {k: v for k, v in {"searxng": searxng_err, "ollama": ollama_err, "brave": brave_err, "jina": jina_err, "ddg": ddg_err}.items() if v}
    return {"results": merged, **({"source_errors": source_errors} if source_errors else {})}


JINA_FALLBACK_THRESHOLD = 3000  # chars — below this, assume thin/boilerplate content and try Jina


def _jina_fetch(url: str) -> dict:
    jina_url = f"https://r.jina.ai/{url}"
    print(f"[web_fetch]  Jina: {jina_url}")
    try:
        jina_key = os.getenv("JINA_API_KEY", "")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; local-web-search/1.0)", "Accept": "text/plain"}
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"
        r = requests.get(
            jina_url,
            headers=headers,
            timeout=20,
            allow_redirects=True,
        )
        r.raise_for_status()
        text = r.text.strip()
        print(f"[web_fetch]  Jina content: {len(text)} chars")
        return {"content": text}
    except Exception as e:
        print(f"[web_fetch]  Jina ERROR: {e}")
        return {"error": str(e)}


def _jina_search(query: str, freshness: Optional[str] = None) -> tuple[Optional[list], Optional[str]]:
    jina_key = os.getenv("JINA_API_KEY", "")
    if not jina_key or "site:" in query:
        return None, None
    try:
        import urllib.parse
        url = f"https://s.jina.ai/{urllib.parse.quote_plus(query)}"
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {jina_key}", "Accept": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        hits = r.json().get("data", [])
        results = [
            {
                "title": h.get("title", ""),
                "url": h.get("url", ""),
                "content": h.get("description", "") or h.get("content", ""),
            }
            for h in hits
        ][:SEARCH_RESULTS_PER_PROVIDER]
        print(f"[jina_search] {len(results)} results")
        return results, None
    except Exception as e:
        print(f"[jina_search] ERROR: {e}")
        return None, str(e)


def _is_local_url(url: str) -> bool:
    """localhost / private / link-local hosts: never hand these to Jina (external, can't reach them anyway)."""
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
    print(f"[web_fetch]  Fetching: {url}")
    local = _is_local_url(url)
    if WEB_FETCH_PROVIDER == "jina" and not local:
        return _jina_fetch(url)
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; local-web-search/1.0)"},
            timeout=12,
            allow_redirects=True,
        )
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "json" in ctype:
            return {"content": r.text}
        if ctype.startswith("text/plain"):
            # Keep line structure (e.g. the local intel inbox's item blocks).
            text = r.text.strip()
        else:
            text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", r.text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        print(f"[web_fetch]  content: {len(text)} chars, snippet: {text[:200]!r}")
        if len(text) < JINA_FALLBACK_THRESHOLD and not local:
            return _jina_fetch(url)
        return {"content": text}
    except requests.exceptions.HTTPError as e:
        print(f"[web_fetch] ERROR: {e}")
        return {"error": str(e)}
    except requests.exceptions.Timeout:
        print("[web_fetch] ERROR: Request timed out")
        return {"error": "Web fetch timed out"}
    except Exception as e:
        print(f"[web_fetch] ERROR: {e}")
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
    db.log_tool_call(session_id, name, args, result, duration_ms)
    return result
