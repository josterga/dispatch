"""Reading layer over past digests, shared by previous-run awareness (pipeline) and digest chat.

  catalog()       Layer 0: one block per dispatch (label, agent, description, tags, schedule, latest run + headings)
  outline()       a digest cut down to headings + short lines, URLs kept whole
  urls()          URLs a digest cites
  run_sources()   Layer 2: what a run read (titles, URLs, snippets), from its synthesizer payloads
"""
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import db

_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+")
_TRAILING = ".,;:!?*_"


def urls(text: Optional[str]) -> list[str]:
    """Cited URLs in order of first appearance, trailing punctuation stripped."""
    out: list[str] = []
    for m in _URL_RE.findall(text or ""):
        u = m.rstrip(_TRAILING)
        if u not in out:
            out.append(u)
    return out


def headings(text: Optional[str]) -> list[str]:
    return [l.lstrip("#").strip() for l in (text or "").splitlines() if re.match(r"#{1,3}\s", l)]


def _clip_line(line: str, width: int) -> str:
    """Cut prose to `width` chars but never inside a URL: URLs found in the line are appended whole."""
    if len(line) <= width:
        return line
    found = urls(line)
    head = line[:width].rstrip()
    # Drop a URL cut in half at the end of the clipped prose; it's re-added whole below.
    head = re.sub(r"\(?https?://\S*$", "", head).rstrip()
    missing = [u for u in found if u not in head]
    return head + "…" + ("" if not missing else " " + " ".join(missing))


def outline(text: Optional[str], max_chars: int = 3000, line_width: int = 140) -> str:
    """Headings + list/prose lines clipped to `line_width` (URLs intact); stops at max_chars at a line boundary."""
    out, size = [], 0
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        line = line if re.match(r"#{1,6}\s", line) else _clip_line(line, line_width)
        if size + len(line) + 1 > max_chars:
            out.append("[…]")
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)


def _local(ts: Optional[str]) -> str:
    if not ts:
        return "—"
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone().strftime("%a %b %d %H:%M").replace(" 0", " ")
    except ValueError:
        return ts


def _tags(v) -> list[str]:
    try:
        t = json.loads(v) if isinstance(v, str) else (v or [])
        return [str(x) for x in t] if isinstance(t, list) else []
    except ValueError:
        return []


def dispatch_tags(d: dict, agent: Optional[dict]) -> list[str]:
    out: list[str] = []
    for t in _tags((agent or {}).get("tags")) + _tags(d.get("tags")):
        if t not in out:
            out.append(t)
    return out


def catalog(days: int = 7, max_headings: int = 8) -> str:
    """Layer 0. Active dispatches first, then archived ones that ran in the last `days`."""
    agents = {a["name"]: a for a in db.get_agents()}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    blocks_active, blocks_archived = [], []
    for d in db.get_dispatches():
        latest = db.recent_outputs(d["id"], 1)
        s = latest[0] if latest else None
        if not d["enabled"] and not (s and s["started_at"] >= since):
            continue
        a = agents.get(d["agent"]) or {}
        tags = dispatch_tags(d, a)
        lines = [
            f"[dispatch {d['id']}] {d.get('label') or a.get('label') or d['agent']}"
            + ("" if d["enabled"] else " (archived)"),
            f"  agent: {a.get('label') or d['agent']} ({d['agent']})" + (f" — {a['description'][:160]}" if a.get("description") else ""),
            f"  tags: {', '.join(tags) or '—'} · schedule: cron `{d['schedule']}` {d.get('timezone') or ''}".rstrip(),
        ]
        if s:
            hs = headings(s["content"])
            lines.append(f"  latest run {s['id']} at {_local(s['started_at'])}"
                         + (f": {' | '.join(hs[:max_headings])}" if hs else ""))
        else:
            lines.append("  no runs yet")
        (blocks_active if d["enabled"] else blocks_archived).append("\n".join(lines))
    return "\n\n".join(blocks_active + blocks_archived)


# Inbox pages (Direct Fetch): "## [category] title\nsource: … · when\nurl: U\nbody"
_INBOX_ITEM = re.compile(r"^## \[(?P<cat>[^\]]*)\] (?P<title>.*)\nsource: (?P<src>.*)\nurl: (?P<url>\S*)\n(?P<body>(?:(?!^## \[).*\n?)*)", re.M)


def _sources_from_buffer(buf: list, snippet: int) -> list[dict]:
    out: list[dict] = []
    for entry in buf if isinstance(buf, list) else []:
        phase = entry.get("phase")
        for item in entry.get("content", []) or []:
            if phase == "search":
                for r in item.get("results", []):
                    out.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": (r.get("content") or "")[:snippet], "via": "search"})
            elif phase == "direct":
                text = item.get("content", "")
                found = list(_INBOX_ITEM.finditer(text))
                if found:
                    for m in found:
                        out.append({"title": m["title"].strip(), "url": m["url"], "snippet": m["body"].strip()[:snippet],
                                    "via": f"inbox · {m['src'].strip()}"})
                else:
                    out.append({"title": "", "url": item.get("url", ""), "snippet": text[:snippet], "via": "direct"})
            elif phase == "fetch":
                out.append({"title": "", "url": item.get("url", ""), "snippet": (item.get("content") or "")[:snippet], "via": "fetch"})
            elif phase == "summaries":
                facts = item.get("facts") or []
                out.append({"title": item.get("title", ""), "url": item.get("url", ""),
                            "snippet": "; ".join(str(f) for f in facts)[:snippet], "via": "page summary"})
    return out


def run_sources(session_id: int, snippet: int = 300) -> dict:
    """What a run read before writing its digest. Falls back to the digest's own URLs once payloads have expired (30 days)."""
    seen, items = set(), []
    for p in db.get_phase_payloads(session_id, ("synthesizer",)):
        try:
            msgs = json.loads(p["input"] or "[]")
            user = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
            buf, _ = json.JSONDecoder().raw_decode(user.lstrip())
        except (ValueError, StopIteration):
            continue
        for s in _sources_from_buffer(buf, snippet):
            key = s["url"] or s["title"]
            if key and key not in seen:
                seen.add(key)
                items.append({**s, **({"sub": p["sub"]} if p.get("sub") else {})})
    if items:
        return {"session_id": session_id, "complete": True, "sources": items}
    out = db.get_output(session_id)
    return {"session_id": session_id, "complete": False,
            "note": "The run's research inputs are no longer stored (kept 30 days); these are only the URLs its digest cites.",
            "sources": [{"url": u} for u in urls(out)]}


def previous_digests(session_id: int, n: int, max_chars: int = 6000) -> Optional[dict]:
    """Earlier digests of the dispatch this run belongs to, for previous-run awareness.

    {block, urls: {url: date}, headings (latest run), when (latest run), runs, chars}, or None (no dispatch, n=0, no history).
    """
    if n <= 0:
        return None
    dispatch_id = db.session_dispatch_id(session_id)
    if not dispatch_id:
        return None
    prev = db.recent_outputs(dispatch_id, n, before_session=session_id)
    if not prev:
        return None
    per_run = max_chars // len(prev)
    parts, seen_urls = [], {}
    for r in prev:
        when = _local(r["started_at"])
        parts.append(f"--- Digest of {when} ---\n" + outline(r["content"], max_chars=per_run))
        for u in urls(r["content"]):
            seen_urls.setdefault(u, when)  # newest first, so the most recent date wins
    block = "\n\n".join(parts)
    return {"block": block, "urls": seen_urls, "headings": headings(prev[0]["content"]),
            "when": _local(prev[0]["started_at"]), "runs": len(prev), "chars": len(block)}
