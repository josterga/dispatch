"""Per-run instrumentation.

Every LLM call, search, fetch, link check, pipeline decision and host sample of a run becomes a
`run_events` row (full LLM prompts/responses go to `run_payloads`, kept PAYLOAD_RETENTION_DAYS).
finish() rolls the events up into `sessions.metrics`, which the run card and dashboard read.
"""
import dataclasses
import glob
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

import db

METRICS_VERSION = 2  # 2: self-checks (degraded, checks)
PAYLOAD_RETENTION_DAYS = int(os.getenv("PAYLOAD_RETENTION_DAYS", "30"))
HOST_SAMPLE_SECONDS = float(os.getenv("HOST_SAMPLE_SECONDS", "5"))
SLOW_FACTOR = float(os.getenv("CHECK_SLOW_FACTOR", "3"))  # slower than this × the dispatch's median = degraded
_RELOAD_MS = 1000           # load_duration above this = the model was (re)loaded for the call
_CTX_NEAR_FULL = 0.95       # prompt tokens >= 95% of num_ctx = input probably cut by Ollama

_recorders: dict[int, "RunRecorder"] = {}
_recorders_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


# ── Static run context ────────────────────────────────────────────────────────

_build_cache: Optional[dict] = None


def build_info() -> dict:
    """Deployed code version: BUILD (written when staging the deploy), else git in the source tree."""
    global _build_cache
    if _build_cache is None:
        here = Path(__file__).parent
        try:
            _build_cache = json.loads((here / "BUILD").read_text())
        except Exception:
            try:
                commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5).stdout.strip()
                dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, capture_output=True, text=True, timeout=5).stdout.strip())
                _build_cache = {"commit": commit or None, "dirty": dirty}
            except Exception:
                _build_cache = {"commit": None}
    return _build_cache


def prompt_hashes() -> dict:
    import prompts
    return {
        name: hashlib.sha1(value.encode()).hexdigest()[:10]
        for name, value in vars(prompts).items()
        if name.startswith("PROMPT_") and isinstance(value, str)
    }


def env_knobs() -> dict:
    keys = [
        "PROVIDER", "MODEL", "FAST_MODEL", "CONTEXT_BUDGET", "NUM_CTX", "SYNTHESIZER_CTX",
        "SYNTHESIZER_MAX_TOKENS", "SYNTHESIZER_MAX_CHARS", "TEMPERATURE", "TIMEOUT", "KEEP_ALIVE",
        "MAX_ITERATIONS", "PLANNER_QUERY_COUNT", "PARALLEL_SUB_PIPELINES", "WEB_FETCH_PROVIDER",
        "OLLAMA_SEARCH_RESULTS_PER_PROVIDER",
    ]
    out = {k: os.environ[k] for k in keys if k in os.environ}
    # Which search providers are configured (never the keys themselves).
    out["search_providers"] = [p for p, k in (("searxng", "SEARXNG_URL"), ("ollama", "OLLAMA_API_KEY"), ("brave", "BRAVE_API_KEY"), ("jina", "JINA_API_KEY")) if os.getenv(k)]
    return out


def ollama_ps() -> Optional[list]:
    if os.getenv("PROVIDER", "ollama") != "ollama":
        return None
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        models = requests.get(f"{base}/api/ps", timeout=3).json().get("models", [])
        return [
            {"name": m.get("name"), "size": m.get("size"), "size_vram": m.get("size_vram"),
             "context_length": m.get("context_length"), "expires_at": m.get("expires_at")}
            for m in models
        ]
    except Exception:
        return None


# ── Host sampler ──────────────────────────────────────────────────────────────

def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _hwmon_temp(name: str) -> Optional[str]:
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        if _read(f"{d}/name") == name and os.path.exists(f"{d}/temp1_input"):
            return f"{d}/temp1_input"
    return None


class HostSampler:
    """CPU/GPU/memory/temperature every HOST_SAMPLE_SECONDS while a run is active (Linux only)."""

    def __init__(self, t0: float):
        self.t0 = t0
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._gpu_dir = next((os.path.dirname(p) for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")), None)
        self._temps = {k: _hwmon_temp(n) for k, n in (("gpu_c", "amdgpu"), ("cpu_c", "k10temp"), ("nvme_c", "nvme"))}
        self._prev_cpu: Optional[tuple[int, int]] = None
        self.enabled = os.path.exists("/proc/stat")

    def start(self) -> None:
        if self.enabled:
            self._cpu()  # prime the CPU delta
            threading.Thread(target=self._loop, daemon=True, name="host-sampler").start()

    def stop(self) -> None:
        self._stop.set()
        if self.enabled:
            self._sample()

    def _cpu(self) -> Optional[float]:
        line = _read("/proc/stat")
        if not line:
            return None
        vals = [int(v) for v in line.splitlines()[0].split()[1:]]
        idle, total = vals[3] + vals[4], sum(vals)
        prev, self._prev_cpu = self._prev_cpu, (idle, total)
        if not prev or total == prev[1]:
            return None
        return round(100 * (1 - (idle - prev[0]) / (total - prev[1])), 1)

    def _sample(self) -> None:
        s: dict = {"t": int((time.monotonic() - self.t0) * 1000), "cpu": self._cpu()}
        mem = _read("/proc/meminfo") or ""
        fields = {k: int(v.split()[0]) * 1024 for k, v in (l.split(":", 1) for l in mem.splitlines() if ":" in l)}
        s["mem_avail"] = fields.get("MemAvailable")
        s["load1"] = float((_read("/proc/loadavg") or "0").split()[0])
        if self._gpu_dir:
            for key, f in (("gpu", "gpu_busy_percent"), ("gtt", "mem_info_gtt_used"), ("vram", "mem_info_vram_used")):
                v = _read(f"{self._gpu_dir}/{f}")
                s[key] = int(v) if v and v.isdigit() else None
        for key, path in self._temps.items():
            v = _read(path) if path else None
            s[key] = round(int(v) / 1000, 1) if v and v.lstrip("-").isdigit() else None
        self.samples.append(s)

    def _loop(self) -> None:
        while not self._stop.wait(HOST_SAMPLE_SECONDS):
            self._sample()

    def summary(self) -> Optional[dict]:
        if not self.samples:
            return None

        def col(k):
            return [s[k] for s in self.samples if s.get(k) is not None]

        out: dict = {"samples": len(self.samples)}
        for k in ("cpu", "gpu"):
            v = col(k)
            if v:
                out[f"{k}_avg"], out[f"{k}_max"] = round(sum(v) / len(v), 1), max(v)
        for k in ("gpu_c", "cpu_c", "nvme_c", "gtt", "vram", "load1"):
            v = col(k)
            if v:
                out[f"{k}_max"] = max(v)
        v = col("mem_avail")
        if v:
            out["mem_avail_min"] = min(v)
        return out


# ── Recorder ──────────────────────────────────────────────────────────────────

class RunRecorder:
    def __init__(self, session_id: int, trigger: str, queued_at: Optional[str] = None,
                 scheduled_for: Optional[str] = None, dispatch_id: Optional[int] = None):
        self.session_id = session_id
        self.t0 = time.monotonic()
        self.run: dict = {
            "trigger": trigger, "dispatch_id": dispatch_id, "scheduled_for": scheduled_for,
            "queued_at": queued_at, "started_at": _now(),
        }
        self.events: list[dict] = []
        self.current_phase: Optional[str] = None
        self.config: Optional[dict] = None
        self._pending: list[dict] = []
        self._payloads: list[dict] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._owner = threading.get_ident()
        self._sampler = HostSampler(self.t0)
        self._sampler.start()
        self.event("ollama_ps", name="start", models=ollama_ps())

    # Generic event. `start` is a time.monotonic() value; duration defaults to now - start.
    def event(self, kind: str, phase: Optional[str] = None, name: Optional[str] = None, *,
              sub: Optional[str] = None, start: Optional[float] = None, duration_ms: Optional[int] = None,
              ok: bool = True, error: Optional[str] = None, payload: Optional[tuple] = None, **data) -> int:
        now = time.monotonic()
        if start is not None and duration_ms is None:
            duration_ms = int((now - start) * 1000)
        t_ms = int(((start if start is not None else now) - self.t0) * 1000)
        with self._lock:
            self._seq += 1
            ev = {
                "seq": self._seq, "t_ms": t_ms, "duration_ms": duration_ms, "kind": kind,
                "phase": phase or self.current_phase, "sub": sub, "name": name,
                "ok": ok, "error": error, "data": {k: v for k, v in data.items() if v is not None},
            }
            self.events.append(ev)
            self._pending.append(ev)
            if payload is not None:
                self._payloads.append({"seq": self._seq, "input": payload[0], "output": payload[1]})
            seq = self._seq
        if threading.get_ident() == self._owner:
            self.flush()  # pool threads leave theirs for the run thread's next event
        return seq

    @contextmanager
    def phase(self, name: str, sub: Optional[str] = None, **data):
        prev, self.current_phase = self.current_phase, name
        start = time.monotonic()
        try:
            yield
        finally:
            self.event("phase", name, name, sub=sub, start=start, **data)
            self.current_phase = prev

    def llm(self, phase: str, *, sub: Optional[str], start: float, model: str, options: dict,
            messages: list, output: Optional[str] = None, usage: Optional[dict] = None,
            tool_calls: int = 0, error: Optional[Exception] = None) -> None:
        wall_ms = int((time.monotonic() - start) * 1000)
        system_chars = sum(len(m.get("content") or "") for m in messages if m.get("role") == "system")
        input_chars = sum(len(m.get("content") or "") for m in messages)
        u = usage or {}
        data: dict = {
            "model": model, **{k: v for k, v in options.items() if v is not None},
            "messages": len(messages), "system_chars": system_chars, "input_chars": input_chars,
            "output_chars": len(output or ""), "tool_calls": tool_calls or None, **u,
        }
        if u.get("total_ms") is not None:
            data["wait_ms"] = max(0, wall_ms - u["total_ms"])
        if u.get("prompt_tokens") and u.get("prompt_ms"):
            data["prefill_tps"] = round(u["prompt_tokens"] / (u["prompt_ms"] / 1000), 1)
        if u.get("output_tokens") and u.get("gen_ms"):
            data["gen_tps"] = round(u["output_tokens"] / (u["gen_ms"] / 1000), 1)
        if u.get("load_ms") is not None:
            data["reloaded"] = u["load_ms"] > _RELOAD_MS
        data["truncated"] = u.get("stop_reason") in ("length", "max_tokens")
        ctx = options.get("num_ctx")
        if ctx and u.get("prompt_tokens"):
            data["ctx_used"] = round(u["prompt_tokens"] / ctx, 3)
            data["ctx_near_full"] = u["prompt_tokens"] >= ctx * _CTX_NEAR_FULL
        clean = [{k: v for k, v in m.items() if k != "_anthropic_blocks"} for m in messages]
        self.event(
            "llm", phase, model, sub=sub, start=start, duration_ms=wall_ms,
            ok=error is None and bool(output or tool_calls),
            error=f"{type(error).__name__}: {error}" if error else (None if output or tool_calls else "empty output"),
            payload=(json.dumps(clean, ensure_ascii=False), output), **data,
        )

    def mark_ended(self) -> None:
        """Pipeline done; what follows (notification) is timed separately."""
        self.run.setdefault("ended_at", _now())

    def flush(self) -> None:
        with self._lock:
            rows, self._pending = self._pending, []
            payloads, self._payloads = self._payloads, []
        if not rows and not payloads:
            return
        try:
            db.insert_run_events(self.session_id, rows)
            db.insert_run_payloads(self.session_id, payloads)
        except Exception as e:
            print(f"[metrics] WARNING: could not write events ({type(e).__name__}: {e})")

    def evaluate(self, status: str, output: Optional[str], error: Optional[BaseException] = None) -> dict:
        """Rollup + self-checks of everything recorded so far (used before notifying, then again at finish)."""
        self.mark_ended()
        if error is not None and "error" not in self.run:
            self.run["error"] = {"type": type(error).__name__, "message": str(error)[:2000], "phase": self.current_phase}
        metrics = rollup(self.events, self.run, status, output, self.config, self._sampler.summary())
        apply_checks(metrics, self.session_id)
        return metrics

    def finish(self, status: str, output: Optional[str], error: Optional[BaseException] = None) -> dict:
        self._sampler.stop()
        self.event("ollama_ps", name="end", models=ollama_ps())
        samples = self._sampler.samples
        if samples:
            self.event("host", name="samples", interval_s=HOST_SAMPLE_SECONDS,
                       **{k: [s.get(k) for s in samples] for k in samples[0]})
        self.flush()
        metrics = self.evaluate(status, output, error)
        try:
            db.set_session_metrics(self.session_id, metrics)
            db.purge_run_payloads(PAYLOAD_RETENTION_DAYS)
        except Exception as e:
            print(f"[metrics] WARNING: could not write metrics ({type(e).__name__}: {e})")
        return metrics


class _NullRecorder:
    """Stands in when a session has no active recorder (CLI runs)."""
    current_phase = None
    config = None

    def event(self, *a, **k) -> int:
        return 0

    @contextmanager
    def phase(self, *a, **k):
        yield

    def llm(self, *a, **k) -> None:
        pass

    def mark_ended(self) -> None:
        pass

    def evaluate(self, *a, **k) -> None:
        return None

    def flush(self) -> None:
        pass


_NULL = _NullRecorder()


def start(session_id: int, trigger: str, **kw) -> RunRecorder:
    rec = RunRecorder(session_id, trigger, **kw)
    with _recorders_lock:
        _recorders[session_id] = rec
    return rec


def get(session_id: Optional[int]):
    with _recorders_lock:
        return _recorders.get(session_id, _NULL) if session_id is not None else _NULL


def finish(session_id: int, status: str, output: Optional[str], error: Optional[BaseException] = None) -> Optional[dict]:
    with _recorders_lock:
        rec = _recorders.pop(session_id, None)
    return rec.finish(status, output, error) if rec else None


def record_config(rec, config, model: str) -> None:
    """Snapshot of everything that shapes a run, so changes in results can be traced to changes here."""
    if rec is _NULL:
        return
    cfg = dataclasses.asdict(config)
    rec.config = cfg
    rec.event("config", name=config.agent_name, agent=cfg, model=model, env=env_knobs(),
              build=build_info(), prompts=prompt_hashes(),
              output_template_sha=hashlib.sha1((config.output_template or "").encode()).hexdigest()[:10])


# ── Inbox pages (local intel inbox served to Direct Fetch) ────────────────────

_INBOX_BACKLOG = re.compile(r"(\d+) more items queued")


def inbox_stats(content: str) -> Optional[dict]:
    if not content.startswith("# Intel inbox"):
        return None
    head = content.splitlines()[0]
    m = _INBOX_BACKLOG.search(head)
    return {
        "items": sum(1 for l in content.splitlines() if l.startswith("## [")),
        "backlog": int(m.group(1)) if m else 0,
        "item_urls": [l[5:].strip() for l in content.splitlines() if l.startswith("url: ") and l[5:].strip()],
    }


# ── Rollup ────────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+")


def _output_stats(output: Optional[str], config: Optional[dict]) -> dict:
    text = output or ""
    headings = [l.lstrip("#").strip() for l in text.splitlines() if re.match(r"^#{1,6} ", l)]
    urls = sorted({u.rstrip(".,;:") for u in _URL_RE.findall(text)})
    out: dict = {"chars": len(text), "words": len(text.split()), "headings": headings, "urls": urls}
    template = (config or {}).get("output_template") or ""
    required = [l.lstrip("#").strip() for l in template.splitlines() if re.match(r"^#{1,6} ", l)]
    if required:
        have = {h.lower() for h in headings}
        out["sections"] = {r: r.lower() in have for r in required}
        out["missing_sections"] = [r for r in required if r.lower() not in have]
    return out


def _ms_between(a: Optional[str], b: Optional[str]) -> Optional[int]:
    try:
        return int((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() * 1000)
    except Exception:
        return None


def rollup(events: list[dict], run: dict, status: str, output: Optional[str],
           config: Optional[dict], host: Optional[dict]) -> dict:
    by_kind: dict[str, list[dict]] = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)

    # Timing
    phases: dict[str, int] = {}
    subs: dict[str, int] = {}
    for e in by_kind.get("phase", []):
        if e["name"] == "pipeline":  # one per sub-pipeline of a multi-agent run
            subs[e["sub"] or "?"] = e["duration_ms"] or 0
        else:
            phases[e["name"]] = phases.get(e["name"], 0) + (e["duration_ms"] or 0)
    timing = {
        "late_ms": _ms_between(run.get("scheduled_for"), run.get("started_at")),
        "queue_ms": _ms_between(run.get("queued_at"), run.get("started_at")),
        "run_ms": _ms_between(run.get("started_at"), run.get("ended_at")),
        "phases": phases,
        "subs": subs or None,
    }

    # LLM
    llm_calls = by_kind.get("llm", [])
    by_phase: dict[str, dict] = {}
    for e in llm_calls:
        d = e["data"]
        p = by_phase.setdefault(e["phase"] or "?", {"calls": 0, "ms": 0, "prompt_tokens": 0, "output_tokens": 0, "errors": 0})
        p["calls"] += 1
        p["ms"] += e["duration_ms"] or 0
        p["prompt_tokens"] += d.get("prompt_tokens") or 0
        p["output_tokens"] += d.get("output_tokens") or 0
        p["errors"] += 0 if e["ok"] else 1
        for k in ("gen_tps", "prefill_tps"):
            if d.get(k):
                p.setdefault(f"_{k}", []).append(d[k])
    for p in by_phase.values():
        for k in ("gen_tps", "prefill_tps"):
            v = p.pop(f"_{k}", None)
            if v:
                p[k] = round(sum(v) / len(v), 1)
    llm = {
        "calls": len(llm_calls),
        "errors": sum(1 for e in llm_calls if not e["ok"]),
        "prompt_tokens": sum(e["data"].get("prompt_tokens") or 0 for e in llm_calls),
        "output_tokens": sum(e["data"].get("output_tokens") or 0 for e in llm_calls),
        "thinking_chars": sum(e["data"].get("thinking_chars") or 0 for e in llm_calls),
        "reloads": sum(1 for e in llm_calls if e["data"].get("reloaded")),
        "truncated": sum(1 for e in llm_calls if e["data"].get("truncated")),
        "ctx_near_full": sum(1 for e in llm_calls if e["data"].get("ctx_near_full")),
        "wait_ms": sum(e["data"].get("wait_ms") or 0 for e in llm_calls),
        "by_phase": by_phase,
    }

    # Search
    searches = by_kind.get("search", [])
    providers: dict[str, dict] = {}
    url_providers: dict[str, set] = {}
    hit_domains: dict[str, int] = {}
    for e in searches:
        meta = e["data"]
        for name, p in (meta.get("providers") or {}).items():
            agg = providers.setdefault(name, {"calls": 0, "ok": 0, "errors": 0, "off": 0, "cooldown": 0, "hits": 0, "ms": 0})
            st = p.get("status")
            agg["calls"] += 1 if st in ("ok", "error") else 0
            agg["ok"] += 1 if st == "ok" else 0
            agg["errors"] += 1 if st == "error" else 0
            agg["off"] += 1 if st == "off" else 0
            agg["cooldown"] += 1 if st == "cooldown" else 0
            agg["hits"] += p.get("count") or 0
            agg["ms"] += p.get("ms") or 0
        for h in meta.get("hits") or []:
            url_providers.setdefault(h["url"], set()).update(h.get("by", {}))
            d = _domain(h["url"])
            hit_domains[d] = hit_domains.get(d, 0) + 1
    for urls_by in url_providers.values():
        if len(urls_by) == 1:
            providers[next(iter(urls_by))]["unique"] = providers[next(iter(urls_by))].get("unique", 0) + 1
    search = {
        "queries": len(searches),
        "retry_queries": sum(1 for e in searches if (e["data"].get("round") or 0) > 0),
        "failed_queries": sum(1 for e in searches if not e["ok"]),
        "hits": sum(len(e["data"].get("hits") or []) for e in searches),
        "unique_urls": len(url_providers),
        "dupes": sum(e["data"].get("dupes") or 0 for e in searches),
        "ddg_fallback": sum(1 for e in searches if e["data"].get("ddg_fallback")),
        "providers": providers,
    } if searches else None

    # Link checks
    checks = by_kind.get("link_check", [])
    links = {
        "checked": sum(len(e["data"].get("urls") or []) for e in checks),
        "dead": sum(sum(1 for u in e["data"].get("urls") or [] if not u.get("ok")) for e in checks),
    } if checks else None

    # Fetches
    fetches = by_kind.get("fetch", [])
    inbox_items = inbox_backlog = 0
    inbox_seen = False
    for e in fetches:
        ib = e["data"].get("inbox")
        if ib:
            inbox_seen = True
            inbox_items += ib.get("items") or 0
            inbox_backlog = max(inbox_backlog, ib.get("backlog") or 0)
    fetch = {
        "pages": len(fetches),
        "ok": sum(1 for e in fetches if e["ok"]),
        "failed": sum(1 for e in fetches if not e["ok"]),
        "jina": sum(1 for e in fetches if e["data"].get("jina")),
        "chars": sum(e["data"].get("chars") or 0 for e in fetches),
        "truncated": sum(1 for e in fetches if e["data"].get("truncated")),
        "ms": sum(e["duration_ms"] or 0 for e in fetches),
    } if fetches else None

    # What reached the synthesizer
    synth_inputs = by_kind.get("synth_input", [])
    source_urls: set[str] = set()
    breakdown = {"inbox_items": 0, "direct_pages": 0, "fetched_pages": 0, "summarized_pages": 0, "search_results": 0}
    for e in synth_inputs:
        d = e["data"]
        for k in breakdown:
            breakdown[k] += d.get(k) or 0
        source_urls.update(d.get("urls") or [])
    # None when the run never got to the synthesizer (e.g. failed earlier): unknown, not zero.
    sources = len(source_urls) + breakdown["inbox_items"] if synth_inputs else None
    source_domains: dict[str, int] = {}
    for u in source_urls:
        dm = _domain(u)
        source_domains[dm] = source_domains.get(dm, 0) + 1
    synth_llm = [e for e in llm_calls if e["phase"] == "synthesizer"]
    synth = {
        "input_chars": sum(e["data"].get("chars_before") or 0 for e in synth_inputs),
        "sent_chars": sum(e["data"].get("chars_after") or 0 for e in synth_inputs),
        "trimmed": sum(1 for e in synth_inputs if e["data"].get("trim_cap")),
        "hard_truncated": sum(1 for e in synth_inputs if e["data"].get("hard_truncated")),
        "prompt_tokens": sum(e["data"].get("prompt_tokens") or 0 for e in synth_llm),
        "output_tokens": sum(e["data"].get("output_tokens") or 0 for e in synth_llm),
        "ms": sum(e["duration_ms"] or 0 for e in synth_llm),
        "output_cut_off": sum(1 for e in synth_llm if e["data"].get("truncated")),
        "ctx_near_full": sum(1 for e in synth_llm if e["data"].get("ctx_near_full")),
    } if synth_inputs else None

    # Output
    out = _output_stats(output if status == "done" else None, config)
    if source_urls:
        cited = [u for u in out["urls"] if u in source_urls]
        out["cited_sources"] = len(cited)
        out["citation_rate"] = round(len(cited) / len(source_urls), 3)

    notify_ev = next((e for e in reversed(by_kind.get("notify", []))), None)
    notify = {"sent": notify_ev["ok"], "status": notify_ev["data"].get("http_status"), "ms": notify_ev["duration_ms"],
              "summary_ms": notify_ev["data"].get("summary_ms")} if notify_ev else None
    if notify_ev:
        timing["notify_ms"] = notify_ev["duration_ms"]

    cfg_ev = next(iter(by_kind.get("config", [])), None)
    decisions = {}
    for e in by_kind.get("decision", []):
        decisions.setdefault(e["name"], []).append(e["data"])

    return {
        "v": METRICS_VERSION,
        "source": "live",
        "status": status,
        "run": run,
        "model": (cfg_ev or {}).get("data", {}).get("model"),
        "build": (cfg_ev or {}).get("data", {}).get("build"),
        "timing": timing,
        "sources": sources,
        "source_breakdown": breakdown,
        "inbox": {"items": inbox_items, "backlog": inbox_backlog} if inbox_seen else None,
        "llm": llm,
        "search": search,
        "links": links,
        "fetch": fetch,
        "synth": synth,
        "output": out,
        "notify": notify,
        "host": host,
        "planner_queries": sum(len(d.get("queries") or []) for d in decisions.get("planner", []) if not d.get("direct_sources")) or None,
        "validator": decisions.get("validator"),
        "source_domains": sorted(source_domains.items(), key=lambda kv: -kv[1])[:20],
        "hit_domains": sorted(hit_domains.items(), key=lambda kv: -kv[1])[:20],
        "events": len(events),
    }


# ── Self-checks (rule-based, no LLM) ──────────────────────────────────────────

def run_checks(m: dict, median_ms: Optional[int]) -> list[dict]:
    """Rules that mark a finished, non-error run as degraded. Each hit: {rule, detail}."""
    hits: list[dict] = []
    if m.get("status") not in ("done", "empty"):
        return hits

    def hit(rule: str, detail: str) -> None:
        hits.append({"rule": rule, "detail": detail})

    if m.get("status") == "empty":
        hit("empty_output", "the pipeline produced no answer")
    if m.get("sources") == 0:
        inbox = m.get("inbox")
        hit("no_sources", "nothing new in the inbox" if inbox is not None else "no sources reached the synthesizer")
    search = m.get("search") or {}
    if search.get("queries") and search.get("failed_queries") == search.get("queries"):
        hit("search_failed", f"all {search['queries']} searches failed")
    elif search.get("providers") and not any(p.get("ok") for p in search["providers"].values()) and any(p.get("calls") for p in search["providers"].values()):
        hit("search_failed", "no search provider answered")
    fetch = m.get("fetch") or {}
    if fetch.get("pages") and not fetch.get("ok"):
        hit("fetch_failed", f"all {fetch['pages']} fetches failed")
    synth = m.get("synth") or {}
    if synth.get("output_cut_off"):
        hit("output_cut_off", "the synthesizer hit its output token cap")
    if synth.get("hard_truncated"):
        hit("input_truncated", "research was cut to fit the synthesizer input limit")
    if synth.get("ctx_near_full"):
        hit("context_full", "the synthesizer prompt filled the context window (Ollama drops the overflow)")
    missing = (m.get("output") or {}).get("missing_sections")
    if missing and m.get("status") == "done":
        hit("missing_sections", "missing: " + ", ".join(missing))
    run_ms = (m.get("timing") or {}).get("run_ms")
    if median_ms and run_ms and run_ms > SLOW_FACTOR * median_ms:
        hit("slow", f"{run_ms // 1000}s vs a median of {median_ms // 1000}s")
    return hits


def apply_checks(m: dict, session_id: Optional[int], use_history: bool = True) -> dict:
    median = None
    dispatch_id = (m.get("run") or {}).get("dispatch_id")
    if use_history and dispatch_id:
        try:
            prior = sorted(db.recent_run_ms(dispatch_id, exclude_session=session_id))
            if len(prior) >= 3:
                median = prior[len(prior) // 2]
        except Exception:
            pass
    m["checks"] = run_checks(m, median)
    m["degraded"] = bool(m["checks"])
    m["median_ms"] = median
    m["v"] = METRICS_VERSION
    return m


def refresh_checks() -> int:
    """Add self-checks to runs whose metrics predate them."""
    n = 0
    for row in db.sessions_needing_checks():
        try:
            db.set_session_metrics(row["id"], apply_checks(row["metrics"], row["id"]))
            n += 1
        except Exception as e:
            print(f"[metrics] check refresh skipped session {row['id']} ({type(e).__name__}: {e})")
    if n:
        print(f"[metrics] self-checks added to {n} sessions")
    return n


def problem_note(m: Optional[dict]) -> Optional[str]:
    """One line for a notification when a run needs a look, else None."""
    if not m:
        return None
    if m.get("status") == "error":
        err = (m.get("run") or {}).get("error") or {}
        return f"Failed: {err.get('type', 'error')}: {err.get('message', '')}"[:300]
    if m.get("degraded"):
        return "Degraded: " + "; ".join(c["detail"] for c in m["checks"])
    return None


# ── Backfill (runs recorded before metrics existed: parse their logs) ─────────

_RE_PHASE_MS = re.compile(r"^\[pipeline\] (planner|validator|synthesizer): (\d+)ms$")
_RE_SYNTH_IN = re.compile(r"^\[pipeline\] synthesizer: input=(\d+) chars")
_RE_QUERIES = re.compile(r"^\[pipeline\] planner: (\d+) (?:queries generated|direct_sources)")
_RE_MERGED = re.compile(r"^\[web_search\] (\d+) merged results \((.*)\)")
_RE_PROV_ERR = re.compile(r"^\[(searxng|ollama|brave|jina|ddg)_search\] ERROR")
_RE_FETCH = re.compile(r"^\[web_fetch\]\s+content: (\d+) chars, snippet: (.*)$")
_RE_FETCH_ERR = re.compile(r"^\[web_fetch\]\s+(?:Jina )?ERROR")
_RE_INBOX = re.compile(r"(\d+) new items")
_RE_RELEVANT = re.compile(r"page-summarizer: (\d+)/(\d+) pages relevant")


def metrics_from_logs(session: dict, messages: list[dict]) -> dict:
    logs = session.get("logs") or ""
    phases: dict[str, int] = {}
    providers: dict[str, dict] = {}
    queries = synth_in = 0
    fetch_ok = fetch_err = fetch_chars = inbox_items = 0
    inbox_seen = False
    summarized = None
    for line in logs.splitlines():
        if m := _RE_PHASE_MS.match(line):
            phases[m.group(1)] = phases.get(m.group(1), 0) + int(m.group(2))
        elif m := _RE_SYNTH_IN.match(line):
            synth_in += int(m.group(1))
        elif m := _RE_QUERIES.match(line):
            queries += int(m.group(1))
        elif m := _RE_MERGED.match(line):
            for part in m.group(2).split(","):
                name, _, n = part.strip().partition("=")
                agg = providers.setdefault(name, {"calls": 0, "ok": 0, "errors": 0, "hits": 0})
                if n.isdigit() and int(n) > 0:
                    agg["calls"] += 1
                    agg["ok"] += 1
                    agg["hits"] += int(n)
        elif m := _RE_PROV_ERR.match(line):
            agg = providers.setdefault(m.group(1), {"calls": 0, "ok": 0, "errors": 0, "hits": 0})
            agg["calls"] += 1
            agg["errors"] += 1
        elif m := _RE_FETCH.match(line):
            fetch_ok += 1
            fetch_chars += int(m.group(1))
            if "Intel inbox" in m.group(2):
                inbox_seen = True
                if im := _RE_INBOX.search(m.group(2)):
                    inbox_items += int(im.group(1))
        elif _RE_FETCH_ERR.match(line):
            fetch_err += 1
        elif m := _RE_RELEVANT.search(line):
            summarized = (summarized or 0) + int(m.group(1))

    roles = [m["role"] for m in messages]
    final = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), None)
    if "error" in roles:
        status = "error"
    elif not final or final.startswith("[no result"):
        status = "empty"
    else:
        status = "done"
    # Inbox pages past the batch say "No more items on page N", without "new items": count only real ones.
    sources = inbox_items if inbox_seen else (summarized or 0)
    return {
        "v": METRICS_VERSION,
        "source": "backfill",
        "status": status,
        "run": {"trigger": session.get("source"), "dispatch_id": session.get("dispatch_id"),
                "started_at": session.get("started_at"), "ended_at": session.get("ended_at")},
        "model": session.get("model"),
        "timing": {"run_ms": _ms_between(session.get("started_at"), session.get("ended_at")), "phases": phases},
        "sources": sources if (inbox_seen or summarized is not None) else None,
        "inbox": {"items": inbox_items} if inbox_seen else None,
        "search": {"queries": queries, "providers": providers} if providers else None,
        "fetch": {"ok": fetch_ok, "failed": fetch_err, "pages": fetch_ok + fetch_err, "chars": fetch_chars} if fetch_ok or fetch_err else None,
        "synth": {"input_chars": synth_in} if synth_in else None,
        "output": _output_stats(final if status == "done" else None, None),
    }


def backfill() -> int:
    n = 0
    for session in db.sessions_without_metrics():
        try:
            detail = db.get_session_detail(session["id"])
            db.set_session_metrics(session["id"], apply_checks(metrics_from_logs(detail, detail["messages"]), None, use_history=False))
            n += 1
        except Exception as e:
            print(f"[metrics] backfill skipped session {session['id']} ({type(e).__name__}: {e})")
    if n:
        print(f"[metrics] backfilled {n} sessions from logs")
    refresh_checks()
    return n


# ── Stats for the dashboard ───────────────────────────────────────────────────

def stats(days: int) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = db.sessions_with_metrics(since)
    runs = []
    providers: dict[str, dict] = {}
    domains: dict[str, int] = {}
    for r in rows:
        m = r["metrics"]
        t = m.get("timing") or {}
        runs.append({
            "id": r["id"], "agent": r["agent"], "dispatch_id": r["dispatch_id"], "source": r["source"],
            "started_at": r["started_at"], "status": m.get("status"), "sources": m.get("sources"),
            "run_ms": t.get("run_ms"), "phases": t.get("phases"), "late_ms": t.get("late_ms"),
            "synth_chars": (m.get("synth") or {}).get("sent_chars") or (m.get("synth") or {}).get("input_chars"),
            "synth_ms": (m.get("synth") or {}).get("ms") or (t.get("phases") or {}).get("synthesizer"),
            "tokens": ((m.get("llm") or {}).get("prompt_tokens"), (m.get("llm") or {}).get("output_tokens")),
            "metrics_source": m.get("source"),
            "missing_sections": (m.get("output") or {}).get("missing_sections"),
            "degraded": m.get("degraded"),
            "checks": [c["detail"] for c in m.get("checks") or []],
        })
        for name, p in ((m.get("search") or {}).get("providers") or {}).items():
            agg = providers.setdefault(name, {"calls": 0, "ok": 0, "errors": 0, "hits": 0, "ms": 0})
            for k in agg:
                agg[k] += p.get(k) or 0
        for d, c in m.get("source_domains") or []:
            domains[d] = domains.get(d, 0) + c
    done = [r for r in runs if r["status"] == "done" and not r["degraded"]]
    durations = sorted(r["run_ms"] for r in runs if r["run_ms"])
    return {
        "days": days,
        "runs": runs,
        "providers": providers,
        "domains": sorted(domains.items(), key=lambda kv: -kv[1])[:15],
        "kpi": {
            "runs": len(runs),
            "success_pct": round(100 * len(done) / len(runs), 1) if runs else None,
            "median_ms": durations[len(durations) // 2] if durations else None,
        },
    }
