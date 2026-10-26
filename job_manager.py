import io
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from queue import Queue
from typing import Optional

import os

import db
import metrics
import notify
from pipeline import MODEL, create_pipeline

# Runs due while dispatch was down: the latest one is caught up if it was due within this window,
# the rest are recorded as missed. 0 = never catch up (only record).
CATCH_UP_HOURS = float(os.getenv("DISPATCH_CATCH_UP_HOURS", "12"))
_MAX_MISSED_RECORDS = 24  # per dispatch and startup, so a long outage doesn't flood the run list
_JOB_TTL_SECONDS = 3600  # finished jobs stay in memory this long (the UI reads their final status), then go


_DOW_NAMES = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]


def _cron_dow(field: str) -> str:
    """Cron weekdays (0 or 7 = Sunday) → APScheduler names. APScheduler 3 numbers them from Monday,
    so numeric fields must not reach it. Expanded to an explicit list: its name ranges can't wrap."""
    if field == "*" or not any(c.isdigit() for c in field):
        return field
    days: set[int] = set()
    for part in field.split(","):
        rng, _, step = part.partition("/")
        if rng == "*":
            a, b = 0, 6
        else:
            lo, _, hi = rng.partition("-")
            a = int(lo)
            b = int(hi) if hi else (6 if step else a)
        if not (0 <= a <= 7 and 0 <= b <= 7 and a <= b):
            raise ValueError(f"bad day of week: {part}")
        days.update(d % 7 for d in range(a, b + 1, int(step) if step else 1))
    return ",".join(_DOW_NAMES[d] for d in sorted(days))


def cron_trigger(expr: str, tz: Optional[str], end_date: Optional[str] = None):
    """Like CronTrigger.from_crontab, but with standard cron weekday numbering."""
    from apscheduler.triggers.cron import CronTrigger
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"expected 5 cron fields, got {len(fields)}")
    m, h, dom, mon, dow = fields
    return CronTrigger(minute=m, hour=h, day=dom, month=mon, day_of_week=_cron_dow(dow),
                       timezone=tz or "UTC", end_date=end_date)


class TeeStream:
    def __init__(self, original, flush_cb=None):
        self.original = original
        self._buf = io.StringIO()
        self._flush_cb = flush_cb

    def write(self, text):
        self.original.write(text)
        self._buf.write(text)
        if self._flush_cb and '\n' in text:
            self._flush_cb(self._buf.getvalue())

    def flush(self):
        self.original.flush()

    def getvalue(self):
        return self._buf.getvalue()


# Log prefixes → short step names for the UI's running indicator (checked newest line first).
_STEPS = [
    ("[pipeline] planner", "planning"),
    ("[pipeline] searcher", "searching"),
    ("_search]", "searching"),
    ("[pipeline] ranker", "ranking"),
    ("[pipeline] fetcher", "fetching"),
    ("[web_fetch]", "fetching"),
    ("[pipeline] page-summarizer", "reading"),
    ("[pipeline] link-validator", "checking links"),
    ("[pipeline] validator", "checking"),
    ("[pipeline] synthesizer", "writing"),
    ("[agent]", "thinking"),
]


def _current_step(logs: str) -> Optional[str]:
    for line in reversed(logs[-4000:].splitlines()):
        for prefix, step in _STEPS:
            if prefix in line:
                return step
    return None


def _notify(client, session_id: int, title: str, result: Optional[str], verdict: Optional[dict]) -> None:
    """Push summary + ntfy, recorded as one "notify" event of the run. Failed or empty runs send
    only what went wrong; degraded runs lead with their self-check results."""
    note = metrics.problem_note(verdict) if notify.ALERT_PROBLEMS else None
    no_answer = (verdict or {}).get("status") in ("error", "empty")  # nothing worth summarizing
    if no_answer and not note:
        return
    t0 = time.monotonic()
    summary = "" if no_answer else notify.summarize(client, result, session_id)
    summary_ms = int((time.monotonic() - t0) * 1000)
    body = "\n\n".join(x for x in (note, summary) if x)
    status, err = notify.send(title=("⚠ " if note else "") + title, body=body,
                              tags=["warning"] if note else [], priority="high" if note else "default")
    metrics.get(session_id).event("notify", "notify", title, start=t0, ok=status is not None and status < 400,
                                  error=err, summary_ms=summary_ms, http_status=status, chars=len(body), alert=bool(note))


def _run_status(answer: Optional[str], cancelled: bool) -> str:
    return "cancelled" if cancelled else "done" if answer else "empty"


class JobManager:
    def __init__(self, client, scheduler):
        self.client = client
        self.scheduler = scheduler
        self.pipeline_configs: dict = {}
        self._jobs: dict = {}
        self._lock = threading.Lock()
        self._queue: Queue = Queue()

    def start_worker(self) -> None:
        threading.Thread(target=self._worker, daemon=True, name="job-worker").start()

    def _worker(self) -> None:
        while True:
            job_id, fn, args = self._queue.get()
            try:
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job and job["status"] == "cancelled":
                        continue
                    if job:
                        job["status"] = "running"
                fn(*args)
            finally:
                self._queue.task_done()

    # ── State accessors ───────────────────────────────────────────────────────

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_by_session(self, session_id: int) -> Optional[dict]:
        with self._lock:
            for job_id, job in self._jobs.items():
                if job["session_id"] == session_id:
                    return {
                        "job_id": job_id,
                        "status": job["status"],
                        "session_id": job["session_id"],
                        "started_at": job["started_at"],
                    }
        return None

    def _prune(self) -> None:
        """Drop jobs that finished over _JOB_TTL_SECONDS ago (results stay in the DB). Caller holds the lock."""
        cutoff = time.monotonic() - _JOB_TTL_SECONDS
        for jid in [jid for jid, j in self._jobs.items() if j.get("finished_at", cutoff) < cutoff]:
            del self._jobs[jid]

    def _queue_position(self, job_id: str) -> Optional[int]:
        """1-based place among queued jobs (dict order = submit order). Caller holds the lock."""
        queued = [jid for jid, j in self._jobs.items() if j["status"] == "queued"]
        return queued.index(job_id) + 1 if job_id in queued else None

    def queue_position(self, job_id: str) -> Optional[int]:
        with self._lock:
            return self._queue_position(job_id)

    def active_for_dispatch(self, dispatch_id: int) -> Optional[str]:
        """Status of the dispatch's queued or running job, if any."""
        with self._lock:
            for job in self._jobs.values():
                if job.get("dispatch_id") == dispatch_id and job["status"] in ("queued", "running"):
                    return job["status"]
        return None

    def list_active(self) -> list:
        with self._lock:
            return [
                {
                    "job_id": jid,
                    "status": j["status"],
                    "session_id": j["session_id"],
                    "agent": j.get("agent"),
                    "dispatch_id": j.get("dispatch_id"),
                    "started_at": j["started_at"],
                    "queue_position": self._queue_position(jid),
                    "step": _current_step(j["tee"].getvalue()) if j["status"] == "running" and j.get("tee") else None,
                }
                for jid, j in self._jobs.items()
                if j["status"] in ("queued", "running")
            ]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return False
        job["cancel_event"].set()
        return True

    # ── Job submission ────────────────────────────────────────────────────────

    def submit(self, body, cancel_event, session_id: int, inline_config=None) -> str:
        """Queue an HTTP-originated job. session_id must be pre-created by the caller."""
        job_id = uuid.uuid4().hex
        agent_name = "inline" if inline_config else (
            body.agent if body.agent in self.pipeline_configs else "general"
        )

        with self._lock:
            self._prune()
            self._jobs[job_id] = {
                "status": "queued",
                "agent": agent_name,
                "dispatch_id": getattr(body, "dispatch_id", None),
                "session_id": session_id,
                "result": None, "error": None,
                "cancel_event": cancel_event,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        self._queue.put((job_id, self._execute_http, (job_id, body, cancel_event, session_id, agent_name, inline_config)))
        return job_id

    def submit_and_wait(self, body, cancel_event, inline_config=None) -> dict:
        """Queue a job and block until it completes. Returns the job dict."""
        job_id = uuid.uuid4().hex
        done_event = threading.Event()

        with self._lock:
            self._prune()
            self._jobs[job_id] = {
                "status": "queued",
                "session_id": None, "result": None, "error": None,
                "cancel_event": cancel_event,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

        def _run():
            self._execute_http(job_id, body, cancel_event, session_id=None, agent_name=None, inline_config=inline_config)
            done_event.set()

        self._queue.put((job_id, _run, ()))
        done_event.wait()
        with self._lock:
            return self._jobs[job_id]

    def _scheduled_for(self, dispatch_id: int) -> Optional[str]:
        """The fire time APScheduler is running this job for (its latest fire time <= now)."""
        try:
            now = datetime.now(timezone.utc)
            job = self.scheduler.get_job(f"dispatch_{dispatch_id}")
            if job is None or not hasattr(job.trigger, "fields"):  # one-off DateTrigger, possibly already removed
                d = next((d for d in db.get_dispatches() if d["id"] == dispatch_id), None)
                if d and d.get("trigger_type") == "once":
                    t = datetime.fromisoformat(d["schedule"])
                    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).isoformat()
                return None
            prev, t = None, job.trigger.get_next_fire_time(None, now - timedelta(hours=1))
            while t and t <= now:
                prev, t = t, job.trigger.get_next_fire_time(t, t + timedelta(seconds=1))
            return prev.astimezone(timezone.utc).isoformat() if prev else None
        except Exception:
            return None

    def catch_up_missed(self, dispatches: list[dict], down_since: str) -> None:
        """On startup: find fire times that passed while dispatch was not running (APScheduler keeps
        no history and skips them silently). Catch up the latest one once, record the others as missed.
        down_since: the previous process's last heartbeat."""
        from apscheduler.triggers.date import DateTrigger
        now = datetime.now(timezone.utc)
        until = now - timedelta(seconds=90)  # anything newer is APScheduler's to run
        lines = []
        for d in dispatches:
            try:
                if d.get("trigger_type") == "once":
                    t = datetime.fromisoformat(d["schedule"])
                    trig = DateTrigger(run_date=t if t.tzinfo else t.replace(tzinfo=timezone.utc))
                else:
                    trig = cron_trigger(d["schedule"], d.get("timezone"))
                last = db.get_latest_session_for_dispatch(d["id"], include_missed=True)
                since = datetime.fromisoformat((last or {}).get("started_at") or d["created_at"])
                if since.tzinfo is None:
                    since = since.replace(tzinfo=timezone.utc)
                since = max(since, datetime.fromisoformat(down_since) - timedelta(seconds=60))
                end = until
                if d.get("repeat_until"):
                    ru = datetime.fromisoformat(d["repeat_until"])
                    end = min(end, ru if ru.tzinfo else ru.replace(tzinfo=timezone.utc))
                fires, t = [], trig.get_next_fire_time(None, since + timedelta(seconds=1))
                while t and t < end and len(fires) < 1000:
                    fires.append(t)
                    t = trig.get_next_fire_time(t, t + timedelta(seconds=1))
                if not fires:
                    continue
                latest = fires[-1]
                catch = CATCH_UP_HOURS > 0 and now - latest <= timedelta(hours=CATCH_UP_HOURS)
                missed = fires[:-1] if catch else fires
                for f in missed[-_MAX_MISSED_RECORDS:]:
                    db.record_missed_run(d["id"], d["agent"], MODEL, d["prompt"], f.astimezone(timezone.utc).isoformat())
                name = d.get("label") or d["agent"]
                print(f"[dispatch] {name}: {len(fires)} run(s) missed while down" + (", catching up the latest" if catch else ""))
                lines.append(f"{name}: {len(fires)} missed" + (" (caught up the latest)" if catch else ""))
                if catch:
                    self.run_scheduled(d["id"], d["agent"], d["prompt"], trigger="catch_up",
                                       scheduled_for=latest.astimezone(timezone.utc).isoformat())
            except Exception as e:
                print(f"[dispatch] catch-up check failed for dispatch {d.get('id')}: {type(e).__name__}: {e}")
        if lines and notify.ALERT_PROBLEMS:
            notify.send(title="⚠ dispatch missed runs", body="Dispatch was not running when these were due:\n" + "\n".join(lines),
                        tags=["warning"], priority="high")

    def run_scheduled(self, dispatch_id: int, agent: str, prompt: str, trigger: str = "scheduled",
                      scheduled_for: Optional[str] = None) -> None:
        """Called by APScheduler for dispatch-triggered jobs (and on create with run_on_create)."""
        active = self.active_for_dispatch(dispatch_id)
        if active:
            print(f"[dispatch] skipping dispatch_id={dispatch_id}: job already active ({active})")
            return

        agent_name = agent if agent in self.pipeline_configs else "general"
        job_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        if trigger == "scheduled":
            scheduled_for = self._scheduled_for(dispatch_id)

        session_id = db.start_session(MODEL, agent=agent_name, source="dispatch", dispatch_id=dispatch_id)
        db.log_message(session_id, "user", prompt)

        with self._lock:
            self._prune()
            self._jobs[job_id] = {
                "status": "queued",
                "agent": agent_name,
                "dispatch_id": dispatch_id,
                "session_id": session_id,
                "result": None, "error": None,
                "cancel_event": cancel_event,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

        def _run():
            metrics.start(session_id, trigger, queued_at=self._jobs[job_id]["started_at"],
                          scheduled_for=scheduled_for, dispatch_id=dispatch_id)
            tee = TeeStream(sys.stdout, flush_cb=lambda logs: db.set_session_logs(session_id, logs))
            sys.stdout = tee
            with self._lock:
                self._jobs[job_id]["tee"] = tee
            _status, _result, _err = "error", None, None
            answer, _exc, _mstatus = None, None, "error"
            try:
                pipeline = create_pipeline(self.client, session_id, self.pipeline_configs[agent_name], cancel_event)
                answer = pipeline.run(prompt)
                _status = "cancelled" if cancel_event.is_set() else "done"
                _mstatus = _run_status(answer, cancel_event.is_set())
                _result = answer if answer else "[no result — pipeline returned empty]"
                if _status != "cancelled":
                    db.log_message(session_id, "assistant", _result)
            except Exception as e:
                _exc = e
                _err = str(e)
                db.log_message(session_id, "error", _err)
            finally:
                sys.stdout = tee.original
                db.set_session_logs(session_id, tee.getvalue())
                with self._lock:
                    self._jobs[job_id]["status"] = _status
                    self._jobs[job_id]["finished_at"] = time.monotonic()
                    self._jobs[job_id].pop("tee", None)
                    if _status == "done":
                        self._jobs[job_id]["result"] = _result
                    elif _err:
                        self._jobs[job_id]["error"] = _err
                db.end_session(session_id)
                updated = db.update_dispatch_run(dispatch_id)
                if not updated["enabled"]:
                    try:
                        self.scheduler.remove_job(f"dispatch_{dispatch_id}")
                    except Exception:
                        pass
                verdict = metrics.get(session_id).evaluate(_mstatus, answer, _exc)
                if (_status == "done" and _result) or _mstatus == "error":
                    _notify(self.client, session_id, agent, _result, verdict)
                metrics.finish(session_id, _mstatus, answer, _exc)

        self._queue.put((job_id, _run, ()))

    # ── Internal execution ────────────────────────────────────────────────────

    def _execute_http(self, job_id: str, body, cancel_event, session_id, agent_name, inline_config=None):
        if agent_name is None:
            agent_name = "inline" if inline_config else (
                body.agent if body.agent in self.pipeline_configs else "general"
            )
        if session_id is None:
            session_id = db.start_session(body.model, agent=agent_name, source="http", dispatch_id=body.dispatch_id)
            with self._lock:
                self._jobs[job_id]["session_id"] = session_id
            for m in reversed(body.messages):
                if m.role == "user":
                    db.log_message(session_id, "user", m.content)
                    break
        metrics.start(session_id, "manual" if getattr(body, "dispatch_id", None) else "http",
                      queued_at=self._jobs[job_id]["started_at"], dispatch_id=getattr(body, "dispatch_id", None))

        tee = TeeStream(sys.stdout, flush_cb=lambda logs: db.set_session_logs(session_id, logs))
        sys.stdout = tee
        with self._lock:
            self._jobs[job_id]["tee"] = tee
        _status, _result, _err = "error", None, None
        answer, _exc, _mstatus = None, None, "error"
        try:
            query = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
            config = inline_config or self.pipeline_configs.get(agent_name)
            pipeline = create_pipeline(self.client, session_id, config, cancel_event)
            answer = pipeline.run(query)
            _status = "cancelled" if cancel_event.is_set() else "done"
            _mstatus = _run_status(answer, cancel_event.is_set())
            _result = answer if answer else "[no result — pipeline returned empty]"
            if _status != "cancelled":
                db.log_message(session_id, "assistant", _result)
        except Exception as e:
            _exc = e
            _err = str(e)
            db.log_message(session_id, "error", _err)
        finally:
            sys.stdout = tee.original
            db.set_session_logs(session_id, tee.getvalue())
            with self._lock:
                self._jobs[job_id]["status"] = _status
                self._jobs[job_id]["finished_at"] = time.monotonic()
                self._jobs[job_id].pop("tee", None)
                if _status == "done":
                    self._jobs[job_id]["result"] = _result
                elif _err:
                    self._jobs[job_id]["error"] = _err
            db.end_session(session_id)
            verdict = metrics.get(session_id).evaluate(_mstatus, answer, _exc)
            if getattr(body, "dispatch_id", None):
                if (_status == "done" and _result) or _mstatus == "error":
                    _notify(self.client, session_id, agent_name, _result, verdict)
            metrics.finish(session_id, _mstatus, answer, _exc)
