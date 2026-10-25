import io
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from queue import Queue
from typing import Optional

import db
import metrics
import notify
from pipeline import MODEL, create_pipeline


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


def _notify(client, session_id: int, title: str, result: str) -> None:
    """Push summary + ntfy, recorded as one "notify" event of the run."""
    t0 = time.monotonic()
    summary = notify.summarize(client, result, session_id)
    summary_ms = int((time.monotonic() - t0) * 1000)
    status, err = notify.send(title=title, body=summary)
    metrics.get(session_id).event("notify", "notify", title, start=t0, ok=status is not None and status < 400,
                                  error=err, summary_ms=summary_ms, http_status=status, chars=len(summary))


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
                    "queue_position": j.get("queue_position") if j["status"] == "queued" else None,
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
            self._jobs[job_id] = {
                "status": "queued",
                "agent": agent_name,
                "dispatch_id": getattr(body, "dispatch_id", None),
                "queue_position": self._queue.qsize() + 1,
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
            self._jobs[job_id] = {
                "status": "queued",
                "queue_position": self._queue.qsize() + 1,
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

    def run_scheduled(self, dispatch_id: int, agent: str, prompt: str, trigger: str = "scheduled") -> None:
        """Called by APScheduler for dispatch-triggered jobs (and on create with run_on_create)."""
        with self._lock:
            for job in self._jobs.values():
                if job.get("dispatch_id") == dispatch_id and job["status"] in ("queued", "running"):
                    print(f"[dispatch] skipping dispatch_id={dispatch_id}: job already active ({job['status']})")
                    return

        agent_name = agent if agent in self.pipeline_configs else "general"
        job_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        scheduled_for = self._scheduled_for(dispatch_id) if trigger == "scheduled" else None

        session_id = db.start_session(MODEL, agent=agent_name, source="dispatch", dispatch_id=dispatch_id)
        db.log_message(session_id, "user", prompt)

        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "agent": agent_name,
                "dispatch_id": dispatch_id,
                "queue_position": self._queue.qsize() + 1,
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
                _status = "done"
                _mstatus = _run_status(answer, cancel_event.is_set())
                _result = answer if answer else "[no result — pipeline returned empty]"
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
                metrics.get(session_id).mark_ended()
                if _status == "done" and _result:
                    _notify(self.client, session_id, agent, _result)
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
                self._jobs[job_id].pop("tee", None)
                if _status == "done":
                    self._jobs[job_id]["result"] = _result
                elif _err:
                    self._jobs[job_id]["error"] = _err
            db.end_session(session_id)
            metrics.get(session_id).mark_ended()
            if getattr(body, "dispatch_id", None):
                if _status == "done" and _result:
                    _notify(self.client, session_id, agent_name, _result)
            metrics.finish(session_id, _mstatus, answer, _exc)
