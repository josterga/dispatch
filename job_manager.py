import io
import sys
import threading
import uuid
from datetime import datetime, timezone
from queue import Queue
from typing import Optional

import db
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
                    "queue_position": j.get("queue_position") if j["status"] == "queued" else None,
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

    def submit(self, body, cancel_event, session_id: int) -> str:
        """Queue an HTTP-originated job. session_id must be pre-created by the caller."""
        job_id = uuid.uuid4().hex
        agent_name = body.agent if body.agent in self.pipeline_configs else "general"

        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "queue_position": self._queue.qsize() + 1,
                "session_id": session_id,
                "result": None, "error": None,
                "cancel_event": cancel_event,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        self._queue.put((job_id, self._execute_http, (job_id, body, cancel_event, session_id, agent_name)))
        return job_id

    def submit_and_wait(self, body, cancel_event) -> dict:
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
            self._execute_http(job_id, body, cancel_event, session_id=None, agent_name=None)
            done_event.set()

        self._queue.put((job_id, _run, ()))
        done_event.wait()
        with self._lock:
            return self._jobs[job_id]

    def run_scheduled(self, dispatch_id: int, agent: str, prompt: str) -> None:
        """Called by APScheduler for dispatch-triggered jobs."""
        with self._lock:
            for job in self._jobs.values():
                if job.get("dispatch_id") == dispatch_id and job["status"] in ("queued", "running"):
                    print(f"[dispatch] skipping dispatch_id={dispatch_id}: job already active ({job['status']})")
                    return

        agent_name = agent if agent in self.pipeline_configs else "general"
        job_id = uuid.uuid4().hex
        cancel_event = threading.Event()

        session_id = db.start_session(MODEL, agent=agent_name, source="dispatch", dispatch_id=dispatch_id)
        db.log_message(session_id, "user", prompt)

        with self._lock:
            self._jobs[job_id] = {
                "status": "queued",
                "dispatch_id": dispatch_id,
                "queue_position": self._queue.qsize() + 1,
                "session_id": session_id,
                "result": None, "error": None,
                "cancel_event": cancel_event,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

        def _run():
            tee = TeeStream(sys.stdout, flush_cb=lambda logs: db.set_session_logs(session_id, logs))
            sys.stdout = tee
            _status, _result, _err = "error", None, None
            try:
                pipeline = create_pipeline(self.client, session_id, self.pipeline_configs[agent_name], cancel_event)
                answer = pipeline.run(prompt)
                _status = "done"
                _result = answer if answer else "[no result — pipeline returned empty]"
                db.log_message(session_id, "assistant", _result)
            except Exception as e:
                _err = str(e)
                db.log_message(session_id, "error", _err)
            finally:
                sys.stdout = tee.original
                db.set_session_logs(session_id, tee.getvalue())
                with self._lock:
                    self._jobs[job_id]["status"] = _status
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
                label = prompt[:60] + ("…" if len(prompt) > 60 else "")
                if _status == "done" and _result:
                    snippet = _result[:300].strip() + ("…" if len(_result) > 300 else "")
                    notify.send(
                        title=f"[{agent}] Run complete",
                        body=f"{label}\n\n{snippet}",
                        tags=["white_check_mark"],
                        priority="default",
                    )
                elif _status == "error":
                    notify.send(
                        title=f"[{agent}] Run failed",
                        body=f"{label}\n\nError: {_err or 'unknown'}",
                        tags=["x"],
                        priority="high",
                    )

        self._queue.put((job_id, _run, ()))

    # ── Internal execution ────────────────────────────────────────────────────

    def _execute_http(self, job_id: str, body, cancel_event, session_id, agent_name):
        if agent_name is None:
            agent_name = body.agent if body.agent in self.pipeline_configs else "general"
        if session_id is None:
            session_id = db.start_session(body.model, agent=agent_name, source="http", dispatch_id=body.dispatch_id)
            with self._lock:
                self._jobs[job_id]["session_id"] = session_id
            for m in reversed(body.messages):
                if m.role == "user":
                    db.log_message(session_id, "user", m.content)
                    break

        tee = TeeStream(sys.stdout, flush_cb=lambda logs: db.set_session_logs(session_id, logs))
        sys.stdout = tee
        _status, _result, _err = "error", None, None
        try:
            query = next((m.content for m in reversed(body.messages) if m.role == "user"), "")
            pipeline = create_pipeline(self.client, session_id, self.pipeline_configs[agent_name], cancel_event)
            answer = pipeline.run(query)
            _status = "cancelled" if cancel_event.is_set() else "done"
            _result = answer if answer else "[no result — pipeline returned empty]"
            if _status != "cancelled":
                db.log_message(session_id, "assistant", _result)
        except Exception as e:
            _err = str(e)
            db.log_message(session_id, "error", _err)
        finally:
            sys.stdout = tee.original
            db.set_session_logs(session_id, tee.getvalue())
            with self._lock:
                self._jobs[job_id]["status"] = _status
                if _status == "done":
                    self._jobs[job_id]["result"] = _result
                elif _err:
                    self._jobs[job_id]["error"] = _err
            db.end_session(session_id)
