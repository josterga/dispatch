import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

import db
import metrics
from providers import LLMClient
from pipeline import MODEL, OLLAMA_TIMEOUT
from agent_loader import build_registry
from agent_config import DomainConfig
from job_manager import JobManager, cron_trigger

load_dotenv(override=True)

SERVER_API_KEY = os.getenv("SERVER_API_KEY")
HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", "8000"))

PIPELINE_CONFIGS: dict = {}  # populated in lifespan after DB seed

_scheduler = BackgroundScheduler(timezone="UTC")
_client = LLMClient(timeout=OLLAMA_TIMEOUT)
_jobs = JobManager(_client, _scheduler)


def _reload_registry() -> None:
    global PIPELINE_CONFIGS
    PIPELINE_CONFIGS = build_registry()
    _jobs.pipeline_configs = PIPELINE_CONFIGS


def _register_dispatch_job(dispatch: dict) -> None:
    from apscheduler.triggers.date import DateTrigger
    if dispatch.get("trigger_type") == "once":
        run_at = datetime.fromisoformat(dispatch["schedule"])
        trigger = DateTrigger(run_date=run_at, timezone="UTC")
    else:
        # end_date: nothing fires after "repeat until" (update_dispatch_run only archives after a run)
        trigger = cron_trigger(dispatch["schedule"], dispatch.get("timezone"), end_date=dispatch.get("repeat_until"))
    _scheduler.add_job(
        _jobs.run_scheduled,
        trigger,
        args=[dispatch["id"], dispatch["agent"], dispatch["prompt"]],
        id=f"dispatch_{dispatch['id']}",
        replace_existing=True,
        misfire_grace_time=300,  # a run delayed a few minutes (busy box, clock jump) still happens
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _jobs.start_worker()
    db.seed_agents_from_yaml()
    db.seed_default_dispatches()
    _reload_registry()
    for dispatch in db.get_dispatches(enabled_only=True):
        try:
            _register_dispatch_job(dispatch)
        except Exception:
            pass
    _scheduler.start()
    threading.Thread(target=metrics.backfill, daemon=True, name="metrics-backfill").start()
    # Downtime = last heartbeat of the previous process → now; runs due in it were never fired.
    alive_at = db.get_setting("_alive_at")
    _scheduler.add_job(lambda: db.set_setting("_alive_at", datetime.now(timezone.utc).isoformat()),
                       "interval", seconds=60, id="heartbeat", next_run_time=datetime.now(timezone.utc))
    if alive_at:
        threading.Thread(target=_jobs.catch_up_missed, args=(db.get_dispatches(enabled_only=True), alive_at),
                         daemon=True, name="catch-up").start()
    yield
    _scheduler.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL
    messages: list[Message]
    stream: bool = False
    agent: str = "general"
    think: bool = False
    dispatch_id: Optional[int] = None
    goal_template: Optional[str] = None
    output_template: Optional[str] = None
    sources: Optional[list[str]] = None
    freshness: Optional[str] = None

    @field_validator("messages", mode="before")
    @classmethod
    def coerce_single_message(cls, v):
        if isinstance(v, dict):
            return [v]
        return v


class DispatchRequest(BaseModel):
    agent: str = "news"
    prompt: str
    schedule: str
    label: str
    run_on_create: bool = False
    trigger_type: str = "cron"
    repeat_until: Optional[str] = None
    max_runs: Optional[int] = None
    timezone: Optional[str] = None  # IANA zone the cron is written in (the browser's); default: server zone
    tags: list[str] = []  # the dispatch's own; its agent's tags apply too


class DispatchPatch(BaseModel):
    enabled: Optional[bool] = None


def _check_auth(request: Request) -> None:
    if not SERVER_API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {SERVER_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Settings ──────────────────────────────────────────────────────────────────

_SETTINGS_KEYS = [
    "PROVIDER", "MODEL", "FAST_MODEL", "OLLAMA_BASE_URL",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FIREWORKS_API_KEY",
    "OLLAMA_API_KEY", "BRAVE_API_KEY", "JINA_API_KEY", "SEARXNG_URL",
    "LOCATION",
    "CONTEXT_BUDGET", "TEMPERATURE", "TIMEOUT",
]

_ENV_PATH = Path(__file__).parent / ".env"


def _update_env_file(updates: dict[str, str]) -> None:
    lines: list[str] = []
    if _ENV_PATH.exists():
        lines = _ENV_PATH.read_text().splitlines(keepends=True)

    written: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.rstrip("\n")
        if "=" in stripped and not stripped.lstrip().startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}\n")
                written.add(key)
                continue
        new_lines.append(line if line.endswith("\n") else line + "\n")

    for key, value in updates.items():
        if key not in written:
            new_lines.append(f"{key}={value}\n")

    _ENV_PATH.write_text("".join(new_lines))


def _resolve_config(body: ChatRequest) -> tuple[str, Optional[DomainConfig]]:
    if body.goal_template:
        config = DomainConfig(
            agent_name="inline",
            goal_template=body.goal_template,
            output_template=body.output_template or "",
            sources=body.sources or [],
            freshness=body.freshness,
        )
        return "inline", config
    agent_name = body.agent if body.agent in PIPELINE_CONFIGS else "general"
    return agent_name, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def ui():
    ui_path = Path(__file__).parent / "ui.html"
    # The page never contains the API key; the UI asks for it once and keeps it in localStorage.
    return HTMLResponse(ui_path.read_text())


@app.get("/api/sessions")
async def list_sessions(request: Request, limit: int = 50, agent: Optional[str] = None):
    _check_auth(request)
    return db.get_sessions(limit=limit, agent=agent)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: int, request: Request):
    _check_auth(request)
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Session not found")
    db.mark_session_viewed(session_id)
    return detail


@app.get("/api/sessions/{session_id}/events")
async def get_session_events(session_id: int, request: Request):
    _check_auth(request)
    return db.get_run_events(session_id)


@app.get("/api/sessions/{session_id}/payloads/{seq}")
async def get_session_payload(session_id: int, seq: int, request: Request):
    _check_auth(request)
    payload = db.get_run_payload(session_id, seq)
    if not payload:
        raise HTTPException(status_code=404, detail="Payload not found (kept for %d days)" % metrics.PAYLOAD_RETENTION_DAYS)
    return payload


@app.get("/api/stats")
async def get_stats(request: Request, days: int = 7):
    _check_auth(request)
    return metrics.stats(max(1, min(days, 365)))


@app.post("/api/jobs")
async def submit_job(request: Request, body: ChatRequest):
    _check_auth(request)
    agent_name, inline_config = _resolve_config(body)
    cancel_event = threading.Event()
    if body.dispatch_id and _jobs.active_for_dispatch(body.dispatch_id):
        raise HTTPException(status_code=409, detail="This dispatch is already queued or running")

    # Create session immediately so it appears in the sidebar while queued.
    session_id = db.start_session(body.model, agent=agent_name, source="http", viewed=True, dispatch_id=body.dispatch_id)
    for m in reversed(body.messages):
        if m.role == "user":
            db.log_message(session_id, "user", m.content)
            break

    job_id = _jobs.submit(body, cancel_event, session_id, inline_config=inline_config)
    return {"job_id": job_id, "session_id": session_id}


@app.get("/api/jobs/by-session/{session_id}")
async def get_job_by_session(session_id: int, request: Request):
    _check_auth(request)
    result = _jobs.get_by_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="No active job for this session")
    return result


@app.get("/api/jobs/active")
async def list_active_jobs(request: Request):
    _check_auth(request)
    return _jobs.list_active()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    _check_auth(request)
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "queue_position": _jobs.queue_position(job_id),
        "session_id": job["session_id"],
        "result": job["result"],
        "error": job["error"],
        "started_at": job["started_at"],
    }


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    _check_auth(request)
    if not _jobs.cancel(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": "cancelling"}


class AgentRequest(BaseModel):
    name: str
    label: str = ""
    description: str = ""
    enabled: bool = True
    goal_template: str
    output_template: str = ""
    sources: list[str] = []
    sub_queries: Optional[list[dict]] = None
    tool_bindings: list[str] = ["web_search"]
    direct_sources: list[str] = []
    freshness: Optional[str] = None
    planner_query_count: int = 4
    max_validator_loops: int = 1
    validate_links: bool = False
    fetch_content: bool = False
    color: Optional[int] = None
    tags: list[str] = []
    recall_runs: Optional[int] = None  # None = default (2); 0 = off


@app.get("/api/tools")
async def list_tools():
    import tools as _tools
    return [
        {
            "id": t["function"]["name"],
            "description": t["function"]["description"],
        }
        for t in _tools.TOOLS
    ]


@app.get("/api/agents")
async def list_agents(request: Request):
    _check_auth(request)
    return db.get_agents()


@app.get("/api/agents/{name}")
async def get_agent_route(name: str, request: Request):
    _check_auth(request)
    agent = db.get_agent(name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@app.post("/api/agents")
async def create_agent_route(request: Request, body: AgentRequest):
    _check_auth(request)
    import json as _json
    data = body.model_dump()
    data["sources"] = _json.dumps(data["sources"])
    data["tool_bindings"] = _json.dumps(data["tool_bindings"])
    data["direct_sources"] = _json.dumps(data["direct_sources"])
    data["tags"] = _json.dumps(db.clean_tags(data["tags"]))
    if data.get("sub_queries") is not None:
        data["sub_queries"] = _json.dumps(data["sub_queries"])
    data["is_builtin"] = 0
    try:
        agent_id = db.create_agent(data)
    except Exception as e:
        msg = "name already exists" if "UNIQUE" in str(e) else str(e)
        raise HTTPException(status_code=400, detail=msg)
    _reload_registry()
    return {"id": agent_id, "name": body.name}


@app.put("/api/agents/{name}")
async def update_agent_route(name: str, request: Request, body: AgentRequest):
    _check_auth(request)
    import json as _json
    if not db.get_agent(name):
        raise HTTPException(status_code=404, detail="Agent not found")
    data = body.model_dump()
    data["sources"] = _json.dumps(data["sources"])
    data["tool_bindings"] = _json.dumps(data["tool_bindings"])
    data["direct_sources"] = _json.dumps(data["direct_sources"])
    data["tags"] = _json.dumps(db.clean_tags(data["tags"]))
    if data.get("sub_queries") is not None:
        data["sub_queries"] = _json.dumps(data["sub_queries"])
    db.update_agent(name, data)
    _reload_registry()
    return {"name": name}


@app.delete("/api/agents/{name}")
async def delete_agent_route(name: str, request: Request):
    _check_auth(request)
    agent = db.get_agent(name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent.get("is_builtin"):
        raise HTTPException(status_code=403, detail="Cannot delete a built-in agent")
    db.delete_agent(name)
    _reload_registry()
    return {"deleted": name}


@app.get("/api/agents/{name}/export")
async def export_agent_route(name: str, request: Request):
    """Export agent as YAML."""
    _check_auth(request)
    import json as _json
    try:
        import yaml as _yaml
    except ImportError:
        raise HTTPException(status_code=500, detail="pyyaml not installed")
    agent = db.get_agent(name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Decode JSON fields back to Python objects for clean YAML output
    for key in ("sources", "tool_bindings", "direct_sources", "tags"):
        if agent.get(key):
            try:
                agent[key] = _json.loads(agent[key])
            except Exception:
                pass
    if agent.get("sub_queries"):
        try:
            agent["sub_queries"] = _json.loads(agent["sub_queries"])
        except Exception:
            pass
    # Drop DB-internal fields
    for key in ("id", "created_at", "updated_at"):
        agent.pop(key, None)
    from fastapi.responses import Response
    return Response(
        content=_yaml.dump(agent, allow_unicode=True, sort_keys=False),
        media_type="text/yaml",
        headers={"Content-Disposition": f"attachment; filename={name}.yaml"},
    )


@app.post("/api/agents/import")
async def import_agent_route(request: Request):
    """Import agent from YAML body."""
    _check_auth(request)
    import json as _json
    try:
        import yaml as _yaml
    except ImportError:
        raise HTTPException(status_code=500, detail="pyyaml not installed")
    raw = await request.body()
    try:
        data = _yaml.safe_load(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    if not data or not data.get("name"):
        raise HTTPException(status_code=400, detail="Missing required field: name")
    db._normalise_agent_yaml(data)
    data["is_builtin"] = 0
    existing = db.get_agent(data["name"])
    if existing:
        db.update_agent(data["name"], data)
    else:
        db.create_agent(data)
    _reload_registry()
    return {"name": data["name"], "action": "updated" if existing else "created"}


@app.get("/api/dispatches")
async def list_dispatches(request: Request):
    _check_auth(request)
    return db.get_dispatches()


@app.get("/api/dispatches/digest")
async def get_dispatches_digest(request: Request):
    _check_auth(request)
    dispatches = db.get_dispatches(enabled_only=True)
    return [{"dispatch": d, "session": db.get_latest_session_for_dispatch(d["id"])} for d in dispatches]


@app.post("/api/dispatches")
async def create_dispatch_route(request: Request, body: DispatchRequest):
    _check_auth(request)
    tz = body.timezone or db.local_timezone()
    if body.trigger_type == "once":
        try:
            datetime.fromisoformat(body.schedule)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid datetime for once trigger")
    else:
        try:
            cron_trigger(body.schedule, tz)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cron expression or timezone")
    dispatch_id = db.create_dispatch(
        body.agent, body.prompt, body.schedule, body.label, body.run_on_create,
        trigger_type=body.trigger_type, repeat_until=body.repeat_until, max_runs=body.max_runs,
        timezone_name=tz, tags=body.tags,
    )
    dispatch = next((d for d in db.get_dispatches() if d["id"] == dispatch_id), None)
    if dispatch:
        try:
            _register_dispatch_job(dispatch)
        except Exception:
            pass
    if body.run_on_create:
        threading.Thread(target=_jobs.run_scheduled, args=(dispatch_id, body.agent, body.prompt, "run_on_create"), daemon=True).start()
    return {"id": dispatch_id}


@app.put("/api/dispatches/{dispatch_id}")
async def update_dispatch_route(dispatch_id: int, request: Request, body: DispatchRequest):
    _check_auth(request)
    tz = body.timezone or db.local_timezone()
    if body.trigger_type == "once":
        try:
            datetime.fromisoformat(body.schedule)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid datetime for once trigger")
    else:
        try:
            cron_trigger(body.schedule, tz)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cron expression or timezone")
    db.update_dispatch(
        dispatch_id, body.agent, body.prompt, body.schedule, body.label,
        trigger_type=body.trigger_type, repeat_until=body.repeat_until, max_runs=body.max_runs,
        timezone_name=tz, tags=body.tags,
    )
    try:
        _scheduler.remove_job(f"dispatch_{dispatch_id}")
    except Exception:
        pass
    dispatch = next((d for d in db.get_dispatches() if d["id"] == dispatch_id), None)
    if dispatch and dispatch["enabled"]:
        try:
            _register_dispatch_job(dispatch)
        except Exception:
            pass
    return {"id": dispatch_id}


@app.patch("/api/dispatches/{dispatch_id}")
async def patch_dispatch_route(dispatch_id: int, request: Request, body: DispatchPatch):
    _check_auth(request)
    if body.enabled is not None:
        db.toggle_dispatch(dispatch_id, body.enabled)
        if body.enabled:
            dispatch = next((d for d in db.get_dispatches() if d["id"] == dispatch_id), None)
            if dispatch:
                try:
                    _register_dispatch_job(dispatch)
                except Exception:
                    pass
        else:
            try:
                _scheduler.remove_job(f"dispatch_{dispatch_id}")
            except Exception:
                pass
    return {"id": dispatch_id}


@app.delete("/api/dispatches/{dispatch_id}")
async def delete_dispatch_route(dispatch_id: int, request: Request):
    _check_auth(request)
    db.delete_dispatch(dispatch_id)
    try:
        _scheduler.remove_job(f"dispatch_{dispatch_id}")
    except Exception:
        pass
    return {"deleted": dispatch_id}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    """OpenAI-compatible endpoint — queued, blocks until job completes."""
    _check_auth(request)
    _, inline_config = _resolve_config(body)
    cancel_event = threading.Event()
    job = _jobs.submit_and_wait(body, cancel_event, inline_config=inline_config)
    answer = job.get("result") or "[reached max tool iterations without a final answer]"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": body.model,
        "session_id": job.get("session_id"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
    }


@app.get("/v1/models")
async def list_models(request: Request):
    _check_auth(request)
    return {
        "object": "list",
        "data": [{"id": MODEL, "object": "model", "created": int(datetime.now(timezone.utc).timestamp()), "owned_by": "dispatch"}],
    }


@app.get("/api/settings")
async def get_settings_route(request: Request):
    _check_auth(request)
    result = {}
    for key in _SETTINGS_KEYS:
        result[key] = {"value": os.environ.get(key, ""), "source": "env"}
    return result


@app.post("/api/settings")
async def save_settings_route(request: Request):
    _check_auth(request)
    body: dict = await request.json()
    updates = {k: str(v) for k, v in body.items() if k in _SETTINGS_KEYS}
    for key, value in updates.items():
        db.set_setting(key, value)
        os.environ[key] = value
    _update_env_file(updates)
    return {"saved": True, "restart_required": True}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=600)
