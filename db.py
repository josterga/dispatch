import json
import os
import random
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

DB_PATH = Path(os.getenv("DISPATCH_DB_PATH", str(Path(__file__).parent / "search_history.db")))

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not getattr(_local, "conn", None):
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(_local.conn)
    return _local.conn


def _init_schema(conn: sqlite3.Connection) -> None:
    # Migration: rename crons → dispatches (runs once on old DBs, no-op on new)
    try:
        conn.execute("ALTER TABLE crons RENAME TO dispatches")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            model      TEXT    NOT NULL,
            agent      TEXT,
            source     TEXT,
            started_at TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            role       TEXT    NOT NULL,
            content    TEXT,
            created_at TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tool_calls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL REFERENCES sessions(id),
            tool_name   TEXT    NOT NULL,
            args        TEXT    NOT NULL,
            result      TEXT,
            duration_ms INTEGER,
            created_at  TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dispatches (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            agent          TEXT    NOT NULL,
            prompt         TEXT    NOT NULL,
            schedule       TEXT    NOT NULL,
            label          TEXT    NOT NULL,
            enabled        INTEGER NOT NULL DEFAULT 1,
            run_on_create  INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS agents (
            id                           INTEGER PRIMARY KEY AUTOINCREMENT,
            name                         TEXT    NOT NULL UNIQUE,
            label                        TEXT    NOT NULL DEFAULT '',
            description                  TEXT    DEFAULT '',
            tier                         INTEGER DEFAULT 1,
            enabled                      INTEGER DEFAULT 1,
            goal_template                TEXT    NOT NULL DEFAULT '',
            output_template              TEXT    DEFAULT '',
            system_prompt                TEXT    DEFAULT '',
            sources                      TEXT    DEFAULT '[]',
            sub_queries                  TEXT    DEFAULT NULL,
            tool_bindings                TEXT    DEFAULT '["web_search"]',
            direct_sources               TEXT    DEFAULT NULL,
            freshness                    TEXT    DEFAULT NULL,
            planner_query_count          INTEGER DEFAULT 4,
            max_validator_loops          INTEGER DEFAULT 1,
            validate_links               INTEGER DEFAULT 0,
            fetch_content                INTEGER DEFAULT 0,
            always_research              INTEGER DEFAULT 1,
            fast_model                   TEXT    DEFAULT '',
            recall_agent                 TEXT    DEFAULT '',
            synthesizer_snippet_chars    INTEGER DEFAULT -1,
            vault_agent                  INTEGER DEFAULT 0,
            query_builder_plugin         TEXT    DEFAULT NULL,
            query_decompose_prompt       TEXT    DEFAULT NULL,
            post_search_extractor_plugin TEXT    DEFAULT NULL,
            is_builtin                   INTEGER DEFAULT 0,
            created_at                   TEXT    DEFAULT (datetime('now')),
            updated_at                   TEXT    DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    # Migrate columns that may be absent in older databases before creating indexes
    for col, typedef in [
        ("color", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE agents ADD COLUMN {col} {typedef}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    for col, typedef in [("agent", "TEXT"), ("source", "TEXT"), ("ended_at", "TEXT"), ("viewed_at", "TEXT"), ("logs", "TEXT"), ("cron_id", "INTEGER"), ("dispatch_id", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {typedef}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    # Backfill dispatch_id from cron_id for sessions created before the rename
    try:
        conn.execute("UPDATE sessions SET dispatch_id = cron_id WHERE cron_id IS NOT NULL AND dispatch_id IS NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    for col, typedef in [
        ("trigger_type", "TEXT NOT NULL DEFAULT 'cron'"),
        ("repeat_until",  "TEXT"),
        ("max_runs",      "INTEGER"),
        ("run_count",     "INTEGER NOT NULL DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE dispatches ADD COLUMN {col} {typedef}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_sessions_agent        ON sessions(agent);
        CREATE INDEX IF NOT EXISTS idx_sessions_started_at   ON sessions(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_session_role ON messages(session_id, role);
    """)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_session(model: str, agent: str = "general", source: str = "cli", viewed: bool = False, dispatch_id: Optional[int] = None) -> int:
    conn = _get_conn()
    viewed_at = _now() if viewed else None
    cur = conn.execute(
        "INSERT INTO sessions (model, agent, source, started_at, viewed_at, dispatch_id) VALUES (?, ?, ?, ?, ?, ?)",
        (model, agent, source, _now(), viewed_at, dispatch_id),
    )
    conn.commit()
    return cur.lastrowid


def mark_session_viewed(session_id: int) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET viewed_at = ? WHERE id = ? AND viewed_at IS NULL",
        (_now(), session_id),
    )
    conn.commit()


def end_session(session_id: int) -> None:
    conn = _get_conn()
    conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id))
    conn.commit()


def log_message(session_id: int, role: str, content: str) -> int:
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, _now()),
    )
    conn.commit()
    return cur.lastrowid


def log_tool_call(
    session_id: int,
    tool_name: str,
    args: dict,
    result: dict,
    duration_ms: int,
) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO tool_calls
               (session_id, tool_name, args, result, duration_ms, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, tool_name, json.dumps(args), json.dumps(result), duration_ms, _now()),
    )
    conn.commit()


def get_sessions(limit: int = 50, agent: Optional[str] = None) -> list[dict]:
    conn = _get_conn()
    where = "WHERE s.agent = ?" if agent else ""
    params = (agent, limit) if agent else (limit,)
    rows = conn.execute(
        f"""
        SELECT
            s.id, s.agent, s.source, s.model, s.started_at,
            (s.viewed_at IS NOT NULL) AS viewed,
            (SELECT content FROM messages WHERE session_id = s.id AND role = 'user' ORDER BY id LIMIT 1) AS prompt,
            (SELECT COUNT(*) FROM tool_calls WHERE session_id = s.id) AS tool_count,
            CASE
                WHEN s.ended_at IS NOT NULL
                THEN CAST(ROUND((JULIANDAY(s.ended_at) - JULIANDAY(s.started_at)) * 86400000) AS INTEGER)
                ELSE (SELECT COALESCE(SUM(duration_ms), 0) FROM tool_calls WHERE session_id = s.id)
            END AS total_duration_ms
        FROM sessions s
        {where}
        ORDER BY s.id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def set_session_logs(session_id: int, logs: str) -> None:
    conn = _get_conn()
    conn.execute("UPDATE sessions SET logs = ? WHERE id = ?", (logs, session_id))
    conn.commit()


def get_session_detail(session_id: int) -> Optional[dict]:
    conn = _get_conn()
    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        return None
    messages = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    tool_calls = conn.execute(
        "SELECT tool_name, args, result, duration_ms, created_at FROM tool_calls WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return {
        **dict(session),
        "messages": [dict(m) for m in messages],
        "tool_calls": [dict(t) for t in tool_calls],
    }


def get_latest_session_for_dispatch(dispatch_id: int) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, agent, started_at, ended_at FROM sessions WHERE dispatch_id = ? ORDER BY id DESC LIMIT 1",
        (dispatch_id,),
    ).fetchone()
    if not row:
        return None
    s = dict(row)
    msg = conn.execute(
        "SELECT content FROM messages WHERE session_id = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1",
        (s["id"],),
    ).fetchone()
    s["content"] = msg["content"] if msg else None
    return s


def create_dispatch(
    agent: str,
    prompt: str,
    schedule: str,
    label: str,
    run_on_create: bool = False,
    trigger_type: str = "cron",
    repeat_until: Optional[str] = None,
    max_runs: Optional[int] = None,
) -> int:
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO dispatches
               (agent, prompt, schedule, label, enabled, run_on_create, created_at,
                trigger_type, repeat_until, max_runs, run_count)
           VALUES (?,?,?,?,1,?,?,?,?,?,0)""",
        (agent, prompt, schedule, label, int(run_on_create), _now(),
         trigger_type, repeat_until, max_runs),
    )
    conn.commit()
    return cur.lastrowid


def get_dispatches(enabled_only: bool = False) -> list[dict]:
    conn = _get_conn()
    where = "WHERE enabled = 1" if enabled_only else ""
    rows = conn.execute(f"SELECT * FROM dispatches {where} ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def delete_dispatch(dispatch_id: int) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM dispatches WHERE id = ?", (dispatch_id,))
    conn.commit()


def update_dispatch_run(dispatch_id: int) -> dict:
    conn = _get_conn()
    conn.execute("UPDATE dispatches SET run_count = run_count + 1 WHERE id = ?", (dispatch_id,))
    conn.commit()
    row = dict(conn.execute("SELECT * FROM dispatches WHERE id = ?", (dispatch_id,)).fetchone())
    should_disable = row.get("trigger_type") == "once"
    if row.get("max_runs") is not None and row["run_count"] >= row["max_runs"]:
        should_disable = True
    if row.get("repeat_until") is not None and _now() >= row["repeat_until"]:
        should_disable = True
    if should_disable:
        conn.execute("UPDATE dispatches SET enabled = 0 WHERE id = ?", (dispatch_id,))
        conn.commit()
        row["enabled"] = 0
    return row


def toggle_dispatch(dispatch_id: int, enabled: bool) -> None:
    conn = _get_conn()
    conn.execute("UPDATE dispatches SET enabled = ? WHERE id = ?", (int(enabled), dispatch_id))
    conn.commit()


def update_dispatch(
    dispatch_id: int,
    agent: str,
    prompt: str,
    schedule: str,
    label: str,
    trigger_type: str = "cron",
    repeat_until: Optional[str] = None,
    max_runs: Optional[int] = None,
) -> None:
    conn = _get_conn()
    conn.execute(
        """UPDATE dispatches
           SET agent=?, prompt=?, schedule=?, label=?,
               trigger_type=?, repeat_until=?, max_runs=?
           WHERE id=?""",
        (agent, prompt, schedule, label, trigger_type, repeat_until, max_runs, dispatch_id),
    )
    conn.commit()


def search_sessions(
    agent: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    conn = _get_conn()
    conditions: list[str] = []
    params: list = []
    if agent:
        conditions.append("s.agent = ?")
        params.append(agent)
    if keyword:
        conditions.append(
            "(EXISTS (SELECT 1 FROM messages WHERE session_id = s.id AND role = 'user'      AND content LIKE ?)"
            " OR EXISTS (SELECT 1 FROM messages WHERE session_id = s.id AND role = 'assistant' AND content LIKE ?))"
        )
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(min(limit, 20))
    rows = conn.execute(
        f"""
        SELECT
            s.id, s.agent, s.source, s.started_at,
            (SELECT content FROM messages WHERE session_id = s.id AND role = 'user'      ORDER BY id LIMIT 1) AS prompt,
            (SELECT content FROM messages WHERE session_id = s.id AND role = 'assistant' ORDER BY id LIMIT 1) AS response
        FROM sessions s
        {where}
        ORDER BY s.id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("response") and len(d["response"]) > 600:
            d["response"] = d["response"][:600] + "…"
        results.append(d)
    return results


_DEFAULT_DISPATCHES = [
    {
        "agent":    "news",
        "prompt":   "Find today's most important news stories from the last 24 hours. Cover: breaking/major news, viral stories, events affecting many people, surprising or unusual stories, and local SF/Bay Area/California news.",
        "schedule": "0 7 * * *",
        "label":    "Daily news — 7am",
    },
    {
        "agent":    "reddit",
        "prompt":   "Run the Reddit monitor: digest configured subreddits and search the keyword list.",
        "schedule": "30 7 * * *",
        "label":    "Reddit digest — 7:30am",
    },
    {
        "agent":    "ai",
        "prompt":   "Find the latest developments in AI/ML: recent research papers, technology breakthroughs, business and product trends, and operational/MLOps/LLMOps challenges.",
        "schedule": "0 9 * * 2",
        "label":    "AI briefing — Tue 9am",
    },
    {
        "agent":    "local-events",
        "prompt":   "Find upcoming SF events this week, this month, and this year; active local SF communities; and notable new or reopening SF restaurants.",
        "schedule": "0 9 * * 4",
        "label":    "SF events — Thu 9am",
    },
    {
        "agent":    "media",
        "prompt":   "Find what movies, TV shows, streaming titles, and video games are newly released or playing this week in SF and on major streaming platforms.",
        "schedule": "0 10 * * 6",
        "label":    "Media — Sat 10am",
    },
    {
        "agent":    "camping",
        "prompt":   "Find available campsite reservations in California over the next 60 days to 6 months.",
        "schedule": "0 10 * * 0",
        "label":    "Camping — Sun 10am",
    },
]


def seed_default_dispatches() -> None:
    """No-op — reserved for future onboarding flows."""
    pass


_AGENT_FIELDS = [
    "name", "label", "description", "enabled",
    "goal_template", "output_template",
    "sources", "sub_queries", "tool_bindings", "direct_sources",
    "freshness", "planner_query_count", "max_validator_loops",
    "validate_links", "fetch_content",
    "is_builtin", "color", "created_at", "updated_at",
]


def get_agents(enabled_only: bool = False) -> list[dict]:
    conn = _get_conn()
    where = "WHERE enabled = 1" if enabled_only else ""
    rows = conn.execute(f"SELECT * FROM agents {where} ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_agent(name: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def create_agent(data: dict) -> int:
    conn = _get_conn()
    if "color" not in data or data["color"] is None:
        data["color"] = random.randint(0, 359)
    fields = [f for f in _AGENT_FIELDS if f in data and f not in ("created_at", "updated_at")]
    placeholders = ", ".join("?" for _ in fields)
    col_list = ", ".join(fields)
    values = [data[f] for f in fields]
    cur = conn.execute(
        f"INSERT INTO agents ({col_list}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    return cur.lastrowid


def update_agent(name: str, data: dict) -> None:
    conn = _get_conn()
    fields = [f for f in _AGENT_FIELDS if f in data and f not in ("name", "created_at", "updated_at", "is_builtin")]
    if not fields:
        return
    set_clause = ", ".join(f"{f} = ?" for f in fields)
    values = [data[f] for f in fields] + [_now(), name]
    conn.execute(
        f"UPDATE agents SET {set_clause}, updated_at = ? WHERE name = ?",
        values,
    )
    conn.commit()


def delete_agent(name: str) -> None:
    conn = _get_conn()
    conn.execute("DELETE FROM agents WHERE name = ? AND is_builtin = 0", (name,))
    conn.commit()


def seed_agents_from_yaml(seeds_dir: Optional[Path] = None) -> None:
    """Upsert agent definitions from YAML files into the DB.

    Files that already exist in the DB are skipped unless the YAML has changed
    (detected by comparing goal_template + sources as a lightweight fingerprint).
    Built-in agents (is_builtin=1) are always refreshed from YAML on startup so
    edits to seed files take effect without manual DB surgery.
    """
    if not _YAML_AVAILABLE:
        return
    if seeds_dir is None:
        env_dir = os.getenv("DISPATCH_AGENTS_DIR")
        seeds_dir = Path(env_dir) if env_dir else Path(__file__).parent / "agents"
    if not seeds_dir.is_dir():
        return

    conn = _get_conn()
    for path in sorted(seeds_dir.glob("*.yaml")):
        try:
            data = _yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not data or not isinstance(data, dict) or not data.get("name"):
            continue

        _normalise_agent_yaml(data)

        existing = conn.execute(
            "SELECT id, is_builtin FROM agents WHERE name = ?", (data["name"],)
        ).fetchone()

        if existing is None:
            # New agent — insert; assign a random color if not specified in YAML
            if "color" not in data or data["color"] is None:
                data["color"] = random.randint(0, 359)
            fields = [f for f in _AGENT_FIELDS if f in data and f not in ("created_at", "updated_at")]
            placeholders = ", ".join("?" for _ in fields)
            col_list = ", ".join(fields)
            conn.execute(
                f"INSERT INTO agents ({col_list}) VALUES ({placeholders})",
                [data[f] for f in fields],
            )
        elif data.get("is_builtin", 1):
            # Built-in agent that already exists — refresh from YAML but preserve user-set color
            fields = [f for f in _AGENT_FIELDS if f in data and f not in ("name", "created_at", "is_builtin", "color")]
            if fields:
                set_clause = ", ".join(f"{f} = ?" for f in fields)
                conn.execute(
                    f"UPDATE agents SET {set_clause} WHERE name = ?",
                    [data[f] for f in fields] + [data["name"]],
                )
        # User-created agents (is_builtin=0) are never overwritten by seed runs

    conn.commit()


def _normalise_agent_yaml(data: dict) -> None:
    """Convert YAML parsed dict into DB-ready types in-place."""
    data.setdefault("is_builtin", 1)
    # JSON-encode list/dict fields
    for key in ("sources", "tool_bindings", "direct_sources"):
        v = data.get(key)
        if isinstance(v, list):
            data[key] = json.dumps(v)
        elif v is None and key in ("sources", "tool_bindings"):
            data[key] = "[]"
    if "sub_queries" in data and isinstance(data["sub_queries"], list):
        data["sub_queries"] = json.dumps(data["sub_queries"])
    # Booleans → ints
    for key in ("enabled", "validate_links", "fetch_content"):
        if key in data:
            data[key] = int(bool(data[key]))


def get_settings() -> dict[str, str]:
    conn = _get_conn()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    conn = _get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
        (key, value, _now()),
    )
    conn.commit()


def close() -> None:
    conn = getattr(_local, "conn", None)
    if conn:
        conn.close()
        _local.conn = None
