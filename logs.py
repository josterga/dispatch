#!/usr/bin/env python3
"""
Display session logs from search_history.db.

Usage:
  python logs.py              # last 20 sessions
  python logs.py -n 50        # last 50 sessions
  python logs.py --agent news # filter by agent
  python logs.py --session 42 # show full detail for one session
"""
import argparse
import sqlite3
import textwrap
from pathlib import Path

DB_PATH = Path(__file__).parent / "search_history.db"
WIDTH = 100


def divider(char="─"):
    print(char * WIDTH)


def wrap(text: str, indent: int = 4) -> str:
    if not text:
        return ""
    prefix = " " * indent
    return textwrap.fill(text, width=WIDTH - indent, initial_indent=prefix, subsequent_indent=prefix)


def show_sessions(agent: str | None, limit: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            s.id,
            s.agent,
            s.model,
            s.started_at,
            (SELECT content FROM messages WHERE session_id = s.id AND role = 'user'   ORDER BY id LIMIT 1) AS prompt,
            (SELECT content FROM messages WHERE session_id = s.id AND role = 'assistant' ORDER BY id DESC LIMIT 1) AS response,
            (SELECT COUNT(*) FROM tool_calls WHERE session_id = s.id) AS tool_count
        FROM sessions s
        {where}
        ORDER BY s.id DESC
        LIMIT ?
    """
    where = "WHERE s.agent = ?" if agent else ""
    params = (agent, limit) if agent else (limit,)
    rows = conn.execute(query.format(where=where), params).fetchall()

    if not rows:
        print("No sessions found.")
        return

    for row in reversed(rows):
        divider("═")
        agent_label = (row["agent"] or "general").upper()
        print(f"  [{row['id']}]  {agent_label}  ·  {row['started_at']}  ·  {row['model']}  ·  {row['tool_count']} tool calls")
        divider()
        print("  PROMPT")
        print(wrap(row["prompt"] or "(none)"))
        print()
        print("  RESPONSE")
        print(wrap(row["response"] or "(none)"))

    divider("═")
    print(f"  {len(rows)} session(s) shown")
    divider("═")
    conn.close()


def show_session_detail(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        print(f"Session {session_id} not found.")
        return

    divider("═")
    print(f"  SESSION {session_id}  ·  {(session['agent'] or 'general').upper()}  ·  {session['started_at']}  ·  {session['model']}")

    messages = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()

    tool_calls = conn.execute(
        "SELECT tool_name, args, result, duration_ms, created_at FROM tool_calls WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()

    divider()
    print("  MESSAGES")
    for m in messages:
        print(f"\n  [{m['role'].upper()}]  {m['created_at']}")
        print(wrap(m["content"] or "(empty)"))

    if tool_calls:
        divider()
        print("  TOOL CALLS")
        for t in tool_calls:
            print(f"\n  [{t['tool_name']}]  {t['created_at']}  ({t['duration_ms']}ms)")
            print(wrap(t["args"], indent=4))

    divider("═")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View dispatch session logs")
    parser.add_argument("-n", type=int, default=20, help="Number of sessions to show (default: 20)")
    parser.add_argument("--agent", type=str, default=None, help="Filter by agent name")
    parser.add_argument("--session", type=int, default=None, help="Show full detail for a session ID")
    args = parser.parse_args()

    if args.session:
        show_session_detail(args.session)
    else:
        show_sessions(agent=args.agent, limit=args.n)
