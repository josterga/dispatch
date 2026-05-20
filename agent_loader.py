"""Loads agent definitions from the DB into DomainConfig objects at runtime."""

import json
from typing import Optional

import db
from agent_config import DomainConfig, SubQueryConfig
from pipeline import PLANNER_QUERY_COUNT


def agent_row_to_config(row: dict) -> DomainConfig:
    sub_queries_raw = row.get("sub_queries")
    sub_queries = []
    if sub_queries_raw:
        parsed = json.loads(sub_queries_raw)
        for sq in (parsed if isinstance(parsed, list) else []):
            sub_queries.append(SubQueryConfig(
                label=sq["label"],
                goal=sq["goal"],
                sources=sq.get("sources", []),
                query_count=sq.get("query_count", 4),
                freshness=sq.get("freshness"),
            ))

    return DomainConfig(
        agent_name=row["name"],
        goal_template=row["goal_template"],
        output_template=row.get("output_template", ""),
        sources=json.loads(row.get("sources") or "[]"),
        sub_queries=sub_queries or None,
        freshness=row.get("freshness"),
        planner_query_count=row.get("planner_query_count") or PLANNER_QUERY_COUNT,
        max_validator_loops=row.get("max_validator_loops", 1),
        validate_links=bool(row.get("validate_links", 0)),
        fetch_content=bool(row.get("fetch_content", 0)),
        direct_sources=json.loads(row.get("direct_sources") or "[]"),
        tool_bindings=json.loads(row.get("tool_bindings") or "[]"),
    )


def build_registry() -> dict[str, DomainConfig]:
    """Return {agent_name: DomainConfig} for all enabled agents."""
    rows = db.get_agents(enabled_only=True)
    registry: dict[str, DomainConfig] = {}
    for row in rows:
        try:
            registry[row["name"]] = agent_row_to_config(row)
        except Exception as e:
            print(f"[agent_loader] WARNING: failed to load agent '{row['name']}': {e}")
    return registry
