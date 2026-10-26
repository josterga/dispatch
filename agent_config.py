from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SubQueryConfig:
    label: str
    goal: str
    sources: list[str] = field(default_factory=list)
    query_count: int = 4
    freshness: Optional[str] = None  # "d", "w", "m", "y" — passed to search providers


@dataclass
class DomainConfig:
    agent_name: str
    goal_template: str
    output_template: str
    sources: list[str] = field(default_factory=list)
    max_validator_loops: int = 1
    planner_query_count: int = 4
    validate_links: bool = False
    fetch_content: bool = False
    freshness: Optional[str] = None  # "d", "w", "m", "y" — passed to search providers
    sub_queries: Optional[list[SubQueryConfig]] = None
    direct_sources: list[str] = field(default_factory=list)  # URL templates fetched directly; {today} substituted
    tool_bindings: list[str] = field(default_factory=list)
    recall_runs: int = 2  # previous digests of the same dispatch shown to the synthesizer; 0 = off
