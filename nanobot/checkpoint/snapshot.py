"""Frozen data snapshots used by the checkpoint review system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ToolCallSummary:
    """A single tool-call record for analysis."""

    tool_name: str
    detail: str
    file_path: str | None = None
    iteration: int = 0


@dataclass(slots=True, frozen=True)
class CheckpointSnapshot:
    """Point-in-time snapshot of subagent state for policy evaluation."""

    total_iterations: int
    max_iterations: int
    tool_calls: tuple[ToolCallSummary, ...] = ()
    files_touched: frozenset[str] = frozenset()
    last_llm_outputs: tuple[str, ...] = ()
    error_count: int = 0
    loop_detected: bool = False
    stuck_score: float = 0.0
    task_id: str = ""  # nanobot-tasks ID for callback routing (Phase 3)
