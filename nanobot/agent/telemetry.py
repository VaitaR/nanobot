"""In-process telemetry collector: per-turn JSONL metrics."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from loguru import logger


class ErrorCategory(Enum):
    """Structured error categories for observability."""

    LLM_API = "llm_api"  # LLM provider API errors (rate limit, timeout, auth)
    TOOL_EXECUTION = "tool_exec"  # Tool runtime errors (command failed, file not found)
    VALIDATION = "validation"  # Input/output validation errors
    SESSION = "session"  # Session management errors
    CONFIGURATION = "config"  # Configuration errors
    UNKNOWN = "unknown"  # Unclassified errors


def classify_error(error_str: str | None) -> ErrorCategory:
    """Classify an error string into a category based on heuristics.

    # TODO: improve with pattern matching or ML classification
    """
    if not error_str:
        return ErrorCategory.UNKNOWN
    lower = error_str.lower()
    # Check session and config first — their keywords ("lock", "save", "env", "missing")
    # can overlap with LLM_API ("timeout") and VALIDATION ("invalid") heuristics.
    if any(kw in lower for kw in ("session", "lock", "save", "load")):
        return ErrorCategory.SESSION
    if any(kw in lower for kw in ("config", "setting", "env", "missing")):
        return ErrorCategory.CONFIGURATION
    if any(kw in lower for kw in ("rate limit", "429", "timeout", "connection", "api_error", "api key")):
        return ErrorCategory.LLM_API
    if any(kw in lower for kw in ("command", "execution", "exit code", "permission denied", "file not found")):
        return ErrorCategory.TOOL_EXECUTION
    if any(kw in lower for kw in ("validation", "invalid", "malformed", "schema")):
        return ErrorCategory.VALIDATION
    return ErrorCategory.UNKNOWN


class TelemetryCollector:
    """Append-only JSONL writer for turn-level metrics."""

    def __init__(self, workspace: Path) -> None:
        self._path = Path(workspace) / "memory" / "improvement-log.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record_turn(
        self,
        *,
        ts: str,
        session: str,
        channel: str,
        chat_id: str,
        model: str,
        usage: dict,
        duration_ms: int,
        stop_reason: str,
        error: str | None,
        tools_used: list[str],
        skills: list[str],
        files_touched: list[str],
        request_id: str | None = None,  # OBS-001
        estimated_cost_usd: float | None = None,
    ) -> None:
        record = {
            "ts": ts,
            "session": session,
            "channel": channel,
            "chat_id": chat_id,
            "model": model,
            "request_id": request_id,  # OBS-001
            "usage": usage,
            "estimated_cost_usd": estimated_cost_usd,
            # OBS-003: cumulative token totals at top level
            "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "duration_ms": duration_ms,
            "stop_reason": stop_reason,
            "error": error,
            "error_category": classify_error(error).value if error else None,  # OBS-002
            "tools": sorted(set(tools_used)),
            "skills": sorted(set(skills)),
            "files_touched": sorted(set(files_touched)),
        }
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Failed to write telemetry record", exc_info=True)

    @staticmethod
    def extract_from_events(tool_events: list[dict]) -> tuple[list[str], list[str]]:
        """Extract skill names and files_touched from tool_events.

        Returns (skills, files_touched).
        """
        skills: list[str] = []
        files_touched: list[str] = []
        for ev in tool_events:
            name = ev.get("name", "")
            args = ev.get("arguments", {}) or {}
            if name == "read_file":
                path = args.get("path", "")
                if "/SKILL.md" in path:
                    skills.append(Path(path).parent.name)
            elif name in ("write_file", "edit_file"):
                path = args.get("path", "")
                if path:
                    files_touched.append(path)
        return skills, files_touched
