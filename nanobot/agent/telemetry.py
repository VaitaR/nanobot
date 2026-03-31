"""In-process telemetry collector: per-turn JSONL metrics."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


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
    ) -> None:
        record = {
            "ts": ts,
            "session": session,
            "channel": channel,
            "chat_id": chat_id,
            "model": model,
            "usage": usage,
            "duration_ms": duration_ms,
            "stop_reason": stop_reason,
            "error": error,
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
