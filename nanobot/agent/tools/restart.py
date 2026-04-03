"""Restart gateway tool for the agent loop."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager

logger = logging.getLogger(__name__)


class RestartGatewayTool(Tool):
    """Tool to restart the nanobot gateway process.

    Writes a ``.restart-pending`` marker with the reason, then re-execs
    the current Python interpreter so launchd (or the parent process)
    respawns a fresh gateway instance.
    """

    _PENDING_FILE = ".restart-pending"

    def __init__(self, workspace: Path, subagent_manager: SubagentManager | None = None) -> None:
        self._workspace = workspace
        self._subagent_manager = subagent_manager

    @property
    def name(self) -> str:
        return "restart_gateway"

    @property
    def description(self) -> str:
        return (
            "Restart the nanobot gateway process in-place. "
            "Writes a marker file with the reason, then re-executes the process. "
            "Launchd (or the parent supervisor) will respawn a fresh instance. "
            "Safety checks run as warnings only and never block the restart."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the restart is happening (e.g. 'applied code patch')",
                    "maxLength": 200,
                },
            },
            "required": ["reason"],
        }

    async def execute(self, reason: str, **kwargs: Any) -> str:
        """Execute the restart with optional safety warnings."""
        if not reason or not reason.strip():
            return "Error: 'reason' is required and must be non-empty"

        reason = reason.strip()[:200]

        # --- Safety checks (warnings only, never block) ---
        warnings: list[str] = []

        if self._subagent_manager is not None:
            try:
                count = self._subagent_manager.get_running_count()
                if count > 0:
                    warnings.append(f"⚠️ {count} subagent(s) running, will be orphaned")
            except Exception:
                pass

        try:
            lock_files = list(self._workspace.glob("*.lock"))
            if lock_files:
                names = ", ".join(f.name for f in lock_files[:5])
                warnings.append(f"⚠️ Lock files found: {names}")
        except Exception:
            pass

        # --- Write pending marker ---
        try:
            pending = self._workspace / self._PENDING_FILE
            payload = {
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            pending.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            return f"Error: failed to write pending marker: {exc}"

        logger.info("Gateway restart requested: %s", reason)

        parts: list[str] = []
        if warnings:
            parts.append("\n".join(warnings))
        parts.append("Restarting gateway…")

        # --- Exec (replaces the process) ---
        try:
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])
        except Exception as exc:
            return "\n".join(parts) + f"\nError: os.execv failed: {exc}"

        # Unreachable — os.execv replaces the process.
        return "\n".join(parts)  # pragma: no cover
