"""Restart gateway tool for the agent loop."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager

logger = logging.getLogger(__name__)


def _detect_supervisor() -> str | None:
    """Detect if a process supervisor is managing this process.

    Returns the supervisor name or None if no supervisor detected.
    """
    # systemd (user or system) — multiple detection strategies
    if os.environ.get("NOTIFY_SOCKET"):
        return "systemd"
    try:
        # cgroup v2: service name appears in slice path (e.g. nanobot-gateway.service)
        # cgroup v1: contains literal "systemd"
        with open("/proc/self/cgroup") as f:
            content = f.read()
            if "systemd" in content or ".service" in content:
                return "systemd"
    except (OSError, FileNotFoundError):
        pass

    # launchd (macOS)
    if sys.platform == "darwin":
        # launchd sets this for all managed services
        if os.environ.get("LAUNCH_DAEMON_SOCKET_NAME") or os.environ.get("__LAUNCHD_LAUNCH_ONCE"):
            return "launchd"
        # Also check via launchctl
        if shutil.which("launchctl"):
            import subprocess
            try:
                result = subprocess.run(
                    ["launchctl", "list"],
                    capture_output=True, text=True, timeout=5,
                )
                # If we can query launchctl, it's likely managing us
                if result.returncode == 0 and "nanobot" in result.stdout:
                    return "launchd"
            except Exception:
                pass

    return None


class RestartGatewayTool(Tool):
    """Tool to restart the nanobot gateway process.

    Writes a ``.restart-pending`` marker with the reason, then re-execs
    the current Python interpreter so launchd (or the parent process)
    respawns a fresh gateway instance.

    **Safety**: If no supervisor (systemd/launchd) is detected, the restart
    is **blocked** because os.execv would kill the process permanently.
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
            "Restart the nanobot gateway process. "
            "Writes a marker file with the reason, then sends SIGTERM to self. "
            "Systemd (or the parent supervisor) will respawn a fresh instance. "
            "If no supervisor is detected, restart is blocked to prevent self-kill."
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

        # --- Supervisor check (hard block) ---
        supervisor = _detect_supervisor()
        if supervisor is None:
            logger.warning("restart_gateway blocked: no supervisor detected")
            return (
                "❌ Restart blocked: no supervisor (systemd/launchd) detected.\n"
                "os.execv would kill the process with no way to restart.\n"
                "To fix: set up a systemd user service or launchd agent for nanobot,\n"
                "or restart manually: kill <pid> && nanobot gateway &"
            )

        # --- Safety checks (warnings only, never block) ---
        warnings: list[str] = []
        warnings.append(f"Supervisor: {supervisor}")

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

        logger.info("Gateway restart requested: %s (supervisor: %s)", reason, supervisor)

        parts: list[str] = []
        if warnings:
            parts.append("\n".join(warnings))
        parts.append("Restarting gateway…")

        # --- Signal-based restart (works with systemd) ---
        # os.execv does NOT trigger systemd restart because PID stays alive.
        # SIGTERM lets systemd see process death → Restart=always respawns.
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as exc:
            return "\n".join(parts) + f"\nError: os.kill(SIGTERM) failed: {exc}"

        # Unreachable — process is terminating.
        return "\n".join(parts)  # pragma: no cover
