"""Restart gateway tool for the agent loop."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

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
    the current Python interpreter.
    """

    _PENDING_FILE = ".restart-pending"

    def __init__(
        self,
        workspace: Path,
        subagent_manager: SubagentManager | None = None,
        bus: MessageBus | None = None,
    ) -> None:
        self._workspace = workspace
        self._subagent_manager = subagent_manager
        self._bus = bus

    @property
    def name(self) -> str:
        return "restart_gateway"

    @property
    def description(self) -> str:
        return (
            "Restart the nanobot gateway process. "
            "Writes a marker file with the reason, then re-execs the current Python process."
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
                "channel": {
                    "type": "string",
                    "description": "Originating channel for the chat to resume after restart.",
                },
                "chat_id": {
                    "type": "string",
                    "description": "Originating chat ID for the chat to resume after restart.",
                },
                "resume_prompt": {
                    "type": "string",
                    "description": "System resume prompt injected after the gateway comes back online.",
                    "maxLength": 1_000,
                },
            },
            "required": ["reason"],
        }

    async def execute(self, reason: str, **kwargs: Any) -> str:
        """Execute the restart with optional safety warnings."""
        if not reason or not reason.strip():
            return "Error: 'reason' is required and must be non-empty"

        reason = reason.strip()[:200]
        channel = str(kwargs.get("channel") or "").strip()
        chat_id = str(kwargs.get("chat_id") or "").strip()
        resume_prompt = str(kwargs.get("resume_prompt") or "").strip()[:1000]

        # --- Safety checks (warnings only, never block) ---
        warnings: list[str] = []
        supervisor = _detect_supervisor()
        if supervisor is not None:
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
                "channel": channel,
                "chat_id": chat_id,
                "resume_prompt": resume_prompt,
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

        try:
            os.execv(sys.executable, [sys.executable, "-m", "nanobot", "gateway"])
        except Exception as exc:
            return "\n".join(parts) + f"\nError: execv failed: {exc}"

        # Unreachable — process image has been replaced.
        return "\n".join(parts)  # pragma: no cover
