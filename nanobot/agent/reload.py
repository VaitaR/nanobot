"""ReloadChecker — safe reload coordinator checked at heartbeat tick start."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.subagent import SubagentManager


class ReloadChecker:
    """Checks for .pending-reload marker at heartbeat tick start.

    When the marker exists and safety conditions are met (no active subagents,
    no in-flight message processing), triggers a clean process restart via
    SIGTERM so the finally block in gateway() runs before systemd respawns.
    """

    def __init__(
        self,
        workspace: Path,
        agent_loop: AgentLoop,
        subagent_manager: SubagentManager | None,
    ) -> None:
        self._workspace = workspace
        self._agent = agent_loop
        self._subagents = subagent_manager

    def check(self) -> bool:
        """Check if reload is pending and safe conditions are met.

        Returns True if reload was triggered (process will exit).
        Returns False if not ready or no pending reload.
        """
        marker = self._workspace / ".pending-reload"
        if not marker.exists():
            return False

        # Read reason for logging
        try:
            data = json.loads(marker.read_text())
            reason = data.get("reason", "unknown")
        except Exception:
            reason = "unknown (parse error)"

        # Safety check: no running subagents
        if self._subagents is not None and self._subagents.get_running_count() > 0:
            logger.info("reload deferred: subagents running")
            return False

        # Safety check: no in-flight message processing
        if hasattr(self._agent, "_active_tasks") and any(self._agent._active_tasks.values()):
            logger.info("reload deferred: active tasks")
            return False

        # All clear — rename marker and trigger restart
        reloading_marker = self._workspace / ".reloading"
        try:
            os.replace(marker, reloading_marker)
        except Exception:
            logger.warning("reload: could not rename .pending-reload to .reloading")

        logger.info("reload triggered: %s", reason)
        os.kill(os.getpid(), signal.SIGTERM)
        return True
