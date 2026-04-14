"""ReloadChecker — safe reload coordinator checked at heartbeat tick start."""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.subagent import SubagentManager

_STALE_HOURS = 2.0


@dataclass
class ReloadState:
    """Structured reload marker content."""

    reason: str
    subagent_id: str | None = None
    task_id: str | None = None
    files_modified: list[str] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    timestamp: str = ""


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
        self.pending_reload = workspace / ".pending-reload"
        self.reloading = workspace / ".reloading"

    # ── Public API ────────────────────────────────────────────────────────────

    def read_pending(self) -> ReloadState | None:
        """Read .pending-reload marker. Returns None if missing or invalid JSON."""
        if not self.pending_reload.exists():
            return None
        try:
            data = json.loads(self.pending_reload.read_text())
            return ReloadState(
                reason=data.get("reason", "unknown"),
                subagent_id=data.get("subagent_id"),
                task_id=data.get("task_id"),
                files_modified=data.get("files_modified", []),
                verification=data.get("verification", {}),
                timestamp=data.get("timestamp", ""),
            )
        except Exception:
            logger.warning("reload: could not parse .pending-reload")
            return None

    def is_reload_safe(self, running_tasks_count: int, active_tasks_count: int) -> bool:
        """Check safety conditions: no active subagents, no in-flight message processing."""
        if running_tasks_count > 0:
            logger.info("reload deferred: %d subagent(s) running", running_tasks_count)
            return False
        if active_tasks_count > 0:
            logger.info("reload deferred: %d active task(s) in flight", active_tasks_count)
            return False
        return True

    def prepare_reload(self, state: ReloadState) -> bool:
        """Rename .pending-reload → .reloading atomically. Returns False on failure."""
        try:
            os.replace(self.pending_reload, self.reloading)
            return True
        except Exception:
            logger.warning("reload: could not rename .pending-reload to .reloading")
            return False

    def execute_reload(self, state: ReloadState) -> None:
        """Send SIGTERM to own process. Triggers graceful shutdown; systemd respawns."""
        logger.info(
            "reload triggered: %s (files: %s)",
            state.reason,
            state.files_modified or "none listed",
        )
        os.kill(os.getpid(), signal.SIGTERM)

    def check_age(self, state: ReloadState) -> tuple[bool, str | None]:
        """Check if .pending-reload is stale (>2 h). Returns (is_stale, warning_message)."""
        if not state.timestamp:
            return False, None
        try:
            ts = datetime.fromisoformat(state.timestamp.replace("Z", "+00:00"))
            age_h = (datetime.now(UTC) - ts).total_seconds() / 3600
            if age_h >= _STALE_HOURS:
                msg = f"reload pending for {age_h:.1f}h, safety checks not passing"
                return True, msg
        except Exception:
            pass
        return False, None

    def clear_markers(self) -> None:
        """Delete both .pending-reload and .reloading if present (startup cleanup)."""
        self.pending_reload.unlink(missing_ok=True)
        self.reloading.unlink(missing_ok=True)

    # ── Heartbeat integration ─────────────────────────────────────────────────

    def check(self) -> bool:
        """Check for pending reload and execute if safety conditions are met.

        Returns True if reload was triggered (process will exit soon).
        Returns False if no pending reload or conditions not met.
        """
        state = self.read_pending()
        if state is None:
            return False

        is_stale, warning = self.check_age(state)
        if is_stale:
            if warning:
                logger.warning(warning)
            return False

        running = self._subagents.get_running_count() if self._subagents is not None else 0
        active = (
            sum(len(v) for v in self._agent._active_tasks.values())
            if hasattr(self._agent, "_active_tasks")
            else 0
        )

        if not self.is_reload_safe(running, active):
            return False

        if not self.prepare_reload(state):
            return False

        self.execute_reload(state)
        return True


def check_and_reload(running_tasks_count: int, active_tasks_count: int) -> bool:
    """Convenience function: read marker, check safety, execute if safe.

    Returns True if reload was triggered.
    """
    workspace = Path("/root/.nanobot/workspace")
    checker = ReloadChecker.__new__(ReloadChecker)
    checker._workspace = workspace  # noqa: SLF001
    checker._agent = None  # type: ignore[assignment]
    checker._subagents = None
    checker.pending_reload = workspace / ".pending-reload"
    checker.reloading = workspace / ".reloading"

    state = checker.read_pending()
    if state is None:
        return False

    is_stale, warning = checker.check_age(state)
    if is_stale:
        if warning:
            logger.warning(warning)
        return False

    if not checker.is_reload_safe(running_tasks_count, active_tasks_count):
        return False

    if not checker.prepare_reload(state):
        return False

    checker.execute_reload(state)
    return True
