"""Spawn tool for creating background subagents."""

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.task_lifecycle import extract_task_id, mark_task_delegation_success
from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    _POST_CHECK_INTERVAL_SECONDS = 5
    _POST_CHECK_TIMEOUT_SECONDS = 1800

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._bus: MessageBus | None = None
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._message_thread_id: str | int | None = None
        self._post_check_tasks: set[asyncio.Task[None]] = set()

    def set_context(
        self, channel: str, chat_id: str, message_thread_id: str | int | None = None
    ) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"
        self._message_thread_id = message_thread_id

    def set_bus(self, bus: "MessageBus") -> None:
        """Set bus for progress notifications."""
        self._bus = bus

    def _schedule_post_check(self, *, task_id: str, timeout_s: int) -> None:
        """Start a background loop that moves delegated workspace tasks into review."""
        post_check = asyncio.create_task(
            self._run_post_check_loop(task_id=task_id, timeout_s=timeout_s)
        )
        self._post_check_tasks.add(post_check)
        post_check.add_done_callback(self._post_check_tasks.discard)

    # Short ID pattern: exactly 8 digits + T + exactly 6 digits (no suffix)
    _TASK_ID_SHORT_RE = re.compile(r"\b(\d{8}T\d{6})\b")

    def _extract_workspace_task_id(self, *, task: str, label: str | None) -> str | None:
        """Return the workspace task ID referenced by the spawn request, if any."""
        for candidate in (label or "", task):
            task_id = extract_task_id(candidate)
            if task_id:
                return task_id
        return None

    @staticmethod
    def _load_task_file_body(task_id: str) -> str | None:
        """Load the markdown body of a workspace task file by its short ID.

        Returns the file content if ``~/.nanobot/workspace/tasks/{task_id}.md``
        exists, otherwise *None*.
        """
        path = Path.home() / ".nanobot" / "workspace" / "tasks" / f"{task_id}.md"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            _log.warning("spawn: task file not found for ID %s at %s", task_id, path)
            return None
        except OSError:
            _log.warning("spawn: could not read task file for ID %s at %s", task_id, path)
            return None

    async def _load_workspace_task(self, task_id: str) -> Any | None:
        """Load a task from the workspace store, if the workspace package is available."""
        try:
            from nanobot_workspace.tasks.models import TaskNotFoundError
            from nanobot_workspace.tasks.store import TaskStore
        except Exception:
            return None

        workspace = getattr(self._manager, "workspace", None)
        store = TaskStore(workspace=workspace) if workspace is not None else TaskStore()
        try:
            return await asyncio.to_thread(store.load, task_id)
        except TaskNotFoundError:
            return None
        except Exception:
            return None

    async def _record_boredom_task_activity(self, task_id: str) -> None:
        """Update boredom dedup state after an auto-closure, when available."""
        try:
            from nanobot_workspace.proactive.boredom.orchestrator import BoredomOrchestrator
            from nanobot_workspace.proactive.boredom.store import BoredomStore
        except Exception:
            return

        workspace = getattr(self._manager, "workspace", None)
        if workspace is None:
            return

        def _apply() -> None:
            orchestrator = BoredomOrchestrator(
                store=BoredomStore(workspace / "data" / "boredom_state.json")
            )
            record = getattr(orchestrator, "record_task_activity", None)
            if callable(record):
                record(task_id)

        try:
            await asyncio.to_thread(_apply)
        except Exception:
            return

    async def _run_post_check_loop(self, *, task_id: str, timeout_s: int) -> None:
        """Poll task state until a validation note appears, then move to review."""
        deadline = asyncio.get_running_loop().time() + max(
            timeout_s, self._POST_CHECK_INTERVAL_SECONDS
        )
        while asyncio.get_running_loop().time() < deadline:
            task = await self._load_workspace_task(task_id)
            if task is not None:
                status = getattr(task, "status", "")
                if status in ("done", "review"):
                    return
                validation_note = (getattr(task, "validation_note", "") or "").strip()
                if validation_note and status != "done":
                    if getattr(task, "source", "") == "user":
                        import logging

                        logging.getLogger(__name__).warning(
                            "post-check: skipping auto-closure of user task %s", task_id
                        )
                        return
                    await mark_task_delegation_success(task_id)
                    return
            await asyncio.sleep(self._POST_CHECK_INTERVAL_SECONDS)

    _SCOPE_PATTERNS = re.compile(
        r"##\s*scope|scope\s*:|what\s+\w+\s+must\s+(?:not|cover)",
        re.IGNORECASE,
    )
    _ACCEPTANCE_PATTERNS = re.compile(
        r"##\s*acceptance|acceptance\s+criteria",
        re.IGNORECASE,
    )
    _MIN_TASK_LENGTH = 200

    def _validate_task_packet(self, task: str) -> list[str]:
        """Return list of missing required sections; empty means the packet is valid."""
        if len(task) < self._MIN_TASK_LENGTH:
            _log.warning("spawn: thin task packet (%d chars) — may cause subagent drift", len(task))
        missing: list[str] = []
        if not self._SCOPE_PATTERNS.search(task):
            missing.append("Scope")
        if not self._ACCEPTANCE_PATTERNS.search(task):
            missing.append("Acceptance Criteria")
        return missing

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful.\n"
            "Use the 'executor' parameter to choose the execution backend:\n"
            "- 'codex-5.4' or 'codex-5.3': OpenAI Codex via ACPX (code tasks)\n"
            "- 'claude-native' or 'claude-zai': Claude via ACPX (code tasks)\n"
            "- 'glm-turbo': Zhipu GLM-5-turbo API (research, analysis)\n"
            "- 'glm-5.1': Zhipu GLM-5.1 API (more capable, slower)\n"
            "- 'openrouter': fallback via OpenRouter\n"
            "Default: the main agent's own provider/model."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
                "executor": {
                    "type": "string",
                    "description": (
                        "Executor backend alias. Options: codex-5.4, codex-5.3, codex-5.4-mini, "
                        "claude-native, claude-zai, glm-turbo, glm-5.1, openrouter. "
                        "Omit to use the main agent's default provider/model."
                    ),
                    "enum": [
                        "codex-5.4",
                        "codex-5.3",
                        "codex-5.4-mini",
                        "claude-native",
                        "claude-zai",
                        "glm-turbo",
                        "glm-5.1",
                        "openrouter",
                    ],
                },
                "max_iterations": {
                    "type": "integer",
                    "description": (
                        "Max tool iterations for this subagent. "
                        "Default: read from config. "
                        "Use 15 for quick tasks, 30 for medium, 50+ for long research/implementation. "
                        "Prefer tighter budgets — split large tasks into phases if possible."
                    ),
                },
                "hard_cap": {
                    "type": "integer",
                    "description": (
                        "Absolute timeout in seconds. Default: 1800 (30 min). "
                        "Max allowed: 3600 (1 hour)."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        executor: str | None = None,
        max_iterations: int | None = None,
        hard_cap: int | None = None,
        skip_validation: bool = False,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        # If task string contains a short task ID, try loading the .md file body
        resolved_task = task
        short_match = self._TASK_ID_SHORT_RE.search(task)
        if short_match:
            file_body = self._load_task_file_body(short_match.group(1))
            if file_body is not None:
                resolved_task = file_body + "\n\n" + task
        if not skip_validation:
            missing = self._validate_task_packet(resolved_task)
            if missing:
                return (
                    f"⚠️ Task packet rejected — missing sections: {', '.join(missing)}. "
                    "Add Scope and Acceptance Criteria. "
                    "Read skills/coding-agent-task-design/SKILL.md for format."
                )
        # Build kwargs — only pass non-None values so manager defaults apply
        spawn_kwargs: dict[str, Any] = dict(
            task=resolved_task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )
        if self._message_thread_id is not None:
            spawn_kwargs["message_thread_id"] = self._message_thread_id
        if executor is not None:
            spawn_kwargs["executor"] = executor
        if max_iterations is not None:
            spawn_kwargs["max_iterations"] = max_iterations
        # Cap hard_cap at 1 hour for safety; omit to use manager default (1800s)
        if hard_cap is not None:
            spawn_kwargs["hard_cap"] = min(hard_cap, 3600)
        try:
            result = await self._manager.spawn(**spawn_kwargs)
        except Exception as exc:
            from nanobot.agent.subagent import PeakHoursSpawnBlockedError

            if isinstance(exc, PeakHoursSpawnBlockedError):
                return str(exc)
            raise
        workspace_task_id = self._extract_workspace_task_id(task=task, label=label)
        if workspace_task_id is not None:
            self._schedule_post_check(
                task_id=workspace_task_id,
                timeout_s=min(hard_cap, self._POST_CHECK_TIMEOUT_SECONDS)
                if hard_cap is not None
                else self._POST_CHECK_TIMEOUT_SECONDS,
            )
        if self._bus is not None:
            display_label = label or task[:30] + ("..." if len(task) > 30 else "")
            exec_label = executor or "auto"
            metadata: dict[str, Any] = {"_progress": True, "_subagent_spawned": True}
            if self._message_thread_id is not None:
                metadata["message_thread_id"] = self._message_thread_id
            await self._bus.publish_outbound(
                OutboundMessage(
                    channel=self._origin_channel,
                    chat_id=self._origin_chat_id,
                    content=f"⏳ Запускаю субагента: {display_label} [{exec_label}]...",
                    metadata=metadata,
                )
            )
        return result
