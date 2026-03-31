"""Spawn tool for creating background subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

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
            "and use a dedicated subdirectory when helpful."
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
                "max_iterations": {
                    "type": "integer",
                    "description": (
                        "Max tool iterations for this subagent. "
                        "Default: read from config (40). "
                        "Use 15 for quick tasks, 50+ for long research/implementation."
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

    async def execute(self, task: str, label: str | None = None,
                      max_iterations: int | None = None,
                      hard_cap: int | None = None,
                      **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        # Build kwargs — only pass non-None values so manager defaults apply
        spawn_kwargs: dict[str, Any] = dict(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )
        if max_iterations is not None:
            spawn_kwargs["max_iterations"] = max_iterations
        # Cap hard_cap at 1 hour for safety; omit to use manager default (1800s)
        if hard_cap is not None:
            spawn_kwargs["hard_cap"] = min(hard_cap, 3600)
        return await self._manager.spawn(**spawn_kwargs)
