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
        self._message_thread_id: str | int | None = None
        self._request_id: str | None = None

    def set_context(
        self,
        channel: str,
        chat_id: str,
        message_thread_id: str | int | None = None,
        request_id: str | None = None,
    ) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"
        self._message_thread_id = message_thread_id
        self._request_id = request_id

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
                "enable_checkpoint_policy": {
                    "type": "boolean",
                    "description": (
                        "Enable checkpoint review policy for this subagent run. "
                        "If true, defaults are used unless checkpoint_* overrides are provided."
                    ),
                },
                "checkpoint_threshold_pct": {
                    "type": "number",
                    "description": "Checkpoint trigger threshold (0.0-1.0), default 0.80.",
                },
                "checkpoint_read_only_hint": {
                    "type": "boolean",
                    "description": "Hint that task may be read-only; suppresses file-touch stuck signal.",
                },
                "checkpoint_escalation_cooldown": {
                    "type": "integer",
                    "description": "Cooldown (iterations) between escalations, default 5.",
                },
                "checkpoint_review_timeout": {
                    "type": "integer",
                    "description": "Review timeout in seconds, default 120.",
                },
                "checkpoint_loop_window": {
                    "type": "integer",
                    "description": "Loop detector window, default 8.",
                },
                "checkpoint_max_checkpoints": {
                    "type": "integer",
                    "description": "Maximum checkpoints per run, default 3.",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None,
                      max_iterations: int | None = None,
                      hard_cap: int | None = None,
                      enable_checkpoint_policy: bool | None = None,
                      checkpoint_threshold_pct: float | None = None,
                      checkpoint_read_only_hint: bool | None = None,
                      checkpoint_escalation_cooldown: int | None = None,
                      checkpoint_review_timeout: int | None = None,
                      checkpoint_loop_window: int | None = None,
                      checkpoint_max_checkpoints: int | None = None,
                      **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        checkpoint_policy = None
        # Build kwargs — only pass non-None values so manager defaults apply
        spawn_kwargs: dict[str, Any] = dict(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
            request_id=self._request_id,
        )
        if self._message_thread_id is not None:
            spawn_kwargs["message_thread_id"] = self._message_thread_id
        if max_iterations is not None:
            spawn_kwargs["max_iterations"] = max_iterations
        # Cap hard_cap at 1 hour for safety; omit to use manager default (1800s)
        if hard_cap is not None:
            spawn_kwargs["hard_cap"] = min(hard_cap, 3600)

        checkpoint_requested = bool(enable_checkpoint_policy) or any(
            x is not None
            for x in (
                checkpoint_threshold_pct,
                checkpoint_read_only_hint,
                checkpoint_escalation_cooldown,
                checkpoint_review_timeout,
                checkpoint_loop_window,
                checkpoint_max_checkpoints,
            )
        )
        if checkpoint_requested:
            from nanobot.checkpoint.policy import ReviewPolicy

            checkpoint_policy = ReviewPolicy(
                checkpoint_threshold_pct=(
                    checkpoint_threshold_pct if checkpoint_threshold_pct is not None else 0.80
                ),
                read_only_hint=(
                    bool(checkpoint_read_only_hint)
                    if checkpoint_read_only_hint is not None
                    else False
                ),
                escalation_cooldown=(
                    checkpoint_escalation_cooldown
                    if checkpoint_escalation_cooldown is not None
                    else 5
                ),
                review_timeout=(
                    checkpoint_review_timeout if checkpoint_review_timeout is not None else 120
                ),
                loop_window=(checkpoint_loop_window if checkpoint_loop_window is not None else 8),
                max_checkpoints=(
                    checkpoint_max_checkpoints if checkpoint_max_checkpoints is not None else 3
                ),
            )
            spawn_kwargs["checkpoint_policy"] = checkpoint_policy
        return await self._manager.spawn(**spawn_kwargs)
