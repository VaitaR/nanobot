"""Notify tool — lets the LLM enqueue proactive Telegram notifications."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class NotifyTool(Tool):
    """Bridge to the workspace proactive notification queue.

    The tool writes to ``~/.nanobot/proactive/queue.json``.  The runtime
    HeartbeatService drains that file on every tick and delivers pending
    notifications via Telegram.

    This is a thin wrapper — all delivery logic lives in the workspace
    ``nanobot_workspace.proactive.notify`` module and the runtime drain.
    """

    @property
    def name(self) -> str:
        return "notify"

    @property
    def description(self) -> str:
        return (
            "Send a proactive notification to the user via Telegram. "
            "Use after completing a significant task, or to flag a blocking issue. "
            "The notification is queued and delivered on the next heartbeat tick."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Notification body text (markdown-supported).",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Priority level (default: normal).",
                },
                "task_id": {
                    "type": "string",
                    "description": "Optional related task ID for traceability.",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        priority: str = "normal",
        task_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            from nanobot_workspace.proactive.notify import notify
        except ImportError as exc:
            return f"Error: workspace notify module unavailable ({exc})"

        try:
            notification_id = notify(
                content,
                priority=priority,
                source="agent",
                task_id=task_id,
            )
            return f"Notification queued (id: {notification_id}, priority: {priority})"
        except Exception as exc:
            return f"Error: failed to enqueue notification ({exc})"
