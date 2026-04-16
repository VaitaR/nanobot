"""Spawn status tool — shows currently active subagent spawns."""

from typing import Any, Callable

from nanobot.agent.tools.base import Tool


class SpawnStatusTool(Tool):
    """Tool to report all currently active subagent spawns."""

    def __init__(self, get_active_tasks: Callable[[], list[dict[str, Any]]]) -> None:
        self._get_active_tasks = get_active_tasks

    @property
    def name(self) -> str:
        return "spawn_status"

    @property
    def description(self) -> str:
        return (
            "Show all currently active (running) subagent spawns across all sessions. "
            "Use this to verify real spawn state before delegating new tasks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        """Return a formatted summary of active spawns."""
        tasks = self._get_active_tasks()
        if not tasks:
            return "No active spawns."
        lines = [f"Active spawns: {len(tasks)}"]
        for task in tasks:
            session_key = task.get("session_key", "unknown")
            elapsed = task.get("elapsed_seconds", 0.0)
            lines.append(f"  - {session_key}  elapsed={elapsed:.1f}s")
        return "\n".join(lines)
