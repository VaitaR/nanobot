"""Memory search tool — exposes workspace FTS5 + embedding search to the agent.

Wraps the workspace ``nanobot_workspace.memory.search`` module so the
agent can query long-term memory via tool call instead of grep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class MemorySearchTool(Tool):
    """Search long-term memory using FTS5 + optional vector embeddings."""

    def __init__(self, workspace: Path):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search long-term memory (MEMORY.md, HISTORY.md, improvement-log) "
            "using full-text search with optional embedding similarity. "
            "Returns the most relevant chunks ranked by relevance score."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — natural language or keywords",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 3, max 10)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str | None = None, limit: int = 3, **kwargs: Any) -> str:
        if not query:
            return "Error: query is required"

        limit = min(max(1, limit), 10)

        try:
            from nanobot_workspace.memory.search import (
                format_relevant_context,
                search_relevant_chunks,
            )
        except ImportError:
            return "Error: workspace memory search module not available"

        try:
            results = search_relevant_chunks(query, self._workspace, limit=limit)
        except Exception as exc:
            return f"Error searching memory: {exc}"

        if not results:
            return f"No results found for: {query}"

        formatted = format_relevant_context(results)
        return f"Found {len(results)} result(s) for '{query}':\n\n{formatted}"
