"""Structured subagent result envelope.

Replaces lossy double-summarization with a structured envelope that
preserves artifacts and full details while allowing smart routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Artifact:
    """A concrete output produced by the subagent."""

    path: str  # file path or URL
    description: str  # human-readable label
    kind: str  # "file" | "diff" | "url" | "metric"
    verified: bool = False  # post-execution check confirmed existence


@dataclass(slots=True)
class ResultEnvelope:
    """Structured subagent result.

    The ``summary`` field is a short (1-2 sentence) description meant for
    direct display.  ``details`` preserves the full output so nothing is
    lost.  ``artifacts`` lists concrete deliverables.
    """

    status: str  # "ok" | "error" | "partial"
    summary: str  # 1-2 sentence summary for the user
    artifacts: list[Artifact] = field(default_factory=list)
    details: str = ""  # full text, preserved for detailed view
    stop_reason: str = "completed"
    error: str | None = None


def extract_artifacts(tool_events: list[dict[str, Any]]) -> list[Artifact]:
    """Build an ``Artifact`` list from runner ``tool_events``.

    Scans for *write_file*, *edit_file* calls and records the target
    paths.  Also captures *exec* calls that reference file paths.
    """
    artifacts: list[Artifact] = []
    seen: set[str] = set()

    for event in tool_events:
        if event.get("status") != "ok":
            continue
        name = event.get("name", "")
        args = event.get("arguments", {})
        path = args.get("path") or args.get("file_path") or ""

        if name in ("write_file", "edit_file") and path and path not in seen:
            kind = "diff" if name == "edit_file" else "file"
            artifacts.append(Artifact(path=str(path), description=f"{name}: {path}", kind=kind))
            seen.add(path)
        elif name == "exec" and path and path not in seen:
            artifacts.append(Artifact(path=str(path), description=f"exec: {path}", kind="file"))
            seen.add(path)

    return artifacts
