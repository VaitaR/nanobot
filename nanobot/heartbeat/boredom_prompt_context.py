"""Context helpers for boredom delegation prompts."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

TITLE_KEYS = {"title", "candidate_title"}
TITLE_MAX_CHARS = 2000
TASK_LINE_RE = re.compile(r"^(?P<id>\S+)\s+(?P<status>\S+)\s+(?P<title>.+)$")
UPDATED_RE = re.compile(r"^updated:\s*(?P<value>.+?)\s*$", re.MULTILINE)


def build_boredom_prompt_context(workspace: Path, *, now: datetime | None = None) -> str:
    """Return recent boredom prompt context sections."""
    # Check if boredom is disabled — return empty to effectively no-op the entire pipeline
    state_path = workspace / "data" / "boredom_state.json"
    try:
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            # State is nested under "state" key
            state_data = data.get("state", {})
            if isinstance(state_data, dict) and state_data.get("disabled", False):
                return ""
    except Exception:
        pass

    now = now or datetime.now(UTC)
    sections: list[str] = []
    health = _health_summary(workspace)
    if health:
        sections.append("\n\nCurrent health snapshot:\n" + health)

    previous = _recent_delegation_titles(workspace, now=now)
    if previous:
        sections.append(
            "\n\nPreviously proposed topics (DO NOT suggest these again):\n"
            + "\n".join(f"- {title}" for title in previous)
        )

    completed = _recent_done_task_titles(workspace, now=now)
    if completed:
        sections.append(
            "\n\nRecently completed tasks (SKIP these topics):\n"
            + "\n".join(f"- {title}" for title in completed)
        )

    return "".join(sections)


def _health_summary(workspace: Path) -> str:
    path = workspace / "data" / "health" / "current_state.json"
    if not path.exists():
        return ""
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    overall = state.get("overall", {})
    l1 = state.get("levels", {}).get("l1", {})
    lines = [f"- overall: {overall.get('status', 'unknown')} — {overall.get('summary', 'no summary')}"]
    lines.append(f"- queue: {l1.get('task_queue_depth', '?')}")
    lines.append(f"- last error: {(l1.get('last_error') or {}).get('category', 'none')}")
    return "\n".join(lines)


def _recent_delegation_titles(workspace: Path, *, now: datetime, days: int = 7) -> list[str]:
    cutoff = now - timedelta(days=days)
    root = workspace / "data" / "boredom_delegations"
    if not root.exists():
        return []

    titles: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if datetime.fromtimestamp(path.stat().st_mtime, tz=UTC) < cutoff:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for title in _extract_titles(payload):
            normalized = title.casefold()
            if normalized in seen:
                continue
            extra = len(title) + 3
            if total_chars + extra > TITLE_MAX_CHARS:
                return titles
            seen.add(normalized)
            titles.append(title)
            total_chars += extra
    return titles


def _extract_titles(value: Any) -> list[str]:
    titles: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in TITLE_KEYS or key.lower().endswith("_title"):
                if isinstance(nested, str) and nested.strip():
                    titles.append(nested.strip())
            titles.extend(_extract_titles(nested))
    elif isinstance(value, list):
        for item in value:
            titles.extend(_extract_titles(item))
    return titles


def _recent_done_task_titles(workspace: Path, *, now: datetime, days: int = 7) -> list[str]:
    cutoff = now - timedelta(days=days)
    try:
        completed = subprocess.run(
            ["python3", "tasks.py", "list", "--status", "done"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    titles: list[str] = []
    seen: set[str] = set()
    for line in completed.stdout.splitlines():
        match = TASK_LINE_RE.match(line.strip())
        if not match or match.group("status") != "done":
            continue
        task_id = match.group("id")
        title = match.group("title").strip()
        updated = _task_updated_at(workspace, task_id)
        if updated is None or updated < cutoff:
            continue
        normalized = title.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        titles.append(title)
    return titles


def _task_updated_at(workspace: Path, task_id: str) -> datetime | None:
    for path in (workspace / "tasks" / "archive" / f"{task_id}.md", workspace / "tasks" / f"{task_id}.md"):
        if not path.exists():
            continue
        try:
            match = UPDATED_RE.search(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if match:
            return _parse_datetime(match.group("value"))
    return None


def _parse_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None
