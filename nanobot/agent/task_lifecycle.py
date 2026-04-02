"""Subagent task lifecycle tracking — bridges nanobot core → nanobot-tasks CLI.

Extracts nanobot-tasks IDs from spawn labels and updates task status via
subprocess calls to ``uv run nanobot-tasks``. This avoids import-layer
violations between nanobot (core) and nanobot_workspace (workspace package).
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Task ID extraction ──────────────────────────────────────────────────────

# nanobot-tasks IDs look like: 20260329T140912_phase_2_add_fts5_memory_search
_TASK_ID_RE = re.compile(
    r"\b(\d{8}T\d{6}_[a-z0-9_]{3,})\b"
)

# Parenthetical hints: "(hb: 152315)" or "(task: 20260329T140912_phase_2_add_fts5_memory_search)"
_PAREN_HINT_RE = re.compile(
    r"\(\s*(?:hb|task)\s*:\s*(\S+?)\s*\)"
)


def extract_task_id(label: str) -> str | None:
    """Extract a nanobot-tasks ID from a spawn label.

    Recognised patterns (tried in order, first match wins):
      1. ``(hb: …)`` or ``(task: …)`` parenthetical hints
      2. Bare task IDs matching ``YYYYMMDDTHHMMSS_slug``
    """
    if not label:
        return None

    # 1. Parenthetical hint — value may be a full ID or a short ref
    m = _PAREN_HINT_RE.search(label)
    if m:
        candidate = m.group(1)
        # If it looks like a full task ID, return it directly
        if _TASK_ID_RE.fullmatch(candidate):
            return candidate
        # Otherwise it's likely a short ref (e.g. "152315"); skip for now
        # since we can't resolve it to a full ID without the store.
        logger.debug("Parenthetical hint '%s' is not a full task ID, skipping", candidate)

    # 2. Bare task ID anywhere in the label
    m = _TASK_ID_RE.search(label)
    if m:
        return m.group(1)

    return None


# ── CLI bridge ───────────────────────────────────────────────────────────────

_WORKSPACE_DIR = Path.home() / ".nanobot" / "workspace"
_NANOBOT_TASKS_CMD = "uv run nanobot-tasks"


def _build_cmd(args: str) -> str:
    """Return a shell command string for nanobot-tasks CLI invocation."""
    return f"cd {_WORKSPACE_DIR} && {_NANOBOT_TASKS_CMD} {args}"


async def _run_cli(args: str) -> bool:
    """Run a nanobot-tasks CLI command via subprocess.

    Returns True if the command succeeded (exit code 0), False otherwise.
    """
    cmd = _build_cmd(args)
    logger.debug("Task lifecycle CLI: %s", cmd)
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            logger.warning("Task lifecycle CLI failed (exit %d): %s", proc.returncode, err or cmd)
            return False
        return True
    except asyncio.TimeoutError:
        logger.warning("Task lifecycle CLI timed out: %s", cmd)
        return False
    except Exception as exc:
        logger.warning("Task lifecycle CLI error: %s — %s", cmd, exc)
        return False


async def mark_task_delegated(task_id: str) -> bool:
    """Mark a task as in_progress and record a delegation event.

    Returns True if at least one CLI call succeeded.
    """
    # Update status to in_progress
    status_ok = await _run_cli(f"update {shlex.quote(task_id)} --status in_progress --reason 'delegated to subagent'")
    # Add a typed event
    event_ok = await _run_cli(f"event {shlex.quote(task_id)} 'delegated to subagent' --kind delegated")
    return status_ok or event_ok


async def mark_task_delegation_success(task_id: str) -> bool:
    """Record successful delegation completion on a task.

    Does NOT close the task — the coordinator decides that.
    """
    return await _run_cli(
        f"event {shlex.quote(task_id)} 'delegation completed successfully' --kind progress"
    )


async def mark_task_delegation_failure(task_id: str, reason: str) -> bool:
    """Record delegation failure on a task.

    Does NOT close the task — the coordinator decides that.
    """
    safe_reason = reason[:200]  # truncate to avoid shell arg limits
    return await _run_cli(
        f"event {shlex.quote(task_id)} {shlex.quote(f'delegation failed: {safe_reason}')} --kind progress"
    )


async def query_pending_review() -> list[dict[str, str]]:
    """Find tasks delegated to subagents that have completed but not yet closed.

    Returns a list of dicts with keys: id, title, status.
    A task is considered "pending review" when:
      - It has a ``[delegated]`` event in its history
      - It has a ``[progress] delegation completed successfully`` event
      - Its status is not ``done``
    """
    cmd = _build_cmd("list --status in_progress --json")
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            logger.warning("query_pending_review: list failed: %s", stderr.decode(errors="replace"))
            return []
        import json
        tasks = json.loads(stdout.decode("utf-8"))
    except Exception as exc:
        logger.warning("query_pending_review error: %s", exc)
        return []

    # Also check open tasks that might have delegation events but weren't
    # moved to in_progress (e.g. subagent didn't call mark_task_delegated).
    cmd_open = _build_cmd("list --status open --json")
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd_open,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0:
            import json
            open_tasks = json.loads(stdout.decode("utf-8"))
            known_ids = {t["id"] for t in tasks}
            for t in open_tasks:
                if t["id"] not in known_ids:
                    tasks.append(t)
    except Exception:
        pass

    # Filter: task history must contain delegation success marker
    pending: list[dict[str, str]] = []
    for task in tasks:
        tid = task["id"]
        show_cmd = _build_cmd(f"show {shlex.quote(tid)}")
        try:
            proc = await asyncio.create_subprocess_shell(
                show_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            content = stdout.decode("utf-8", errors="replace")
            has_delegated = "[delegated]" in content
            has_success = "delegation completed successfully" in content
            if has_delegated and has_success:
                pending.append({"id": tid, "title": task.get("title", ""), "status": task.get("status", "")})
        except Exception:
            pass
    return pending


async def close_task(task_id: str, validation_note: str) -> bool:
    """Close a task via nanobot-tasks done.

    Returns True if the command succeeded.
    """
    safe_note = validation_note[:300]
    return await _run_cli(
        f"done {shlex.quote(task_id)} --validation-note {shlex.quote(safe_note)}"
    )
