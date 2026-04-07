"""Drain pending notifications from the proactive queue.

Reads ``~/.nanobot/proactive/queue.json``, extracts pending items,
and returns them for delivery.  Callers are responsible for actually
sending notifications (via ``on_notify``) and then calling
``mark_delivered`` to persist the status update atomically.

Also defines the heartbeat tool schema and system prompt used by
``HeartbeatService._decide()``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

# -- Queue paths -----------------------------------------------------------

QUEUE_PATH: Path = Path.home() / ".nanobot" / "proactive" / "queue.json"

# -- Heartbeat tool definition ----------------------------------------------

HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "description": "Report heartbeat decision after reviewing tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["skip", "run", "review"],
                        "description": "skip=nothing, run=active tasks, review=subagent done",
                    },
                    "tasks": {
                        "type": "string",
                        "description": "Natural-language summary of active tasks",
                    },
                    "review_decision": {
                        "type": "array",
                        "description": "For review: [{task_id, verdict(done|failed), note}]",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "verdict": {"type": "string", "enum": ["done", "failed"]},
                                "note": {"type": "string"},
                            },
                            "required": ["task_id", "verdict", "note"],
                        },
                    },
                },
                "required": ["action"],
            },
        },
    }
]

HEARTBEAT_SYSTEM_PROMPT = (
    "You are a heartbeat agent. Call the heartbeat tool to report your decision.\n"
    "Actions: skip (nothing), run (active tasks), review (subagent completed).\n"
    "For review, include review_decision with task_id, verdict (done/failed), note. "
    "Only mark done if subagent likely succeeded. If uncertain, mark failed."
)


def _read_queue(path: Path) -> dict[str, Any]:
    """Read and parse the queue file, returning empty schema on any error."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("drain: failed to read queue at {}", path)
    return {"version": 1, "notifications": []}


def _write_queue(path: Path, data: dict[str, Any]) -> None:
    """Atomic write via temp-file rename."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.exception("drain: failed to write queue at {}", path)


def collect_pending(*, path: Path | None = None) -> list[dict[str, Any]]:
    """Return pending notifications ready for delivery.

    Filters for status == "pending" and channel == "telegram".
    Returns an empty list if the queue is missing or malformed.
    """
    queue_path = path or QUEUE_PATH
    data = _read_queue(queue_path)
    notifications = data.get("notifications", [])
    pending = [
        n for n in notifications
        if n.get("status") == "pending" and n.get("channel") == "telegram"
    ]
    if pending:
        logger.info("drain: {} pending notifications", len(pending))
    return pending


def mark_delivered(
    notification_ids: list[str], *, path: Path | None = None
) -> int:
    """Mark delivered notifications as ``sent`` with a timestamp.

    Returns the number of notifications actually updated.
    """
    if not notification_ids:
        return 0
    queue_path = path or QUEUE_PATH
    data = _read_queue(queue_path)
    id_set = set(notification_ids)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    updated = 0
    for n in data.get("notifications", []):
        if n.get("id") in id_set and n.get("status") == "pending":
            n["status"] = "sent"
            n["sent_at"] = now
            updated += 1
    if updated:
        _write_queue(queue_path, data)
        logger.info("drain: marked {} notifications as sent", updated)
    return updated
