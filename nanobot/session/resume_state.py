"""Helpers for restart resume session persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAST_ACTIVE_SESSION_PATH = Path("data/last_active_session.json")


def _clean_str(value: Any) -> str | None:
    """Return a stripped string value or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_session_payload(data: dict[str, Any] | None) -> dict[str, str | None] | None:
    """Validate and normalize persisted session metadata."""
    if not isinstance(data, dict):
        return None

    channel = _clean_str(data.get("channel"))
    chat_id = _clean_str(data.get("chat_id"))
    if not channel or not chat_id:
        return None

    return {
        "channel": channel,
        "chat_id": chat_id,
        "message_thread_id": _clean_str(data.get("message_thread_id")),
        "resume_prompt": _clean_str(data.get("resume_prompt")),
        "timestamp": _clean_str(data.get("timestamp")),
    }


def load_last_active_session(workspace: Path) -> dict[str, str | None] | None:
    """Load the most recently active routable session from disk."""
    path = workspace / LAST_ACTIVE_SESSION_PATH
    if not path.exists():
        return None

    try:
        return normalize_session_payload(json.loads(path.read_text()))
    except Exception:
        return None


def persist_last_active_session(
    workspace: Path,
    *,
    channel: str,
    chat_id: str,
    message_thread_id: str | int | None = None,
) -> None:
    """Persist the latest active chat/topic for restart fallback."""
    payload = normalize_session_payload(
        {
            "channel": channel,
            "chat_id": chat_id,
            "message_thread_id": message_thread_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    if payload is None:
        return

    path = workspace / LAST_ACTIVE_SESSION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def merge_resume_session(
    primary: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
    *,
    default_resume_prompt: str | None = None,
) -> dict[str, str | None] | None:
    """Merge restart metadata with a fallback session record."""
    primary_payload = primary if isinstance(primary, dict) else {}
    primary_session = normalize_session_payload(primary_payload)
    fallback_session = normalize_session_payload(fallback)

    channel = (primary_session or {}).get("channel") or (fallback_session or {}).get("channel")
    chat_id = (primary_session or {}).get("chat_id") or (fallback_session or {}).get("chat_id")
    if not channel or not chat_id:
        return None

    message_thread_id = (
        _clean_str(primary_payload.get("message_thread_id"))
        or (fallback_session or {}).get("message_thread_id")
    )
    resume_prompt = (
        _clean_str(primary_payload.get("resume_prompt"))
        or (fallback_session or {}).get("resume_prompt")
        or _clean_str(default_resume_prompt)
    )

    return {
        "channel": channel,
        "chat_id": chat_id,
        "message_thread_id": message_thread_id,
        "resume_prompt": resume_prompt,
    }
