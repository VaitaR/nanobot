"""User response actions for checkpoint review."""

from __future__ import annotations

from enum import StrEnum


class UserAction(StrEnum):
    """Actions a user (or timeout fallback) can take at a checkpoint."""

    CONTINUE = "continue"
    DONE = "done"
    STOP = "stop"
    DETAILS = "details"
