"""Unified error classification for all LLM providers."""

from __future__ import annotations

import random
from enum import Enum


class ErrorType(Enum):
    TRANSIENT = "transient"
    FATAL = "fatal"
    UNKNOWN = "unknown"


_TRANSIENT_MARKERS: tuple[str, ...] = (
    "429",
    "rate limit",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "server error",
    "temporarily unavailable",
)

_QUOTA_MARKERS: tuple[str, ...] = (
    "quota",
    "usage limit",
    "1308",
    "limit reached",
    "insufficient_quota",
    "exceeded your",
)

_FATAL_MARKERS: tuple[str, ...] = (
    "401",
    "403",
    "authentication",
    "invalid api",
    "invalid api_key",
    "account deactivated",
)


def classify_provider_error(
    error: Exception | None,
    content: str | None = None,
) -> ErrorType:
    """Classify an error as TRANSIENT, FATAL, or UNKNOWN."""
    # Check status_code attribute first (httpx, requests, etc.)
    if error is not None:
        status = getattr(error, "status_code", None)
        # Some libraries (httpx) expose status_code on the nested response
        if status is None:
            resp = getattr(error, "response", None)
            if resp is not None:
                status = getattr(resp, "status_code", None)
        if status is not None:
            status_str = str(status)
            if status_str in _FATAL_MARKERS:
                return ErrorType.FATAL
            if status_str in _TRANSIENT_MARKERS:
                return ErrorType.TRANSIENT

    # Fall back to content-based classification
    text = (content or (str(error) if error else "")).lower()
    if not text:
        return ErrorType.UNKNOWN

    # Check fatal first (takes priority)
    if any(marker in text for marker in _FATAL_MARKERS):
        return ErrorType.FATAL
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return ErrorType.TRANSIENT
    if any(marker in text for marker in _QUOTA_MARKERS):
        return ErrorType.TRANSIENT
    return ErrorType.UNKNOWN


def is_retryable(error: Exception | None, content: str | None = None) -> bool:
    """Only TRANSIENT errors are retryable. UNKNOWN and FATAL are not.

    This preserves the original ``_is_transient_error`` behaviour where
    unrecognised error content was *not* retried (fail-closed), ensuring
    the image-fallback path in ``chat_with_retry`` still triggers correctly.
    """
    return classify_provider_error(error, content) == ErrorType.TRANSIENT


def get_backoff_seconds(attempt: int, max_delay: float = 8.0) -> float:
    """Exponential backoff with jitter: 2^(attempt+1) * [0.5..1.0), capped at max_delay."""
    base = min(2 ** (attempt + 1), max_delay)
    return base * (0.5 + random.random() * 0.5)


def should_circuit_break(error: Exception | None, content: str | None = None) -> bool:
    """Future circuit breaker — currently always returns False."""
    return False
