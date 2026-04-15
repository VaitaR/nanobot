"""Dynamic context slot resolver for AGENTS.md and SKILL.md content.

Expands !`command` placeholders by executing whitelisted shell commands
and replacing the placeholder with their stdout output.

Safety constraints:
- Only whitelisted commands are executed (exact match or prefix patterns)
- No network requests, no mutation commands
- 5-second timeout per command
- 500-char truncation per slot output
- Results are cached with a configurable TTL
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field

from loguru import logger

# ---------------------------------------------------------------------------
# Whitelist: exact commands
# ---------------------------------------------------------------------------
_WHITELIST_EXACT: set[str] = {
    "git branch --show-current",
    "git diff --stat",
    "git status --short",
    "python3 tasks.py list --status open",
    "python3 tasks.py list --status open --limit 5",
    "uv run pytest tests/ -q --tb=no",
    "date +%Y-%m-%d",
    "whoami",
    "hostname",
}

# ---------------------------------------------------------------------------
# Whitelist: prefix patterns  (regex applied to the full command string)
# ---------------------------------------------------------------------------
_WHITELIST_PREFIX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^git log --oneline -\d+$"),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_OUTPUT_CHARS = 500
_TIMEOUT_SECONDS = 5
_SLOT_PATTERN = re.compile(r"!`(.*?)`", re.DOTALL)


@dataclass
class _CacheEntry:
    value: str
    expires_at: float


@dataclass
class SlotCache:
    """Simple TTL cache for resolved slot values."""

    ttl: float = 300.0  # default 5 minutes
    _store: dict[str, _CacheEntry] = field(default_factory=dict)

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() >= entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    def put(self, key: str, value: str) -> None:
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + self.ttl,
        )


def _is_whitelisted(command: str) -> bool:
    """Check whether *command* is on the whitelist (exact or prefix pattern)."""
    if command in _WHITELIST_EXACT:
        return True
    return any(p.match(command) for p in _WHITELIST_PREFIX_PATTERNS)


def _execute_command(command: str, timeout: float = _TIMEOUT_SECONDS) -> str:
    """Run a single whitelisted command and return its stdout.

    Returns ``[unavailable: <error>]`` on any failure.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:120]
            msg = f"exit {result.returncode}"
            if stderr:
                msg += f": {stderr}"
            return f"[unavailable: {msg}]"
        output = result.stdout.strip()
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[: _MAX_OUTPUT_CHARS - 3] + "..."
        return output or "[empty]"
    except subprocess.TimeoutExpired:
        return f"[unavailable: timeout after {timeout}s]"
    except Exception as exc:
        return f"[unavailable: {exc!s:.120s}]"


def resolve_dynamic_slots(
    content: str,
    cache: SlotCache | dict | None = None,
    ttl: float = 300.0,
) -> str:
    """Resolve all ``!`command``` dynamic slots in *content*.

    Parameters
    ----------
    content:
        Markdown text potentially containing ``!`command``` placeholders.
    cache:
        A :class:`SlotCache` instance, a plain ``dict``, or ``None``.
        When ``None`` a fresh :class:`SlotCache` is created with the given *ttl*.
        When a plain ``dict`` is supplied it is upgraded to a :class:`SlotCache`.
    ttl:
        Cache time-to-live in seconds (only used when *cache* is ``None``
        or a plain ``dict``).

    Returns
    -------
    str
        The content with every slot replaced by the command output
        (or an ``[unavailable: …]`` fallback).
    """

    # Normalise cache argument
    if cache is None:
        slot_cache = SlotCache(ttl=ttl)
    elif isinstance(cache, dict):
        slot_cache = SlotCache(ttl=ttl)
        # seed with caller-supplied dict (useful for testing)
        now = time.monotonic()
        for k, v in cache.items():
            slot_cache._store[k] = _CacheEntry(value=v, expires_at=now + ttl)
    elif isinstance(cache, SlotCache):
        slot_cache = cache
    else:
        raise TypeError(f"cache must be SlotCache, dict, or None, got {type(cache)!r}")

    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        command = match.group(1).strip()
        if not _is_whitelisted(command):
            logger.debug("Rejected non-whitelisted dynamic slot: {}", command[:120])
            return "[unavailable: command not whitelisted]"

        cached = slot_cache.get(command)
        if cached is not None:
            count += 1
            return cached

        output = _execute_command(command)
        slot_cache.put(command, output)
        count += 1
        return output

    resolved = _SLOT_PATTERN.sub(_replace, content)

    if count:
        logger.info("Resolved {} dynamic context slot(s)", count)

    return resolved
