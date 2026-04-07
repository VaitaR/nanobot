"""PermissionEngine: 3-level tool permission gating.

Levels: ALLOW, ASK, DENY.  Tools default to ALLOW when the engine is not
initialised or when the tool name is not present in the policy.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


class Permission(enum.Enum):
    """Permission level for a tool call."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(slots=True)
class PermissionResult:
    """Outcome of a permission check."""

    permission: Permission
    reason: str = ""


# Built-in tool classifications — dangerous tools default to ASK.
_DEFAULT_POLICY: dict[str, Permission] = {
    "exec": Permission.ASK,
    "write_file": Permission.ASK,
    "edit_file": Permission.ASK,
    "restart_gateway": Permission.DENY,
}


@dataclass
class PermissionEngine:
    """Evaluate tool-call permissions.

    The engine merges an optional *policy* (``{tool_name: Permission}``)
    on top of the built-in defaults.  Any tool not listed in either source
    is treated as ``ALLOW``.
    """

    policy: dict[str, Permission] = field(default_factory=dict)
    _merged: dict[str, Permission] | None = field(default=None, init=False, repr=False)

    # -- public API --------------------------------------------------------

    def check(self, tool_name: str, arguments: dict[str, Any] | None = None) -> PermissionResult:
        """Return the permission decision for *tool_name*."""
        merged = self._get_merged()
        level = merged.get(tool_name, Permission.ALLOW)
        if level == Permission.DENY:
            return PermissionResult(permission=Permission.DENY, reason=f"Tool '{tool_name}' is denied by policy.")
        if level == Permission.ASK:
            return PermissionResult(permission=Permission.ASK, reason=f"Tool '{tool_name}' requires confirmation.")
        return PermissionResult(permission=Permission.ALLOW)

    def filter_calls(
        self,
        tool_calls: list[tuple[str, dict[str, Any]]],
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[PermissionResult]]:
        """Split *tool_calls* into allowed and blocked.

        Returns ``(allowed, blocked)``.  Both *ASK* and *DENY* calls are
        placed in *blocked*.
        """
        allowed: list[tuple[str, dict[str, Any]]] = []
        blocked: list[PermissionResult] = []
        for name, args in tool_calls:
            result = self.check(name, args)
            if result.permission == Permission.ALLOW:
                allowed.append((name, args))
            else:
                blocked.append(result)
        return allowed, blocked

    # -- internals ---------------------------------------------------------

    def _get_merged(self) -> dict[str, Permission]:
        if self._merged is None:
            self._merged = {**_DEFAULT_POLICY, **self.policy}
        return self._merged


def create_permission_engine(policy: dict[str, Permission] | None = None) -> PermissionEngine | None:
    """Factory with graceful degradation.

    Returns ``None`` (instead of raising) if initialisation fails, so
    callers can fall back to unconditionally allowing all tool calls.
    """
    if policy is None:
        policy = {}
    try:
        engine = PermissionEngine(policy=policy)
        logger.debug("PermissionEngine initialised with {} custom rules", len(policy))
        return engine
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("PermissionEngine failed to initialise, defaulting to ALLOW-all: {}", exc)
        return None
