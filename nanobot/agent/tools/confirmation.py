"""Configurable confirmation gates for destructive tool actions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class ConfirmationRule:
    """A single rule in the confirmation policy.

    Attributes:
        tool: Tool name to match, or ``"*"`` for all tools.
        pattern: Optional regex pattern matched against tool params
            (for ``exec`` this matches the command string).
        action: ``"confirm"``, ``"allow"``, or ``"deny"``.
    """

    tool: str = "*"
    pattern: str = ""
    action: Literal["confirm", "allow", "deny"] = "confirm"

    # --- internal ---
    _compiled: re.Pattern[str] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.pattern:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, tool_name: str, params: dict[str, Any]) -> bool:
        """Return True if this rule applies to the given tool call."""
        if self.tool != "*" and self.tool != tool_name:
            return False
        if self._compiled is None:
            return True  # no pattern → matches by tool name alone
        # Build a single string from params for pattern matching.
        # For exec: the "command" key; for write_file/edit_file: the "path" key.
        searchable = self._params_to_str(params)
        return bool(self._compiled.search(searchable))

    @staticmethod
    def _params_to_str(params: dict[str, Any]) -> str:
        """Flatten params into a single searchable string."""
        parts: list[str] = []
        # Prioritise common keys
        for key in ("command", "path", "content", "old_text", "new_text"):
            val = params.get(key)
            if val is not None:
                parts.append(str(val))
        # Also include any remaining values
        for key, val in params.items():
            if key not in ("command", "path", "content", "old_text", "new_text") and val is not None:
                parts.append(str(val))
        return "\n".join(parts)


@dataclass
class ConfirmationPolicy:
    """Policy engine that decides whether a tool call needs confirmation.

    Rules are evaluated in order; the **first matching rule wins**.
    If no rule matches, the default is ``allow`` (no confirmation needed).
    """

    rules: list[ConfirmationRule] = field(default_factory=list)

    def requires_confirmation(self, tool_name: str, params: dict[str, Any]) -> bool:
        """Return True if the tool call should be blocked pending user approval."""
        action = self._evaluate(tool_name, params)
        return action == "confirm"

    def is_denied(self, tool_name: str, params: dict[str, Any]) -> bool:
        """Return True if the tool call is outright denied by policy."""
        action = self._evaluate(tool_name, params)
        return action == "deny"

    def describe(self, tool_name: str, params: dict[str, Any]) -> str:
        """Return a human-readable description of what is about to happen."""
        if tool_name == "exec":
            cmd = params.get("command", "(unknown)")
            return f"exec: {cmd}"
        if tool_name == "write_file":
            path = params.get("path", "(unknown)")
            return f"write_file to {path}"
        if tool_name == "edit_file":
            path = params.get("path", "(unknown)")
            old = params.get("old_text", "")
            preview = old[:80] + "..." if len(old) > 80 else old
            return f"edit_file in {path}: replace \"{preview}\""
        if tool_name == "spawn":
            prompt = params.get("prompt", "(unknown)")
            preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            return f"spawn subagent: {preview}"
        # Generic fallback
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{tool_name}({param_str})"

    def _evaluate(self, tool_name: str, params: dict[str, Any]) -> str:
        """Evaluate rules and return the winning action.

        Rules with a non-empty pattern (targeted rules) are evaluated
        before no-pattern rules. Within the same tier, declaration
        order is preserved (first match wins).
        """
        def _sort_key(r: ConfirmationRule) -> tuple[int, int]:
            # Tier 0: rules with patterns (targeted) → evaluated first
            # Tier 1: rules without patterns (broad) → evaluated second
            # Within each tier, preserve declaration order
            return (0 if r.pattern else 1, self.rules.index(r))

        for rule in sorted(self.rules, key=_sort_key):
            if rule.matches(tool_name, params):
                return rule.action
        return "allow"

    # --- factory helpers ---

    @classmethod
    def from_config(cls, rules: list[dict[str, str]]) -> ConfirmationPolicy:
        """Build a policy from a list of dicts (as stored in config.json).

        Each dict should have keys: ``tool``, ``pattern``, ``action``.
        """
        parsed: list[ConfirmationRule] = []
        for r in rules:
            parsed.append(ConfirmationRule(
                tool=r.get("tool", "*"),
                pattern=r.get("pattern", ""),
                action=r.get("action", "confirm"),
            ))
        return cls(rules=parsed)

    @classmethod
    def default_policy(cls, workspace: Path | None = None) -> ConfirmationPolicy:
        """Build the recommended default policy.

        This uses sensible defaults that protect against common destructive
        operations while staying out of the way for safe actions.
        """
        rules: list[ConfirmationRule] = [
            # write_file to an existing path is potentially destructive
            ConfirmationRule(tool="write_file", pattern="", action="confirm"),
            # edit_file always changes existing content
            ConfirmationRule(tool="edit_file", pattern="", action="confirm"),
            # Dangerous exec patterns
            ConfirmationRule(tool="exec", pattern=r"\brm\s+-rf\b", action="confirm"),
            ConfirmationRule(tool="exec", pattern=r"\brm\s+-r\b", action="confirm"),
            ConfirmationRule(tool="exec", pattern=r"\bchmod\b.*[0-7]{3,4}", action="confirm"),
            ConfirmationRule(tool="exec", pattern=r"\bshutdown\b", action="confirm"),
            ConfirmationRule(tool="exec", pattern=r"\bkill\b\s+-9\b", action="confirm"),
            ConfirmationRule(tool="exec", pattern=r"\bmkfs\b", action="confirm"),
            ConfirmationRule(tool="exec", pattern=r"\bdd\b\s+if=", action="confirm"),
            ConfirmationRule(tool="exec", pattern=r"\bformat\b", action="confirm"),
            ConfirmationRule(tool="exec", pattern=r"\bmv\b\s+.*\s+/", action="confirm"),
            # spawn always — long-running autonomous execution
            ConfirmationRule(tool="spawn", pattern="", action="confirm"),
        ]
        return cls(rules=rules)


# Marker string prefix used as the tool result when confirmation is required.
# The LLM sees this and generates a user-facing approval request.
CONFIRM_REQUIRED_PREFIX = "CONFIRM_REQUIRED: "
DENIED_PREFIX = "CONFIRM_DENIED: "
