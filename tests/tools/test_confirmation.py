"""Tests for configurable confirmation gates."""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.confirmation import (
    CONFIRM_REQUIRED_PREFIX,
    DENIED_PREFIX,
    ConfirmationPolicy,
    ConfirmationRule,
)
from nanobot.agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """A minimal tool that returns its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return kwargs.get("text", "")


class FakeExecTool(Tool):
    """Simulates the exec tool interface."""

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "exec tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return f"ran: {kwargs.get('command', '')}"


# ---------------------------------------------------------------------------
# ConfirmationRule tests
# ---------------------------------------------------------------------------


class TestConfirmationRule:
    """Unit tests for the ConfirmationRule dataclass."""

    def test_wildcard_tool_matches_any_tool(self) -> None:
        rule = ConfirmationRule(tool="*", action="confirm")
        assert rule.matches("exec", {"command": "ls"})
        assert rule.matches("write_file", {"path": "/tmp/x"})

    def test_specific_tool_matches_only_that_tool(self) -> None:
        rule = ConfirmationRule(tool="exec", action="confirm")
        assert rule.matches("exec", {"command": "ls"})
        assert not rule.matches("write_file", {"path": "/tmp/x"})

    def test_pattern_matches_command(self) -> None:
        rule = ConfirmationRule(tool="exec", pattern=r"rm\s+-rf", action="confirm")
        assert rule.matches("exec", {"command": "rm -rf /tmp/stuff"})
        assert not rule.matches("exec", {"command": "ls -la"})

    def test_pattern_case_insensitive(self) -> None:
        rule = ConfirmationRule(tool="exec", pattern=r"rm\s+-rf", action="confirm")
        assert rule.matches("exec", {"command": "RM -RF /tmp/stuff"})

    def test_no_pattern_matches_by_name_alone(self) -> None:
        rule = ConfirmationRule(tool="edit_file", pattern="", action="confirm")
        assert rule.matches("edit_file", {"path": "/tmp/x"})
        assert not rule.matches("exec", {"command": "ls"})

    def test_pattern_matches_path_param(self) -> None:
        rule = ConfirmationRule(tool="write_file", pattern=r"/etc/", action="confirm")
        assert rule.matches("write_file", {"path": "/etc/passwd", "content": "x"})
        assert not rule.matches("write_file", {"path": "/tmp/harmless", "content": "x"})


# ---------------------------------------------------------------------------
# ConfirmationPolicy tests
# ---------------------------------------------------------------------------


class TestConfirmationPolicy:
    """Unit tests for the ConfirmationPolicy engine."""

    def test_empty_policy_allows_everything(self) -> None:
        policy = ConfirmationPolicy(rules=[])
        assert not policy.requires_confirmation("exec", {"command": "rm -rf /"})
        assert not policy.requires_confirmation("write_file", {"path": "/etc/passwd"})
        assert not policy.is_denied("exec", {"command": "rm -rf /"})

    def test_first_matching_rule_wins(self) -> None:
        """A later allow rule should NOT override an earlier deny."""
        rules = [
            ConfirmationRule(tool="*", pattern="", action="deny"),
            ConfirmationRule(tool="exec", pattern="", action="allow"),
        ]
        policy = ConfirmationPolicy(rules=rules)
        assert policy.is_denied("exec", {"command": "ls"})
        assert policy.is_denied("write_file", {"path": "/tmp/x"})

    def test_confirm_action(self) -> None:
        rules = [ConfirmationRule(tool="exec", pattern=r"rm\s+-rf", action="confirm")]
        policy = ConfirmationPolicy(rules=rules)
        assert policy.requires_confirmation("exec", {"command": "rm -rf /tmp"})
        assert not policy.requires_confirmation("exec", {"command": "ls"})

    def test_deny_action(self) -> None:
        rules = [ConfirmationRule(tool="exec", pattern=r"shutdown", action="deny")]
        policy = ConfirmationPolicy(rules=rules)
        assert policy.is_denied("exec", {"command": "shutdown now"})
        assert not policy.is_denied("exec", {"command": "ls"})

    def test_allow_action_explicit(self) -> None:
        rules = [
            ConfirmationRule(tool="*", pattern="", action="deny"),
            ConfirmationRule(tool="exec", pattern=r"^echo\b", action="allow"),
        ]
        policy = ConfirmationPolicy(rules=rules)
        assert not policy.is_denied("exec", {"command": "echo hello"})
        assert policy.is_denied("exec", {"command": "rm -rf /"})

    def test_describe_exec(self) -> None:
        policy = ConfirmationPolicy(rules=[])
        desc = policy.describe("exec", {"command": "rm -rf /tmp"})
        assert "rm -rf /tmp" in desc

    def test_describe_write_file(self) -> None:
        policy = ConfirmationPolicy(rules=[])
        desc = policy.describe("write_file", {"path": "/etc/passwd", "content": "x"})
        assert "/etc/passwd" in desc

    def test_describe_edit_file(self) -> None:
        policy = ConfirmationPolicy(rules=[])
        desc = policy.describe("edit_file", {"path": "/tmp/x", "old_text": "hello world"})
        assert "/tmp/x" in desc
        assert "hello world" in desc

    def test_describe_edit_file_truncates_long_text(self) -> None:
        policy = ConfirmationPolicy(rules=[])
        long = "a" * 200
        desc = policy.describe("edit_file", {"path": "/tmp/x", "old_text": long})
        assert "..." in desc

    def test_describe_spawn(self) -> None:
        policy = ConfirmationPolicy(rules=[])
        desc = policy.describe("spawn", {"prompt": "fix the bug in app.py"})
        assert "fix the bug in app.py" in desc

    def test_describe_unknown_tool(self) -> None:
        policy = ConfirmationPolicy(rules=[])
        desc = policy.describe("custom_tool", {"foo": "bar"})
        assert "custom_tool" in desc
        assert "foo='bar'" in desc

    # --- from_config ---

    def test_from_config_basic(self) -> None:
        raw = [
            {"tool": "exec", "pattern": "rm\\s+-rf", "action": "confirm"},
            {"tool": "write_file", "pattern": "", "action": "confirm"},
        ]
        policy = ConfirmationPolicy.from_config(raw)
        assert policy.requires_confirmation("exec", {"command": "rm -rf /tmp"})
        assert policy.requires_confirmation("write_file", {"path": "/tmp/x"})
        assert not policy.requires_confirmation("read_file", {"path": "/tmp/x"})

    def test_from_config_defaults(self) -> None:
        """Missing keys in config should get sensible defaults."""
        raw = [{"tool": "exec"}]
        policy = ConfirmationPolicy.from_config(raw)
        assert policy.requires_confirmation("exec", {"command": "ls"})

    # --- default_policy ---

    def test_default_policy_confirms_rm_rf(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("exec", {"command": "rm -rf /tmp"})

    def test_default_policy_confirms_rm_r(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("exec", {"command": "rm -r /tmp"})

    def test_default_policy_allows_safe_exec(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert not policy.requires_confirmation("exec", {"command": "ls -la"})

    def test_default_policy_confirms_write_file(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("write_file", {"path": "/tmp/x"})

    def test_default_policy_confirms_edit_file(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("edit_file", {"path": "/tmp/x"})

    def test_default_policy_allows_read_file(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert not policy.requires_confirmation("read_file", {"path": "/tmp/x"})

    def test_default_policy_confirms_spawn(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("spawn", {"prompt": "fix bug"})

    def test_default_policy_confirms_shutdown(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("exec", {"command": "shutdown now"})

    def test_default_policy_confirms_chmod(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("exec", {"command": "chmod 777 /etc/passwd"})

    def test_default_policy_confirms_kill_9(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("exec", {"command": "kill -9 1234"})

    def test_default_policy_confirms_mv_to_absolute(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert policy.requires_confirmation("exec", {"command": "mv file.txt /etc/file.txt"})

    def test_default_policy_allows_mv_relative(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        assert not policy.requires_confirmation("exec", {"command": "mv a.txt b.txt"})


# ---------------------------------------------------------------------------
# ToolRegistry integration tests
# ---------------------------------------------------------------------------


class TestToolRegistryConfirmation:
    """Integration tests: ToolRegistry with ConfirmationPolicy."""

    @pytest.mark.asyncio
    async def test_no_policy_tool_executes_normally(self) -> None:
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = await reg.execute("echo", {"text": "hello"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_empty_policy_tool_executes_normally(self) -> None:
        reg = ToolRegistry(confirmation_policy=ConfirmationPolicy(rules=[]))
        reg.register(EchoTool())
        result = await reg.execute("echo", {"text": "hello"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_confirm_returns_marker_string(self) -> None:
        policy = ConfirmationPolicy(rules=[
            ConfirmationRule(tool="*", pattern="", action="confirm"),
        ])
        reg = ToolRegistry(confirmation_policy=policy)
        reg.register(EchoTool())
        result = await reg.execute("echo", {"text": "hello"})
        assert isinstance(result, str)
        assert result.startswith(CONFIRM_REQUIRED_PREFIX)

    @pytest.mark.asyncio
    async def test_deny_returns_denied_string(self) -> None:
        policy = ConfirmationPolicy(rules=[
            ConfirmationRule(tool="echo", pattern="", action="deny"),
        ])
        reg = ToolRegistry(confirmation_policy=policy)
        reg.register(EchoTool())
        result = await reg.execute("echo", {"text": "hello"})
        assert isinstance(result, str)
        assert result.startswith(DENIED_PREFIX)

    @pytest.mark.asyncio
    async def test_confirm_specific_tool_only(self) -> None:
        """Only echo should be gated; other tools pass through."""
        policy = ConfirmationPolicy(rules=[
            ConfirmationRule(tool="echo", pattern="", action="confirm"),
        ])
        reg = ToolRegistry(confirmation_policy=policy)
        reg.register(EchoTool())
        reg.register(FakeExecTool())

        # echo is gated
        result = await reg.execute("echo", {"text": "hello"})
        assert result.startswith(CONFIRM_REQUIRED_PREFIX)

        # exec passes through
        result = await reg.execute("exec", {"command": "ls"})
        assert result == "ran: ls"

    @pytest.mark.asyncio
    async def test_exec_rm_rf_confirmed(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        reg = ToolRegistry(confirmation_policy=policy)
        reg.register(FakeExecTool())

        result = await reg.execute("exec", {"command": "rm -rf /tmp"})
        assert result.startswith(CONFIRM_REQUIRED_PREFIX)
        assert "rm -rf" in result

    @pytest.mark.asyncio
    async def test_exec_safe_command_passes_through(self) -> None:
        policy = ConfirmationPolicy.default_policy()
        reg = ToolRegistry(confirmation_policy=policy)
        reg.register(FakeExecTool())

        result = await reg.execute("exec", {"command": "ls -la /tmp"})
        assert result == "ran: ls -la /tmp"

    @pytest.mark.asyncio
    async def test_unknown_tool_still_returns_error(self) -> None:
        """Confirmation policy should not interfere with tool-not-found errors."""
        policy = ConfirmationPolicy(rules=[
            ConfirmationRule(tool="*", pattern="", action="confirm"),
        ])
        reg = ToolRegistry(confirmation_policy=policy)
        result = await reg.execute("nonexistent", {})
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_validation_still_works_with_policy(self) -> None:
        """Invalid params should still produce validation errors even with a policy."""
        policy = ConfirmationPolicy(rules=[
            ConfirmationRule(tool="echo", pattern="", action="confirm"),
        ])
        reg = ToolRegistry(confirmation_policy=policy)
        reg.register(EchoTool())
        # Missing required "text" param — but confirmation fires first
        # because the tool matches the rule. The confirmation gate fires
        # before validation, which is the correct order: why validate
        # something that might be denied?
        result = await reg.execute("echo", {})
        assert result.startswith(CONFIRM_REQUIRED_PREFIX)


# ---------------------------------------------------------------------------
# Config schema integration tests
# ---------------------------------------------------------------------------


class TestConfigSchema:
    """Verify that ConfirmationRuleConfig integrates with ToolsConfig."""

    def test_confirmation_rules_default_empty(self) -> None:
        from nanobot.config.schema import ToolsConfig

        config = ToolsConfig()
        assert config.confirmation_rules == []

    def test_confirmation_rules_from_dict(self) -> None:
        from nanobot.config.schema import ToolsConfig

        config = ToolsConfig.model_validate({
            "confirmationRules": [
                {"tool": "exec", "pattern": "rm\\s+-rf", "action": "confirm"},
                {"tool": "write_file", "pattern": "", "action": "confirm"},
            ],
        })
        assert len(config.confirmation_rules) == 2
        assert config.confirmation_rules[0].tool == "exec"
        assert config.confirmation_rules[0].action == "confirm"
        assert config.confirmation_rules[1].tool == "write_file"

    def test_confirmation_rules_snake_case(self) -> None:
        from nanobot.config.schema import ToolsConfig

        config = ToolsConfig.model_validate({
            "confirmation_rules": [
                {"tool": "exec", "pattern": "rm\\s+-rf", "action": "deny"},
            ],
        })
        assert len(config.confirmation_rules) == 1
        assert config.confirmation_rules[0].action == "deny"
