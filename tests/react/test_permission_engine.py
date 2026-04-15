"""Tests for PermissionEngine — 3-level tool permission gating."""

from __future__ import annotations

import pytest

from nanobot.react.permission_engine import (
    Permission,
    PermissionEngine,
    create_permission_engine,
)

# ---------------------------------------------------------------------------
# Unit tests: PermissionEngine
# ---------------------------------------------------------------------------


class TestPermissionEngineDefaults:
    """Verify built-in default policy."""

    def setup_method(self):
        self.engine = PermissionEngine()

    def test_exec_defaults_to_ask(self):
        result = self.engine.check("exec")
        assert result.permission == Permission.ASK

    def test_write_file_defaults_to_ask(self):
        result = self.engine.check("write_file")
        assert result.permission == Permission.ASK

    def test_edit_file_defaults_to_ask(self):
        result = self.engine.check("edit_file")
        assert result.permission == Permission.ASK

    def test_restart_gateway_defaults_to_deny(self):
        result = self.engine.check("restart_gateway")
        assert result.permission == Permission.DENY
        assert "denied" in result.reason

    def test_unknown_tool_defaults_to_allow(self):
        result = self.engine.check("read_file")
        assert result.permission == Permission.ALLOW
        assert result.reason == ""

    def test_list_dir_defaults_to_allow(self):
        result = self.engine.check("list_dir")
        assert result.permission == Permission.ALLOW


class TestPermissionEngineCustomPolicy:
    """Custom policy overrides built-in defaults."""

    def test_custom_policy_overrides_default(self):
        engine = PermissionEngine(policy={"exec": Permission.DENY})
        result = engine.check("exec")
        assert result.permission == Permission.DENY

    def test_custom_policy_can_downgrade(self):
        engine = PermissionEngine(policy={"restart_gateway": Permission.ALLOW})
        result = engine.check("restart_gateway")
        assert result.permission == Permission.ALLOW

    def test_custom_policy_does_not_affect_others(self):
        engine = PermissionEngine(policy={"exec": Permission.DENY})
        result = engine.check("read_file")
        assert result.permission == Permission.ALLOW


class TestPermissionEngineFilterCalls:
    """filter_calls splits tool calls into allowed and blocked."""

    def test_all_allowed(self):
        engine = PermissionEngine()
        calls = [("read_file", {"path": "/tmp/a"}), ("web_search", {"query": "hello"})]
        allowed, blocked = engine.filter_calls(calls)
        assert len(allowed) == 2
        assert len(blocked) == 0

    def test_mixed_permissions(self):
        engine = PermissionEngine()
        calls = [
            ("read_file", {"path": "/tmp/a"}),
            ("restart_gateway", {}),
            ("exec", {"command": "ls"}),
        ]
        allowed, blocked = engine.filter_calls(calls)
        allowed_names = {name for name, _ in allowed}
        assert "read_file" in allowed_names
        assert "restart_gateway" not in allowed_names
        assert "exec" not in allowed_names
        assert len(blocked) == 2

    def test_all_blocked(self):
        engine = PermissionEngine()
        calls = [("restart_gateway", {})]
        allowed, blocked = engine.filter_calls(calls)
        assert len(allowed) == 0
        assert len(blocked) == 1
        assert blocked[0].permission == Permission.DENY

    def test_empty_calls(self):
        engine = PermissionEngine()
        allowed, blocked = engine.filter_calls([])
        assert allowed == []
        assert blocked == []


class TestPermissionEngineArguments:
    """check accepts optional arguments (for future use)."""

    def test_check_with_none_arguments(self):
        engine = PermissionEngine()
        result = engine.check("exec", None)
        assert result.permission == Permission.ASK

    def test_check_with_empty_arguments(self):
        engine = PermissionEngine()
        result = engine.check("exec", {})
        assert result.permission == Permission.ASK


class TestCreatePermissionEngine:
    """Factory function with graceful degradation."""

    def test_returns_engine_with_none_policy(self):
        engine = create_permission_engine(None)
        assert engine is not None
        assert isinstance(engine, PermissionEngine)

    def test_returns_engine_with_empty_policy(self):
        engine = create_permission_engine({})
        assert engine is not None

    def test_returns_engine_with_custom_policy(self):
        engine = create_permission_engine({"exec": Permission.DENY})
        assert engine is not None
        assert engine.check("exec").permission == Permission.DENY

    def test_returns_none_on_exception(self):
        def _bad_init():
            raise RuntimeError("boom")

        import nanobot.react.permission_engine as mod
        orig = mod.PermissionEngine
        mod.PermissionEngine = _bad_init  # type: ignore[assignment]
        try:
            result = create_permission_engine()
            assert result is None
        finally:
            mod.PermissionEngine = orig


# ---------------------------------------------------------------------------
# Integration: PermissionEngine in runner
# ---------------------------------------------------------------------------


class TestRunnerPermissionGate:
    """Verify that PermissionEngine gates tool calls in the runner loop."""

    @pytest.mark.asyncio
    async def test_denied_tool_returns_permission_error(self):
        from unittest.mock import AsyncMock, MagicMock

        from nanobot.agent.runner import AgentRunner, AgentRunSpec
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        provider = MagicMock()
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
            content="thinking",
            tool_calls=[ToolCallRequest(id="call_1", name="restart_gateway", arguments={})],
            usage={"prompt_tokens": 5, "completion_tokens": 3},
        ))
        tools = MagicMock()
        tools.get_definitions.return_value = []
        tools.execute = AsyncMock(return_value="should not be called")

        engine = PermissionEngine()
        runner = AgentRunner(provider)
        result = await runner.run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "restart"}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            permission_engine=engine,
        ))

        # restart_gateway is DENY by default
        assert result.stop_reason == "completed"
        assert result.tool_events[0]["status"] == "denied"
        assert "denied" in result.tool_events[0]["detail"].lower()
        tools.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allowed_tool_executes_normally(self):
        from unittest.mock import AsyncMock, MagicMock

        from nanobot.agent.runner import AgentRunner, AgentRunSpec
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        provider = MagicMock()
        call_count = {"n": 0}

        async def fake_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCallRequest(id="c1", name="read_file", arguments={"path": "/tmp/x"})],
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                )
            return LLMResponse(content="done", tool_calls=[], usage={})

        provider.chat_with_retry = fake_chat
        tools = MagicMock()
        tools.get_definitions.return_value = []
        tools.execute = AsyncMock(return_value="file contents")

        engine = PermissionEngine()
        runner = AgentRunner(provider)
        result = await runner.run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "read file"}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            permission_engine=engine,
        ))

        assert result.final_content == "done"
        assert result.tool_events[0]["status"] == "ok"
        tools.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_permission_engine_allows_everything(self):
        """Without a permission engine, all tools execute normally."""
        from unittest.mock import AsyncMock, MagicMock

        from nanobot.agent.runner import AgentRunner, AgentRunSpec
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        provider = MagicMock()
        call_count = {"n": 0}

        async def fake_chat(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCallRequest(id="c1", name="exec", arguments={"command": "ls"})],
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                )
            return LLMResponse(content="done", tool_calls=[], usage={})

        provider.chat_with_retry = fake_chat
        tools = MagicMock()
        tools.get_definitions.return_value = []
        tools.execute = AsyncMock(return_value="output")

        runner = AgentRunner(provider)
        result = await runner.run(AgentRunSpec(
            initial_messages=[{"role": "user", "content": "run ls"}],
            tools=tools,
            model="test-model",
            max_iterations=3,
            # No permission_engine — should allow everything
        ))

        assert result.final_content == "done"
        tools.execute.assert_awaited_once()
