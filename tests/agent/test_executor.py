"""Tests for nanobot.agent.executor — executor resolver."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestResolveExecutorAPI:
    """Tests for API-based executor resolution."""

    @pytest.fixture()
    def mock_config(self):
        """Create a mock config with provider keys."""
        cfg = MagicMock()
        cfg.providers.custom.api_key = "test-zhipu-key"
        cfg.providers.custom.api_base = "https://api.z.ai/api/coding/paas/v4"
        cfg.providers.openrouter.api_key = "test-or-key"
        cfg.providers.openrouter.api_base = None
        return cfg

    def test_resolve_glm_turbo(self, mock_config):
        from nanobot.agent.executor import resolve_executor

        info = resolve_executor("glm-turbo", mock_config)

        assert info.alias == "glm-turbo"
        assert info.model == "glm-5-turbo"
        assert info.is_api
        assert info.provider is not None

    def test_resolve_glm_51(self, mock_config):
        from nanobot.agent.executor import resolve_executor

        info = resolve_executor("glm-5.1", mock_config)

        assert info.alias == "glm-5.1"
        assert info.model == "glm-5.1"
        assert info.is_api
        assert info.provider is not None

    def test_resolve_openrouter(self, mock_config):
        from nanobot.agent.executor import resolve_executor

        info = resolve_executor("openrouter", mock_config)

        assert info.alias == "openrouter"
        assert info.model == "qwen/qwen3.6-plus:free"
        assert info.is_api
        assert info.provider is not None

    def test_provider_created_with_correct_api_key(self, mock_config):
        from nanobot.agent.executor import resolve_executor

        info = resolve_executor("glm-turbo", mock_config)
        # The provider should have been created with the config's api_key
        assert info.provider is not None
        assert info.provider.api_key == "test-zhipu-key"

    def test_no_config_warns_but_still_returns(self):
        """Without config, provider is created with no key (warns in logs)."""
        from nanobot.agent.executor import resolve_executor

        info = resolve_executor("glm-turbo", config=None)

        assert info.is_api
        assert info.model == "glm-5-turbo"
        # Provider still created, just without a key
        assert info.provider is not None


class TestResolveExecutorCLI:
    """Tests for CLI-based executor resolution (V1 stubs)."""

    def test_claude_native_is_cli(self):
        from nanobot.agent.executor import resolve_executor

        info = resolve_executor("claude-native")

        assert info.alias == "claude-native"
        assert info.is_cli
        assert info.acpx_agent == "claude"
        assert info.provider is None
        assert info.model == "claude-sonnet-4-20250514"

    def test_claude_zai_is_cli(self):
        from nanobot.agent.executor import resolve_executor

        info = resolve_executor("claude-zai")

        assert info.is_cli
        assert info.acpx_agent == "claude"
        assert info.model == "claude-sonnet-4-20250514"

    def test_codex_is_cli(self):
        from nanobot.agent.executor import resolve_executor

        for alias, model in [("codex-5.3", "gpt-5.3-codex"), ("codex-5.4", "gpt-5.4"), ("codex-5.4-mini", "gpt-5.4-mini")]:
            info = resolve_executor(alias)
            assert info.is_cli
            assert info.acpx_agent == "codex"
            assert info.model == model


class TestResolveExecutorErrors:
    """Error handling tests."""

    def test_unknown_executor_raises(self):
        from nanobot.agent.executor import resolve_executor

        with pytest.raises(ValueError, match="Unknown executor 'nonexistent'"):
            resolve_executor("nonexistent")

    def test_error_message_lists_known_executors(self):
        from nanobot.agent.executor import resolve_executor

        with pytest.raises(ValueError) as exc_info:
            resolve_executor("bogus")

        msg = str(exc_info.value)
        assert "glm-turbo" in msg
        assert "openrouter" in msg


class TestGetKnownExecutors:
    """Utility function tests."""

    def test_returns_all_executors(self):
        from nanobot.agent.executor import get_known_executors

        executors = get_known_executors()
        assert len(executors) == 8
        assert "glm-turbo" in executors
        assert "glm-5.1" in executors
        assert "openrouter" in executors
        assert "claude-native" in executors
        assert "claude-zai" in executors
        assert "codex-5.3" in executors
        assert "codex-5.4" in executors
        assert "codex-5.4-mini" in executors

    def test_is_sorted(self):
        from nanobot.agent.executor import get_known_executors

        executors = get_known_executors()
        assert executors == sorted(executors)


class TestExecutorInfoProperties:
    """ExecutorInfo dataclass property tests."""

    def test_api_mode_properties(self):
        from nanobot.agent.executor import ExecutorInfo

        info = ExecutorInfo(model="test", mode="api", alias="test-alias")
        assert info.is_api is True
        assert info.is_cli is False
        assert info.acpx_agent is None

    def test_cli_mode_properties(self):
        from nanobot.agent.executor import ExecutorInfo

        info = ExecutorInfo(model="test", mode="cli", alias="test-alias", acpx_agent="codex")
        assert info.is_api is False
        assert info.is_cli is True
        assert info.acpx_agent == "codex"

    def test_cli_mode_without_acpx_agent_is_not_cli(self):
        """is_cli should be False when acpx_agent is None, even if mode is 'cli'."""
        from nanobot.agent.executor import ExecutorInfo

        info = ExecutorInfo(model="test", mode="cli", alias="test-alias")
        assert info.is_cli is False
        assert info.acpx_agent is None

    def test_api_executors_have_no_acpx_agent(self):
        from nanobot.agent.executor import resolve_executor

        for alias in ("glm-turbo", "glm-5.1", "openrouter"):
            info = resolve_executor(alias)
            assert info.acpx_agent is None
            assert info.is_cli is False
