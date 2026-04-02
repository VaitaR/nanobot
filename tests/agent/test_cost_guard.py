"""Tests for CostGuard — token budget enforcement and rate limiting."""

from __future__ import annotations

import time

import pytest

from nanobot.agent.cost_guard import CostGuard, _estimate_cost

# ======================================================================
# Fixtures / helpers
# ======================================================================

def _guard(
    max_tokens_per_turn: int = 0,
    max_tokens_per_session: int = 0,
    max_cost_usd: float = 0.0,
    rate_limit_rpm: int = 0,
) -> CostGuard:
    return CostGuard(
        max_tokens_per_turn=max_tokens_per_turn,
        max_tokens_per_session=max_tokens_per_session,
        max_cost_usd=max_cost_usd,
        rate_limit_rpm=rate_limit_rpm,
    )


def _guard_from_config(
    max_tokens_per_turn: int = 0,
    max_tokens_per_session: int = 0,
    max_cost_usd: float = 0.0,
    rate_limit_rpm: int = 0,
) -> CostGuard:
    from nanobot.config.schema import CostPolicy
    return CostGuard.from_config(CostPolicy(
        max_tokens_per_turn=max_tokens_per_turn,
        max_tokens_per_session=max_tokens_per_session,
        max_cost_usd=max_cost_usd,
        rate_limit_rpm=rate_limit_rpm,
    ))


# ======================================================================
# Construction
# ======================================================================

class TestConstruction:
    def test_disabled_guard_allows_everything(self) -> None:
        guard = CostGuard.disabled()
        assert guard.max_tokens_per_turn == 0
        assert guard.max_tokens_per_session == 0
        assert guard.max_cost_usd == 0.0
        assert guard.rate_limit_rpm == 0

    def test_from_config_reads_fields(self) -> None:
        guard = _guard_from_config(
            max_tokens_per_turn=1000,
            max_tokens_per_session=5000,
            max_cost_usd=1.0,
            rate_limit_rpm=10,
        )
        assert guard.max_tokens_per_turn == 1000
        assert guard.max_tokens_per_session == 5000
        assert guard.max_cost_usd == 1.0
        assert guard.rate_limit_rpm == 10

    def test_disabled_guard_check_always_allowed(self) -> None:
        guard = CostGuard.disabled()
        result = guard.check_before_call()
        assert result.allowed is True
        assert result.reason is None


# ======================================================================
# Session token budget
# ======================================================================

class TestSessionTokenBudget:
    def test_allows_when_under_budget(self) -> None:
        guard = _guard(max_tokens_per_session=1000)
        guard.record_usage({"prompt_tokens": 100, "completion_tokens": 100})
        result = guard.check_before_call()
        assert result.allowed is True

    def test_blocks_when_at_budget(self) -> None:
        guard = _guard(max_tokens_per_session=200)
        guard.record_usage({"prompt_tokens": 100, "completion_tokens": 100})
        result = guard.check_before_call()
        assert result.allowed is False
        assert "Session token budget exceeded" in (result.reason or "")

    def test_blocks_when_over_budget(self) -> None:
        guard = _guard(max_tokens_per_session=150)
        guard.record_usage({"prompt_tokens": 100, "completion_tokens": 100})
        result = guard.check_before_call()
        assert result.allowed is False

    def test_no_limit_when_zero(self) -> None:
        guard = _guard(max_tokens_per_session=0)
        guard.record_usage({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
        result = guard.check_before_call()
        assert result.allowed is True

    def test_session_tokens_property(self) -> None:
        guard = _guard(max_tokens_per_session=10_000)
        guard.record_usage({"prompt_tokens": 100, "completion_tokens": 50})
        assert guard.session_tokens == 150
        guard.record_usage({"prompt_tokens": 200, "completion_tokens": 75})
        assert guard.session_tokens == 425


# ======================================================================
# Per-turn token budget
# ======================================================================

class TestPerTurnTokenBudget:
    def test_blocks_when_turn_exceeds_limit(self) -> None:
        guard = _guard(max_tokens_per_turn=200)
        # First call sets the baseline — 300 tokens
        guard.record_usage({"prompt_tokens": 200, "completion_tokens": 100})
        # Second check: estimated prompt (100) + last call tokens (300) = 400 > 200
        result = guard.check_before_call(estimated_prompt_tokens=100)
        assert result.allowed is False
        assert "Per-turn" in (result.reason or "")

    def test_allows_first_call(self) -> None:
        guard = _guard(max_tokens_per_turn=200)
        # No previous data — should be allowed
        result = guard.check_before_call(estimated_prompt_tokens=300)
        assert result.allowed is True

    def test_no_limit_when_zero(self) -> None:
        guard = _guard(max_tokens_per_turn=0)
        guard.record_usage({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
        result = guard.check_before_call(estimated_prompt_tokens=1_000_000)
        assert result.allowed is True


# ======================================================================
# Cost budget
# ======================================================================

class TestCostBudget:
    def test_blocks_when_cost_exceeded(self) -> None:
        # Use a high enough max_cost_usd that we can hit it
        guard = _guard(max_cost_usd=0.001)
        # claude-opus-4-5: 15/M input, 75/M output
        # 100 prompt tokens = 0.0015 USD — already over 0.001 limit
        guard.record_usage(
            {"prompt_tokens": 100, "completion_tokens": 0},
            model="claude-opus-4-5",
        )
        result = guard.check_before_call()
        assert result.allowed is False
        assert "cost budget exceeded" in (result.reason or "").lower()

    def test_allows_when_under_cost(self) -> None:
        guard = _guard(max_cost_usd=10.0)
        guard.record_usage(
            {"prompt_tokens": 100, "completion_tokens": 100},
            model="claude-opus-4-5",
        )
        result = guard.check_before_call()
        assert result.allowed is True

    def test_no_cost_tracking_without_model(self) -> None:
        guard = _guard(max_cost_usd=0.001)
        guard.record_usage({"prompt_tokens": 100, "completion_tokens": 100})
        # Without a model, cost is not estimated — so cost limit is never hit
        assert guard.session_cost_usd == 0.0
        result = guard.check_before_call()
        assert result.allowed is True

    def test_session_cost_property(self) -> None:
        guard = _guard(max_cost_usd=10.0)
        guard.record_usage(
            {"prompt_tokens": 100, "completion_tokens": 0},
            model="claude-opus-4-5",
        )
        assert guard.session_cost_usd > 0.0


# ======================================================================
# RPM rate limiting
# ======================================================================

class TestRateLimiting:
    def test_allows_within_limit(self) -> None:
        guard = _guard(rate_limit_rpm=3)
        for _ in range(2):
            guard.record_usage({"prompt_tokens": 10, "completion_tokens": 10})
        result = guard.check_before_call()
        assert result.allowed is True

    def test_blocks_at_limit(self) -> None:
        guard = _guard(rate_limit_rpm=2)
        guard.record_usage({"prompt_tokens": 10, "completion_tokens": 10})
        guard.record_usage({"prompt_tokens": 10, "completion_tokens": 10})
        result = guard.check_before_call()
        assert result.allowed is False
        assert "Rate limit" in (result.reason or "")

    def test_sliding_window_allows_after_time_passes(self) -> None:
        guard = _guard(rate_limit_rpm=1)
        guard.record_usage({"prompt_tokens": 10, "completion_tokens": 10})

        # Immediately — should block
        result = guard.check_before_call()
        assert result.allowed is False

        # Manually push timestamp into the past (beyond 60s window)
        guard._call_timestamps[0] = time.monotonic() - 61.0
        result = guard.check_before_call()
        assert result.allowed is True

    def test_no_limit_when_zero(self) -> None:
        guard = _guard(rate_limit_rpm=0)
        for _ in range(100):
            guard.record_usage({"prompt_tokens": 10, "completion_tokens": 10})
        result = guard.check_before_call()
        assert result.allowed is True


# ======================================================================
# Cost estimation
# ======================================================================

class TestCostEstimation:
    def test_known_model(self) -> None:
        cost = _estimate_cost("claude-opus-4-5", prompt_tokens=1_000_000, completion_tokens=0)
        assert cost == pytest.approx(15.0)

    def test_known_model_with_output(self) -> None:
        cost = _estimate_cost("claude-opus-4-5", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost == pytest.approx(90.0)

    def test_unknown_model_returns_zero(self) -> None:
        cost = _estimate_cost("unknown-model-xyz", prompt_tokens=1000, completion_tokens=1000)
        assert cost == 0.0

    def test_prefixed_model_strips_prefix(self) -> None:
        cost = _estimate_cost("anthropic/claude-sonnet-4-20250514", 1_000_000, 0)
        assert cost == pytest.approx(3.0)

    def test_gpt4o_mini(self) -> None:
        cost = _estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.75)

    def test_deepseek(self) -> None:
        cost = _estimate_cost("deepseek-chat", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.42)


# ======================================================================
# Integration: runner respects cost guard
# ======================================================================

@pytest.mark.asyncio
async def test_runner_stops_on_cost_cap():
    """AgentRunner.run() breaks with stop_reason='cost_cap' when guard blocks.

    The first LLM response includes a tool call so the runner loops back.
    On iteration 2, the guard sees session_tokens=200 >= limit=10 and blocks.
    """
    from unittest.mock import AsyncMock, MagicMock

    from nanobot.agent.runner import AgentRunner, AgentRunSpec
    from nanobot.providers.base import LLMResponse, ToolCallRequest

    guard = _guard(max_tokens_per_session=10)

    provider = MagicMock()

    async def chat_with_retry(**kwargs):
        return LLMResponse(
            content="calling a tool",
            tool_calls=[ToolCallRequest(id="tc_1", name="do_thing", arguments={})],
            usage={"prompt_tokens": 100, "completion_tokens": 100},
        )

    provider.chat_with_retry = chat_with_retry

    tools = MagicMock()
    tools.get_definitions.return_value = [
        {"type": "function", "function": {"name": "do_thing", "parameters": {}}}
    ]
    tools.execute = AsyncMock(return_value="done")

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        model="test-model",
        max_iterations=5,
        cost_guard=guard,
    ))

    assert result.stop_reason == "cost_cap"
    assert result.final_content is not None
    assert "token budget" in result.final_content.lower() or "budget" in result.final_content.lower()
    # Only one LLM call should have been made (first succeeded, second is blocked)
    assert guard.session_tokens == 200  # 100 prompt + 100 completion from the one call


@pytest.mark.asyncio
async def test_runner_ignores_guard_when_none():
    """When cost_guard is None, runner behaves normally."""
    from unittest.mock import AsyncMock, MagicMock

    from nanobot.agent.runner import AgentRunner, AgentRunSpec
    from nanobot.providers.base import LLMResponse

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="done",
        tool_calls=[],
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []

    runner = AgentRunner(provider)
    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        model="test-model",
        max_iterations=5,
        cost_guard=None,
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "done"
