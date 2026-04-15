"""Tests for SpawnTool argument forwarding and checkpoint policy wiring."""

from __future__ import annotations

import pytest

from nanobot.agent.tools.spawn import SpawnTool
from nanobot.checkpoint.policy import ReviewPolicy


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def spawn(self, **kwargs):
        self.calls.append(kwargs)
        return "spawned"


@pytest.mark.asyncio
async def test_execute_forwards_context_and_caps_hard_cap() -> None:
    manager = _DummyManager()
    tool = SpawnTool(manager)
    tool.set_context(
        channel="telegram",
        chat_id="chat-1",
        message_thread_id="topic-7",
        request_id="req-123",
    )

    result = await tool.execute(
        task="do thing",
        label="label",
        max_iterations=25,
        hard_cap=99_999,
    )

    assert result == "spawned"
    assert len(manager.calls) == 1
    call = manager.calls[0]
    assert call["task"] == "do thing"
    assert call["label"] == "label"
    assert call["origin_channel"] == "telegram"
    assert call["origin_chat_id"] == "chat-1"
    assert call["session_key"] == "telegram:chat-1"
    assert call["request_id"] == "req-123"
    assert call["message_thread_id"] == "topic-7"
    assert call["max_iterations"] == 25
    assert call["hard_cap"] == 3600


@pytest.mark.asyncio
async def test_execute_enables_checkpoint_policy_from_flag_and_overrides() -> None:
    manager = _DummyManager()
    tool = SpawnTool(manager)

    await tool.execute(
        task="review this",
        enable_checkpoint_policy=True,
        checkpoint_threshold_pct=0.9,
        checkpoint_read_only_hint=True,
        checkpoint_escalation_cooldown=7,
        checkpoint_review_timeout=200,
        checkpoint_loop_window=10,
        checkpoint_max_checkpoints=4,
    )

    call = manager.calls[0]
    assert "checkpoint_policy" in call
    policy = call["checkpoint_policy"]
    assert isinstance(policy, ReviewPolicy)
    assert policy.checkpoint_threshold_pct == 0.9
    assert policy.read_only_hint is True
    assert policy.escalation_cooldown == 7
    assert policy.review_timeout == 200
    assert policy.loop_window == 10
    assert policy.max_checkpoints == 4


@pytest.mark.asyncio
async def test_execute_omits_checkpoint_policy_when_not_requested() -> None:
    manager = _DummyManager()
    tool = SpawnTool(manager)

    await tool.execute(task="simple task")

    call = manager.calls[0]
    assert "checkpoint_policy" not in call
