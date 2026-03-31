"""Tests for the /heartbeat builtin command."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import cmd_heartbeat
from nanobot.command.router import CommandContext


def _make_ctx(channel: str = "telegram", chat_id: str = "123") -> CommandContext:
    msg = InboundMessage(
        channel=channel,
        chat_id=chat_id,
        content="/heartbeat",
        sender_id="user1",
    )
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw="/heartbeat", loop=MagicMock())


@pytest.mark.asyncio
async def test_cmd_heartbeat_no_service() -> None:
    """Returns error message when heartbeat service is None."""
    ctx = _make_ctx()
    ctx.loop.heartbeat_service = None
    out = await cmd_heartbeat(ctx)
    assert "not available" in out.content


@pytest.mark.asyncio
async def test_cmd_heartbeat_skip() -> None:
    """Reports skip when LLM decides nothing to do."""
    ctx = _make_ctx()
    ctx.loop.heartbeat_service = AsyncMock()
    ctx.loop.heartbeat_service.trigger_now.return_value = {
        "action": "skip", "tasks": "", "result": "",
    }
    out = await cmd_heartbeat(ctx)
    assert "skip" in out.content
    assert out.channel == "telegram"
    assert out.chat_id == "123"


@pytest.mark.asyncio
async def test_cmd_heartbeat_run() -> None:
    """Reports task and result when LLM decides run."""
    ctx = _make_ctx()
    ctx.loop.heartbeat_service = AsyncMock()
    ctx.loop.heartbeat_service.trigger_now.return_value = {
        "action": "run", "tasks": "check open tasks", "result": "all clear",
    }
    out = await cmd_heartbeat(ctx)
    assert "run" in out.content
    assert "check open tasks" in out.content
    assert "all clear" in out.content


@pytest.mark.asyncio
async def test_cmd_heartbeat_run_no_result() -> None:
    """Handles empty result gracefully."""
    ctx = _make_ctx()
    ctx.loop.heartbeat_service = AsyncMock()
    ctx.loop.heartbeat_service.trigger_now.return_value = {
        "action": "run", "tasks": "fix bug", "result": "",
    }
    out = await cmd_heartbeat(ctx)
    assert "run" in out.content
    assert "fix bug" in out.content


@pytest.mark.asyncio
async def test_cmd_heartbeat_exception() -> None:
    """Catches exceptions and returns error message."""
    ctx = _make_ctx()
    ctx.loop.heartbeat_service = AsyncMock()
    ctx.loop.heartbeat_service.trigger_now.side_effect = RuntimeError("boom")
    out = await cmd_heartbeat(ctx)
    assert "boom" in out.content
