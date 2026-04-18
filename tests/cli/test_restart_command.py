"""Tests for /restart slash command."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.providers.base import LLMResponse


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


def _make_loop_with_workspace(workspace: Path):
    """Create a minimal AgentLoop with a real workspace path."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestRestartCommand:

    @pytest.mark.asyncio
    async def test_restart_sends_message_and_calls_execv(self):
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop()
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        with patch("nanobot.command.builtin.os.execv") as mock_execv:
            out = await cmd_restart(ctx)
            assert "Restarting" in out.content

            await asyncio.sleep(1.5)
            mock_execv.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_writes_topic_to_restart_pending_marker(self, tmp_path: Path):
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext

        loop = SimpleNamespace(workspace=tmp_path)
        msg = InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="chat-1",
            content="/restart",
            metadata={"message_thread_id": "42"},
        )
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        with patch("nanobot.command.builtin.os.execv"):
            out = await cmd_restart(ctx)
            assert "Restarting" in out.content
            await asyncio.sleep(1.5)

        payload = json.loads((tmp_path / "restart_pending.json").read_text())
        assert payload["reason"] == "slash_command_restart"
        assert payload["channel"] == "telegram"
        assert payload["chat_id"] == "chat-1"
        assert payload["message_thread_id"] == "42"

    @pytest.mark.asyncio
    async def test_restart_intercepted_in_run_loop(self):
        """Verify /restart is handled at the run-loop level, not inside _dispatch."""
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/restart")

        with patch.object(loop, "_dispatch", new_callable=AsyncMock) as mock_dispatch, \
             patch("nanobot.command.builtin.os.execv"):
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(loop.run())
            await asyncio.sleep(0.1)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_not_called()
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "Restarting" in out.content

    @pytest.mark.asyncio
    async def test_status_intercepted_in_run_loop(self):
        """Verify /status is handled at the run-loop level for immediate replies."""
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")

        with patch.object(loop, "_dispatch", new_callable=AsyncMock) as mock_dispatch, \
             patch("nanobot_workspace.observability.usage_tracker.run_checks", return_value=[]), \
             patch("nanobot_workspace.observability.usage_tracker.load_latest_snapshot", return_value=None):
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(loop.run())
            await asyncio.sleep(0.1)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_not_called()
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "nanobot" in out.content.lower() or "Model" in out.content

    @pytest.mark.asyncio
    async def test_run_propagates_external_cancellation(self):
        """External task cancellation should not be swallowed by the inbound wait loop."""
        loop, _bus = _make_loop()

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.1)
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_help_includes_restart(self):
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/help")

        response = await loop._process_message(msg)

        assert response is not None
        assert "/restart" in response.content
        assert "/status" in response.content
        assert response.metadata == {"render_as": "text"}

    @pytest.mark.asyncio
    async def test_status_reports_runtime_info(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}] * 3
        loop.sessions.get_or_create.return_value = session
        loop._start_time = time.time() - 125
        loop._last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        loop.memory_consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(20500, "tiktoken")
        )

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")

        with patch("nanobot_workspace.observability.usage_tracker.run_checks", return_value=[]):
            response = await loop._process_message(msg)

        assert response is not None
        assert "Model: test-model" in response.content
        assert "Tokens: 0 in / 0 out" in response.content
        assert "Context: 20k/64k (31%)" in response.content
        assert "Session: 3 messages" in response.content
        assert "Uptime: 2m 5s" in response.content
        assert response.metadata == {"render_as": "text"}

    @pytest.mark.asyncio
    async def test_run_agent_loop_resets_usage_when_provider_omits_it(self):
        loop, _bus = _make_loop()
        loop.provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(content="first", usage={"prompt_tokens": 9, "completion_tokens": 4}),
            LLMResponse(content="second", usage={}),
        ])

        await loop._run_agent_loop([])
        assert loop._last_usage == {"prompt_tokens": 9, "completion_tokens": 4}

        await loop._run_agent_loop([])
        assert loop._last_usage == {"prompt_tokens": 0, "completion_tokens": 0}

    @pytest.mark.asyncio
    async def test_status_falls_back_to_last_usage_when_context_estimate_missing(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}]
        loop.sessions.get_or_create.return_value = session
        loop._last_usage = {"prompt_tokens": 1200, "completion_tokens": 34}
        loop.memory_consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(0, "none")
        )

        response = await loop._process_message(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")
        )

        assert response is not None
        assert "Tokens: 1200 in / 34 out" in response.content
        assert "Context: 1k/64k (1%)" in response.content

    @pytest.mark.asyncio
    async def test_process_direct_preserves_render_metadata(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        session.get_history.return_value = []
        loop.sessions.get_or_create.return_value = session
        loop.subagents.get_running_count.return_value = 0

        response = await loop.process_direct("/status", session_key="cli:test")

        assert response is not None
        assert response.metadata == {"render_as": "text"}

    @pytest.mark.asyncio
    async def test_process_message_persists_last_active_session(self, tmp_path: Path):
        loop, _bus = _make_loop_with_workspace(tmp_path)
        session = MagicMock()
        loop.sessions.get_or_create.return_value = session

        response = await loop._process_message(
            InboundMessage(
                channel="telegram",
                sender_id="u1",
                chat_id="c1",
                content="/help",
                metadata={"message_thread_id": "77"},
            )
        )

        assert response is not None
        payload = json.loads((tmp_path / "data" / "last_active_session.json").read_text())
        assert payload["channel"] == "telegram"
        assert payload["chat_id"] == "c1"
        assert payload["message_thread_id"] == "77"

    def test_consume_restart_resume_payload_falls_back_to_last_active_session(self, tmp_path: Path):
        from nanobot.cli.commands import _consume_restart_resume_payload

        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "last_active_session.json").write_text(
            json.dumps(
                {
                    "channel": "telegram",
                    "chat_id": "fallback-chat",
                    "message_thread_id": "9",
                    "timestamp": "2026-04-14T17:34:59+00:00",
                }
            )
        )

        payload = _consume_restart_resume_payload(tmp_path)

        assert payload is not None
        assert payload["channel"] == "telegram"
        assert payload["chat_id"] == "fallback-chat"
        assert payload["message_thread_id"] == "9"
        assert "Gateway restart complete" in (payload["resume_prompt"] or "")

    def test_record_cli_last_active_session_persists_restart_fallback(self, tmp_path: Path):
        from nanobot.cli.commands import _record_cli_last_active_session

        _record_cli_last_active_session(tmp_path, "cli", "direct")

        payload = json.loads((tmp_path / "data" / "last_active_session.json").read_text())
        assert payload["channel"] == "cli"
        assert payload["chat_id"] == "direct"
        assert payload["message_thread_id"] is None

    @pytest.mark.asyncio
    async def test_tasks_handles_missing_workspace_dependency(self):
        loop, _bus = _make_loop()

        import builtins

        real_import = builtins.__import__

        def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nanobot_workspace.tasks":
                raise ModuleNotFoundError("No module named 'nanobot_workspace'")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_failing_import):
            response = await loop._process_message(
                InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/tasks")
            )

        assert response is not None
        assert "Error loading tasks:" in response.content
