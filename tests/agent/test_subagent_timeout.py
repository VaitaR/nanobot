"""Tests for adaptive subagent idle timeout behavior."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

WORKSPACE_SRC = Path("/root/.nanobot/workspace/src")
if str(WORKSPACE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_SRC))


@pytest.fixture
def subagent_env(tmp_path):
    """Create a SubagentManager with a mock runner."""
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    manager = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
    manager.runner = MagicMock(spec=["run"])
    manager._announce_result = AsyncMock()
    return manager


def test_watchdog_snapshot_flags_real_stall():
    """Few iterations and no progress should look stuck."""
    from nanobot.agent.subagent import _SubagentWatchdogState

    state = _SubagentWatchdogState(last_activity=0.0, last_progress=0.0)

    snapshot = state.snapshot(now=700.0, idle_timeout=600)

    assert snapshot.idle_for == 700.0
    assert snapshot.no_progress_for == 700.0
    assert snapshot.iterations_last_10m == 0
    assert snapshot.effective_timeout == 600.0
    assert snapshot.extension_reasons == ()


def test_watchdog_snapshot_extends_for_busy_long_running_tools():
    """Frequent exec/spawn work should extend the timeout."""
    from nanobot.agent.subagent import _SubagentWatchdogState

    state = _SubagentWatchdogState(last_activity=0.0, last_progress=0.0)
    exec_call = SimpleNamespace(name="exec")

    for ts in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0):
        state.record_tool_calls([exec_call], ts)
        state.record_iteration_result(
            [{"name": "exec", "status": "ok", "detail": "chunk", "arguments": {}}],
            ts,
        )

    snapshot = state.snapshot(now=250.0, idle_timeout=600)

    assert snapshot.iterations_last_5m == 6
    assert snapshot.effective_timeout == 1500.0
    assert set(snapshot.extension_reasons) == {"high_iteration_rate", "long_running_tool"}


def test_tool_event_counts_as_progress_ignores_read_only_tools():
    """Read-only filesystem calls should not reset progress."""
    from nanobot.agent.subagent import _tool_event_counts_as_progress

    assert not _tool_event_counts_as_progress(
        {"name": "read_file", "status": "ok", "detail": "body", "arguments": {"path": "x"}}
    )
    assert not _tool_event_counts_as_progress(
        {"name": "list_dir", "status": "ok", "detail": "files", "arguments": {"path": "."}}
    )
    assert _tool_event_counts_as_progress(
        {"name": "edit_file", "status": "ok", "detail": "updated", "arguments": {"path": "x.py"}}
    )


def test_resolve_subagent_watchdog_settings_keeps_legacy_defaults_without_config():
    """No explicit config should preserve old runtime defaults."""
    from nanobot.agent.subagent import SubagentManager
    from nanobot.config.schema import Config

    with patch("nanobot.config.loader.load_config", return_value=Config()):
        hard_cap, idle_timeout, poll_interval = SubagentManager._resolve_subagent_watchdog_settings(
            1800,
            600,
        )

    assert (hard_cap, idle_timeout, poll_interval) == (1800, 600, 30)


def test_resolve_subagent_watchdog_settings_applies_config_override():
    """Explicit subagent config should override legacy defaults."""
    from nanobot.agent.subagent import SubagentManager
    from nanobot.config.schema import Config

    cfg = Config.model_validate(
        {
            "agents": {
                "subagent": {
                    "hard_cap": 2400,
                    "idle_timeout": 1200,
                    "watchdog_poll_interval": 15,
                }
            }
        }
    )

    with patch("nanobot.config.loader.load_config", return_value=cfg):
        hard_cap, idle_timeout, poll_interval = SubagentManager._resolve_subagent_watchdog_settings(
            1800,
            600,
        )

    assert (hard_cap, idle_timeout, poll_interval) == (2400, 1200, 15)


@pytest.mark.asyncio
async def test_run_subagent_times_out_only_when_really_stuck(subagent_env):
    """A hung runner with no iterations should be cancelled by the watchdog."""
    manager = subagent_env

    async def hanging_run(_spec):
        await asyncio.sleep(60)
        return MagicMock(stop_reason="completed", final_content="done", tool_events=[], error=None)

    manager.runner.run = AsyncMock(side_effect=hanging_run)

    with patch.object(manager, "_resolve_subagent_watchdog_settings", return_value=(300, 1, 0.05)):
        await manager._run_subagent(
            task_id="STALL01",
            task="do something",
            label="stall",
            origin={"channel": "cli", "chat_id": "direct", "message_thread_id": None},
            idle_timeout=1,
            hard_cap=300,
        )

    envelope = manager._announce_result.await_args.args[3]
    assert envelope.stop_reason == "timeout"
    assert "appeared stuck" in envelope.summary.lower()


@pytest.mark.asyncio
async def test_run_subagent_allows_busy_iteration_without_progress(subagent_env):
    """High iteration count should keep a busy subagent alive even without file progress."""
    from nanobot.agent.hook import AgentHookContext
    from nanobot.providers.base import ToolCallRequest

    manager = subagent_env

    async def busy_run(spec):
        for idx in range(6):
            context = AgentHookContext(
                iteration=idx,
                messages=[],
                tool_calls=[ToolCallRequest(id=f"call_{idx}", name="list_dir", arguments={"path": "."})],
                tool_events=[{"name": "list_dir", "status": "ok", "detail": "files", "arguments": {"path": "."}}],
            )
            await spec.hook.before_execute_tools(context)
            await spec.hook.after_iteration(context)
        await asyncio.sleep(0.7)
        return MagicMock(
            stop_reason="completed",
            final_content="done",
            tool_events=context.tool_events,
            error=None,
            tools_used=["list_dir"],
        )

    manager.runner.run = AsyncMock(side_effect=busy_run)

    with patch.object(manager, "_resolve_subagent_watchdog_settings", return_value=(3, 0.4, 0.05)):
        await manager._run_subagent(
            task_id="BUSY01",
            task="inspect a lot of files",
            label="busy",
            origin={"channel": "cli", "chat_id": "direct", "message_thread_id": None},
            idle_timeout=1,
            hard_cap=3,
        )

    envelope = manager._announce_result.await_args.args[3]
    assert envelope.stop_reason == "completed"
    assert envelope.status == "ok"


@pytest.mark.asyncio
async def test_watchdog_logs_extension_reason(subagent_env):
    """Watchdog should log when it extends timeout for long-running work."""
    from nanobot.agent.hook import AgentHookContext
    from nanobot.providers.base import ToolCallRequest

    manager = subagent_env
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")

    async def exec_run(spec):
        for idx in range(6):
            context = AgentHookContext(
                iteration=idx,
                messages=[],
                tool_calls=[ToolCallRequest(id=f"call_{idx}", name="exec", arguments={"command": "sleep 1"})],
                tool_events=[{"name": "exec", "status": "ok", "detail": "chunk", "arguments": {}}],
            )
            await spec.hook.before_execute_tools(context)
            await spec.hook.after_iteration(context)
        await asyncio.sleep(0.2)
        return MagicMock(
            stop_reason="completed",
            final_content="done",
            tool_events=context.tool_events,
            error=None,
            tools_used=["exec"],
        )

    manager.runner.run = AsyncMock(side_effect=exec_run)

    try:
        with patch.object(manager, "_resolve_subagent_watchdog_settings", return_value=(3, 1, 0.05)):
            await manager._run_subagent(
                task_id="LOGS01",
                task="run command",
                label="logs",
                origin={"channel": "cli", "chat_id": "direct", "message_thread_id": None},
                idle_timeout=1,
                hard_cap=3,
            )
    finally:
        logger.remove(sink_id)

    assert any("watchdog extending timeout" in message for message in messages)
