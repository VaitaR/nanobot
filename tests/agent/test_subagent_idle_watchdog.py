"""Tests for the concurrent idle watchdog in SubagentManager._run_subagent()."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def subagent_env(tmp_path):
    """Set up a SubagentManager with mock provider, and return (manager, bus, mock_runner)."""
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    manager = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
    mock_runner = MagicMock(spec=["run"])
    manager.runner = mock_runner
    return manager, bus, mock_runner


# ── Direct watchdog logic tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_idle_watchdog_cancels_task_on_timeout():
    """Watchdog coroutine should cancel run_task when idle exceeds threshold."""
    idle_timeout = 1
    last_activity = time.monotonic() - 100  # simulate 100s idle

    async def never_completes():
        await asyncio.sleep(60)

    run_task = asyncio.create_task(never_completes())

    async def idle_watchdog():
        while not run_task.done():
            await asyncio.sleep(0.05)  # poll frequently for test speed
            if not run_task.done() and (time.monotonic() - last_activity) > idle_timeout:
                run_task.cancel()
                return

    watchdog = asyncio.create_task(idle_watchdog())

    done, _ = await asyncio.wait({run_task, watchdog}, timeout=5)

    assert run_task.done()
    assert run_task.cancelled()
    watchdog.cancel()
    try:
        await watchdog
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_idle_watchdog_does_not_cancel_active_task():
    """Watchdog should NOT cancel if last_activity is recent."""
    idle_timeout = 1
    last_activity = time.monotonic()  # just now

    async def quick_task():
        await asyncio.sleep(0.1)
        return "done"

    run_task = asyncio.create_task(quick_task())

    async def idle_watchdog():
        while not run_task.done():
            await asyncio.sleep(0.05)
            if not run_task.done() and (time.monotonic() - last_activity) > idle_timeout:
                run_task.cancel()
                return

    watchdog = asyncio.create_task(idle_watchdog())

    done, _ = await asyncio.wait({run_task, watchdog}, timeout=5)

    assert run_task in done
    assert run_task.result() == "done"
    assert not run_task.cancelled()
    watchdog.cancel()
    try:
        await watchdog
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_wall_clock_timeout_via_asyncio_wait():
    """asyncio.wait timeout parameter should enforce wall-clock cap."""
    hard_cap = 1

    async def slow_task():
        await asyncio.sleep(60)
        return "should not reach"

    run_task = asyncio.create_task(slow_task())

    done, pending = await asyncio.wait({run_task}, timeout=hard_cap)

    assert run_task in pending  # did NOT complete in time
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


# ── Integration tests via _run_subagent ─────────────────────────────────
# _run_subagent catches TimeoutError internally and announces the result.
# We verify correct behavior by inspecting the announced message.


@pytest.mark.asyncio
async def test_run_subagent_idle_timeout(subagent_env):
    """_run_subagent should detect idle timeout and announce it."""
    manager, bus, mock_runner = subagent_env

    # Runner hangs forever — simulates LLM stuck
    async def hanging_run(_spec):
        await asyncio.sleep(60)
        return MagicMock(stop_reason="completed", final_content="done",
                         tool_events=[], error=None)

    mock_runner.run = AsyncMock(side_effect=hanging_run)

    # Track what gets announced
    announced: list[dict] = []
    original_announce = manager._announce_result

    async def capture_announce(task_id, label, task, envelope, origin):
        announced.append({"envelope": envelope})
        await original_announce(task_id, label, task, envelope, origin)

    with patch.object(manager, "_announce_result", side_effect=capture_announce):
        with patch("nanobot.agent.subagent._WATCHDOG_POLL_INTERVAL", 0.05):
            await manager._run_subagent(
                task_id="TEST01",
                task="do something",
                label="test",
                origin={"channel": "cli", "chat_id": "direct", "message_thread_id": None},
                idle_timeout=1,
                hard_cap=300,
            )

    assert len(announced) == 1
    envelope = announced[0]["envelope"]
    assert envelope.stop_reason == "timeout"
    assert "idle" in envelope.summary.lower()


@pytest.mark.asyncio
async def test_run_subagent_wall_clock_timeout(subagent_env):
    """_run_subagent should detect wall-clock timeout and announce it."""
    manager, bus, mock_runner = subagent_env

    # Runner hangs — but we use real time with short hard_cap
    async def hanging_run(_spec):
        await asyncio.sleep(60)
        return MagicMock(stop_reason="completed", final_content="done",
                         tool_events=[], error=None)

    mock_runner.run = AsyncMock(side_effect=hanging_run)

    announced: list[dict] = []
    original_announce = manager._announce_result

    async def capture_announce(task_id, label, task, envelope, origin):
        announced.append({"envelope": envelope})
        await original_announce(task_id, label, task, envelope, origin)

    # Use real time: idle_timeout=300 (won't fire), hard_cap=1s (will fire)
    with patch.object(manager, "_announce_result", side_effect=capture_announce):
        with patch("nanobot.agent.subagent._WATCHDOG_POLL_INTERVAL", 0.05):
            await manager._run_subagent(
                task_id="TEST02",
                task="do something",
                label="test",
                origin={"channel": "cli", "chat_id": "direct", "message_thread_id": None},
                idle_timeout=300,
                hard_cap=1,
            )

    assert len(announced) == 1
    envelope = announced[0]["envelope"]
    assert envelope.stop_reason == "timeout"
    assert "timed out" in envelope.summary.lower()


@pytest.mark.asyncio
async def test_run_subagent_normal_completion(subagent_env):
    """_run_subagent should complete normally when runner returns quickly."""
    manager, bus, mock_runner = subagent_env

    mock_result = MagicMock(
        stop_reason="completed",
        final_content="Task completed successfully.",
        tool_events=[],
        error=None,
    )
    mock_runner.run = AsyncMock(return_value=mock_result)

    announced: list[dict] = []
    original_announce = manager._announce_result

    async def capture_announce(task_id, label, task, envelope, origin):
        announced.append({"envelope": envelope})
        await original_announce(task_id, label, task, envelope, origin)

    with patch.object(manager, "_announce_result", side_effect=capture_announce):
        with patch("nanobot.agent.subagent.time") as mock_time:
            now = 1000.0
            mock_time.monotonic.return_value = now
            mock_time.time.return_value = now

            await manager._run_subagent(
                task_id="TEST03",
                task="do something",
                label="test",
                origin={"channel": "cli", "chat_id": "direct", "message_thread_id": None},
                idle_timeout=300,
                hard_cap=300,
            )

    assert len(announced) == 1
    envelope = announced[0]["envelope"]
    assert envelope.status == "ok"
    mock_runner.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_subagent_watchdog_cleaned_up_on_success(subagent_env):
    """Watchdog task should be properly cancelled after successful completion."""
    manager, bus, mock_runner = subagent_env

    mock_result = MagicMock(
        stop_reason="completed",
        final_content="Done.",
        tool_events=[],
        error=None,
    )
    mock_runner.run = AsyncMock(return_value=mock_result)

    with patch("nanobot.agent.subagent.time") as mock_time:
        now = 1000.0
        mock_time.monotonic.return_value = now
        mock_time.time.return_value = now

        await manager._run_subagent(
            task_id="TEST04",
            task="quick task",
            label="test",
            origin={"channel": "cli", "chat_id": "direct", "message_thread_id": None},
            idle_timeout=300,
            hard_cap=300,
        )

    # If we get here without hanging, the watchdog was cleaned up properly
    mock_runner.run.assert_called_once()
