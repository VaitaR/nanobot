"""Tests for the CheckpointBroker."""

from __future__ import annotations

import asyncio

from nanobot.bus.queue import MessageBus
from nanobot.checkpoint.broker import CheckpointBroker
from nanobot.checkpoint.policy import ReviewPolicy
from nanobot.checkpoint.snapshot import CheckpointSnapshot, ToolCallSummary
from nanobot.checkpoint.user_action import UserAction


def _make_snapshot(
    total_iterations: int = 8,
    max_iterations: int = 10,
    tool_calls: tuple[ToolCallSummary, ...] = (),
    error_count: int = 0,
    loop_detected: bool = False,
    **kwargs,
) -> CheckpointSnapshot:
    return CheckpointSnapshot(
        total_iterations=total_iterations,
        max_iterations=max_iterations,
        tool_calls=tool_calls,
        files_touched=frozenset(kwargs.get("files_touched", set())),
        last_llm_outputs=tuple(kwargs.get("last_llm_outputs", [])),
        error_count=error_count,
        loop_detected=loop_detected,
        stuck_score=0.0,
    )


class TestAutoContinue:
    """AUTO_CONTINUE path should return immediately without escalation."""

    async def test_auto_continue_returns_immediately(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(checkpoint_threshold_pct=0.8)
        broker = CheckpointBroker(policy=policy, bus=bus, origin={"channel": "cli", "chat_id": "c1"})
        snapshot = _make_snapshot(total_iterations=8, max_iterations=10)

        action, extra = await broker.pause("task-1", snapshot)

        assert action == UserAction.CONTINUE
        assert extra == 2  # remaining: 10 - 8
        # No escalation message should have been sent
        assert bus.outbound_size == 0

    async def test_auto_continue_with_zero_remaining(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy()
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})
        snapshot = _make_snapshot(total_iterations=10, max_iterations=10)

        action, extra = await broker.pause("task-2", snapshot)

        assert action == UserAction.CONTINUE
        assert extra == 0


class TestEscalate:
    """ESCALATE path should send alert and wait for resolve or timeout."""

    async def test_escalate_sends_alert_and_waits(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=60)
        # Use stuck-triggering snapshot: no files, repetitive LLM
        snapshot = _make_snapshot(
            total_iterations=8,
            max_iterations=10,
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
            tool_calls=tuple(
                ToolCallSummary(tool_name="read_file", detail=f"checking config part {i}", iteration=i)
                for i in range(8)
            ),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={"channel": "cli", "chat_id": "c1"})

        # Start pause in background
        pause_task = asyncio.create_task(broker.pause("task-3", snapshot))

        # Give the coroutine a chance to start and send the escalation
        await asyncio.sleep(0.05)

        # Verify escalation message was sent
        assert bus.outbound_size >= 1
        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
        assert "review needed" in msg.content.lower()

        # Resolve the checkpoint
        resolved = broker.resolve_checkpoint("task-3", UserAction.CONTINUE, 5)
        assert resolved is True

        action, extra = await asyncio.wait_for(pause_task, timeout=1)
        assert action == UserAction.CONTINUE
        assert extra == 5

    async def test_escalate_timeout_auto_continues(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=0.1)  # 100ms timeout
        snapshot = _make_snapshot(
            total_iterations=8,
            max_iterations=10,
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})

        action, extra = await broker.pause("task-4", snapshot)

        assert action == UserAction.CONTINUE
        # Should have sent escalation + timeout notification
        assert bus.outbound_size >= 2

    async def test_resolve_returns_false_for_unknown_task(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy()
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})

        result = broker.resolve_checkpoint("nonexistent", UserAction.CONTINUE)
        assert result is False

    async def test_resolve_returns_false_for_already_set_event(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=60)
        snapshot = _make_snapshot(
            total_iterations=8,
            max_iterations=10,
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})

        pause_task = asyncio.create_task(broker.pause("task-5", snapshot))
        await asyncio.sleep(0.05)

        # First resolve succeeds
        assert broker.resolve_checkpoint("task-5", UserAction.CONTINUE, 3) is True
        await asyncio.wait_for(pause_task, timeout=1)

        # Second resolve should fail (event already consumed)
        assert broker.resolve_checkpoint("task-5", UserAction.CONTINUE) is False


class TestStopAction:
    """STOP resolution should propagate through the broker."""

    async def test_stop_action_returned(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=60)
        snapshot = _make_snapshot(
            total_iterations=8,
            max_iterations=10,
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})

        pause_task = asyncio.create_task(broker.pause("task-6", snapshot))
        await asyncio.sleep(0.05)
        await bus.consume_outbound()  # drain escalation alert

        broker.resolve_checkpoint("task-6", UserAction.STOP)
        action, extra = await asyncio.wait_for(pause_task, timeout=1)

        assert action == UserAction.STOP


class TestDoneAction:
    """DONE resolution should propagate through the broker."""

    async def test_done_action_returned(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=60)
        snapshot = _make_snapshot(
            total_iterations=8,
            max_iterations=10,
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})

        pause_task = asyncio.create_task(broker.pause("task-7", snapshot))
        await asyncio.sleep(0.05)
        await bus.consume_outbound()  # drain escalation alert

        broker.resolve_checkpoint("task-7", UserAction.DONE)
        action, extra = await asyncio.wait_for(pause_task, timeout=1)

        assert action == UserAction.DONE


class TestCleanup:
    """Pending events and results are cleaned up after pause completes."""

    async def test_cleanup_after_auto_continue(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy()
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})
        snapshot = _make_snapshot()

        await broker.pause("task-8", snapshot)

        assert "task-8" not in broker._pending_events
        assert "task-8" not in broker._pending_results

    async def test_cleanup_after_resolve(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=60)
        snapshot = _make_snapshot(
            total_iterations=8,
            max_iterations=10,
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})

        pause_task = asyncio.create_task(broker.pause("task-9", snapshot))
        await asyncio.sleep(0.05)
        await bus.consume_outbound()

        broker.resolve_checkpoint("task-9", UserAction.STOP)
        await asyncio.wait_for(pause_task, timeout=1)

        assert "task-9" not in broker._pending_events
        assert "task-9" not in broker._pending_results
