"""Tests for the CheckpointHook."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.hook import AgentHookContext
from nanobot.checkpoint.broker import CheckpointBroker
from nanobot.checkpoint.hook import CheckpointHook
from nanobot.checkpoint.user_action import UserAction


def _make_context(
    iteration: int = 0,
    tool_events: list[dict[str, str]] | None = None,
    final_content: str | None = None,
) -> AgentHookContext:
    return AgentHookContext(
        iteration=iteration,
        messages=[],
        tool_events=tool_events or [],
        final_content=final_content,
    )


class TestPauseRequestedFlag:
    """pause_requested should be set at the configured threshold."""

    async def test_flag_not_set_below_threshold(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t1",
            label="test",
            max_iterations=10,
            threshold=8,
        )

        for i in range(7):
            await hook.after_iteration(_make_context(iteration=i))

        assert hook.pause_requested is False

    async def test_flag_set_at_threshold(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t2",
            label="test",
            max_iterations=10,
            threshold=8,
        )

        for i in range(8):
            await hook.after_iteration(_make_context(iteration=i))

        assert hook.pause_requested is True

    async def test_flag_set_only_once(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t3",
            label="test",
            max_iterations=20,
            threshold=5,
        )

        for i in range(10):
            await hook.after_iteration(_make_context(iteration=i))

        # Flag was set at iteration 5, stays set
        assert hook.pause_requested is True
        # _iteration should be 10
        assert hook._iteration == 10


class TestBuildSnapshot:
    """build_snapshot should produce an accurate CheckpointSnapshot."""

    async def test_snapshot_reflects_recorded_state(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t4",
            label="test",
            max_iterations=10,
            threshold=8,
        )

        # Simulate 6 iterations with tool events and file changes
        for i in range(6):
            ctx = _make_context(
                iteration=i,
                tool_events=[
                    {
                        "name": "read_file",
                        "status": "ok",
                        "detail": f"content of file {i}",
                        "arguments": {"path": f"/tmp/file{i}.txt"},
                    },
                ],
                final_content=f"Step {i} complete",
            )
            await hook.after_iteration(ctx)

        # Add an error
        ctx = _make_context(
            iteration=6,
            tool_events=[
                {
                    "name": "exec",
                    "status": "error",
                    "detail": "command failed",
                    "arguments": {"command": "make"},
                },
            ],
        )
        await hook.after_iteration(ctx)

        snapshot = hook.build_snapshot()

        assert snapshot.total_iterations == 7
        assert snapshot.max_iterations == 10
        assert snapshot.error_count == 1
        assert len(snapshot.files_touched) == 6
        assert "/tmp/file0.txt" in snapshot.files_touched
        assert len(snapshot.tool_calls) == 7
        assert len(snapshot.last_llm_outputs) == 5  # bounded by deque(maxlen=5)

    async def test_snapshot_bounded_tool_calls(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t5",
            label="test",
            max_iterations=30,
            threshold=25,
        )

        for i in range(25):
            ctx = _make_context(
                iteration=i,
                tool_events=[
                    {"name": "exec", "status": "ok", "detail": f"cmd {i}", "arguments": {}},
                ],
            )
            await hook.after_iteration(ctx)

        snapshot = hook.build_snapshot()
        # tool_calls is bounded to last 20 by deque(maxlen=100) and [-20:] in build_snapshot
        assert len(snapshot.tool_calls) <= 20


class TestApplyPauseResult:
    """apply_pause_result should update flags and scheduling correctly."""

    def test_continue_advances_checkpoint_and_increases_max(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t6",
            label="test",
            max_iterations=10,
            threshold=8,
            cooldown=5,
        )
        # Simulate reaching iteration 8
        hook._iteration = 8
        hook._pause_requested = True

        hook.apply_pause_result(UserAction.CONTINUE, 5)

        assert hook.pause_requested is False
        assert hook.effective_max == 15  # 10 + 5
        assert hook._checkpoints_used == 1
        assert hook._next_checkpoint_at == 8 + 5 + 5  # 18
        assert hook.finalize_requested is False
        assert hook.should_stop is False

    def test_done_sets_finalize_flag(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t7",
            label="test",
            max_iterations=10,
            threshold=8,
        )
        hook._iteration = 8
        hook._pause_requested = True

        hook.apply_pause_result(UserAction.DONE)

        assert hook.finalize_requested is True
        assert hook.should_stop is False
        assert hook._checkpoints_used == 1

    def test_stop_sets_stop_flag(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t8",
            label="test",
            max_iterations=10,
            threshold=8,
        )
        hook._iteration = 8
        hook._pause_requested = True

        hook.apply_pause_result(UserAction.STOP)

        assert hook.should_stop is True
        assert hook.finalize_requested is False
        assert hook._checkpoints_used == 1


class TestMultiCheckpointCooldown:
    """Multiple checkpoints with cooldown should fire at the right intervals."""

    async def test_second_checkpoint_fires_after_cooldown(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t9",
            label="test",
            max_iterations=20,
            threshold=8,
            cooldown=3,
            max_checkpoints=3,
        )

        # Run to threshold (8 iterations)
        for i in range(8):
            await hook.after_iteration(_make_context(iteration=i))
        assert hook.pause_requested is True

        # Resolve with CONTINUE +5 → next checkpoint at 8 + 5 + 3 = 16
        hook.apply_pause_result(UserAction.CONTINUE, 5)

        # Iterations 9-15: no pause
        for i in range(8, 15):
            assert hook.pause_requested is False
            await hook.after_iteration(_make_context(iteration=i))

        # Iteration 16: pause requested again
        await hook.after_iteration(_make_context(iteration=15))
        assert hook.pause_requested is True

    async def test_max_checkpoints_limit(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t10",
            label="test",
            max_iterations=40,
            threshold=5,
            cooldown=2,
            max_checkpoints=2,
        )

        # Checkpoint 1 at iteration 5
        for i in range(5):
            await hook.after_iteration(_make_context(iteration=i))
        assert hook.pause_requested is True
        hook.apply_pause_result(UserAction.CONTINUE, 5)

        # Checkpoint 2 at iteration 5 + 5 + 2 = 12
        for i in range(5, 12):
            await hook.after_iteration(_make_context(iteration=i))
        assert hook.pause_requested is True
        hook.apply_pause_result(UserAction.CONTINUE, 5)

        # No more checkpoints — run past the next threshold
        for i in range(12, 30):
            await hook.after_iteration(_make_context(iteration=i))
        assert hook.pause_requested is False


class TestEffectiveMax:
    """effective_max should reflect base + extra iterations."""

    def test_initial_effective_max(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t11",
            label="test",
            max_iterations=10,
            threshold=8,
        )
        assert hook.effective_max == 10

    def test_effective_max_after_continue(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t12",
            label="test",
            max_iterations=10,
            threshold=8,
        )
        hook._iteration = 8
        hook.apply_pause_result(UserAction.CONTINUE, 5)
        assert hook.effective_max == 15

    def test_effective_max_after_multiple_continues(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t13",
            label="test",
            max_iterations=10,
            threshold=8,
        )
        hook._iteration = 8
        hook.apply_pause_result(UserAction.CONTINUE, 5)
        hook._iteration = 18
        hook.apply_pause_result(UserAction.CONTINUE, 3)
        assert hook.effective_max == 18


class TestOnBeforeExecuteTools:
    """The optional before_execute_tools callback should be invoked."""

    async def test_callback_invoked(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        callback = AsyncMock()
        hook = CheckpointHook(
            broker=broker,
            task_id="t14",
            label="test",
            max_iterations=10,
            threshold=8,
            on_before_execute_tools=callback,
        )

        ctx = _make_context()
        await hook.before_execute_tools(ctx)

        callback.assert_awaited_once_with(ctx)

    async def test_no_callback_no_error(self) -> None:
        broker = MagicMock(spec=CheckpointBroker)
        hook = CheckpointHook(
            broker=broker,
            task_id="t15",
            label="test",
            max_iterations=10,
            threshold=8,
        )

        ctx = _make_context()
        # Should not raise
        await hook.before_execute_tools(ctx)
