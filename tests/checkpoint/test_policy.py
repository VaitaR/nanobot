"""Tests for the ReviewPolicy."""

from __future__ import annotations

from nanobot.checkpoint.policy import CheckpointAction, ReviewPolicy
from nanobot.checkpoint.snapshot import CheckpointSnapshot, ToolCallSummary


def _make_summary(
    tool_name: str = "read_file",
    detail: str = "path=/tmp/a.txt",
    file_path: str | None = "/tmp/a.txt",
    iteration: int = 0,
) -> ToolCallSummary:
    return ToolCallSummary(tool_name=tool_name, detail=detail, file_path=file_path, iteration=iteration)


class TestAutoContinueOnProgress:
    """Healthy agent should get AUTO_CONTINUE."""

    def test_auto_continue_on_progress(self) -> None:
        policy = ReviewPolicy()
        tools = [
            _make_summary("read_file", "path=/tmp/a.txt", "/tmp/a.txt", i)
            for i in range(6)
        ]
        # All different tools
        tools[1] = _make_summary("edit_file", "path=/tmp/a.txt", "/tmp/a.txt", 1)
        tools[2] = _make_summary("exec", "run tests", None, 2)
        tools[3] = _make_summary("read_file", "path=/tmp/b.txt", "/tmp/b.txt", 3)
        tools[4] = _make_summary("write_file", "path=/tmp/c.py", "/tmp/c.py", 4)
        tools[5] = _make_summary("exec", "pytest -v", None, 5)

        snapshot = CheckpointSnapshot(
            total_iterations=6,
            max_iterations=10,
            tool_calls=tuple(tools),
            files_touched=frozenset({"/tmp/a.txt", "/tmp/b.txt", "/tmp/c.py"}),
            last_llm_outputs=("step one", "step two", "step three"),
            error_count=0,
            loop_detected=False,
            stuck_score=0.0,
        )
        decision = policy.evaluate(snapshot)
        assert decision.action == CheckpointAction.AUTO_CONTINUE
        assert decision.confidence > 0.5


class TestEscalateOnLoop:
    """Loop detection should always escalate."""

    def test_escalate_on_loop(self) -> None:
        policy = ReviewPolicy()
        # Create exact repeat pattern
        tools = [
            _make_summary("read_file", "path=/tmp/a.txt", "/tmp/a.txt", i)
            for i in range(8)
        ]
        snapshot = CheckpointSnapshot(
            total_iterations=8,
            max_iterations=10,
            tool_calls=tuple(tools),
            files_touched=frozenset({"/tmp/a.txt"}),
            last_llm_outputs=("checking file", "checking file", "checking file"),
            error_count=0,
            loop_detected=True,
            stuck_score=0.0,
        )
        decision = policy.evaluate(snapshot)
        assert decision.action == CheckpointAction.ESCALATE
        assert "loop" in decision.reason.lower()


class TestEscalateOnStuck:
    """Stuck signals should escalate."""

    def test_escalate_on_stuck(self) -> None:
        policy = ReviewPolicy()
        # Varied tool calls (not a loop) but no files + repetitive LLM outputs
        tools = [
            _make_summary("read_file", f"checking config part {i}", None, i)
            for i in range(8)
        ]
        snapshot = CheckpointSnapshot(
            total_iterations=8,
            max_iterations=10,
            tool_calls=tuple(tools),
            files_touched=frozenset(),  # no files touched
            last_llm_outputs=(
                "The configuration looks correct.",
                "The configuration looks correct.",
                "The configuration looks correct.",
                "The configuration looks correct.",
                "The configuration looks correct.",
            ),
            error_count=0,
            loop_detected=False,
            stuck_score=0.0,
        )
        decision = policy.evaluate(snapshot)
        assert decision.action == CheckpointAction.ESCALATE
        assert "stuck" in decision.reason.lower()

    def test_no_false_positive_read_only(self) -> None:
        """read_only_hint should suppress the no-files-touched signal."""
        policy = ReviewPolicy(read_only_hint=True)
        tools = [
            _make_summary("web_search", "query about python", None, i)
            for i in range(8)
        ]
        # Same tool but varied enough to not trigger loop/cycle
        tools = [
            _make_summary("web_search", f"query about topic {i}", None, i)
            for i in range(8)
        ]
        snapshot = CheckpointSnapshot(
            total_iterations=8,
            max_iterations=10,
            tool_calls=tuple(tools),
            files_touched=frozenset(),
            last_llm_outputs=("result one", "result two", "result three"),
            error_count=0,
            loop_detected=False,
            stuck_score=0.0,
        )
        decision = policy.evaluate(snapshot)
        assert decision.action == CheckpointAction.AUTO_CONTINUE
