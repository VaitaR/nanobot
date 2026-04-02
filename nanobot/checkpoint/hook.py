"""Checkpoint hook — flag-based mid-run pause integration for the agent runner."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.checkpoint.snapshot import CheckpointSnapshot, ToolCallSummary
from nanobot.checkpoint.user_action import UserAction

if TYPE_CHECKING:
    from nanobot.checkpoint.broker import CheckpointBroker


class CheckpointHook(AgentHook):
    """Agent hook that flags checkpoint pauses at configurable thresholds.

    ``after_iteration`` is always fast and non-blocking — it records state
    and sets ``_pause_requested`` when the threshold is reached.  The
    actual ``await`` (via :meth:`do_pause`) happens at the **top** of the
    runner loop where blocking is acceptable.
    """

    FINALIZE_PROMPT = (
        "The task is being wrapped up. Please provide a concise summary "
        "of what you have accomplished and any remaining work."
    )

    def __init__(
        self,
        broker: CheckpointBroker,
        task_id: str,
        label: str,
        max_iterations: int,
        threshold: int,
        cooldown: int = 5,
        max_checkpoints: int = 3,
        origin: dict[str, Any] | None = None,
        on_before_execute_tools: Callable[..., Any] | None = None,
    ) -> None:
        self._broker = broker
        self._task_id = task_id
        self._label = label
        self._base_max = max_iterations
        self._threshold = threshold
        self._cooldown = cooldown
        self._max_checkpoints = max_checkpoints
        self._origin = origin or {}
        self._on_before_execute_tools = on_before_execute_tools

        # Bounded state buffers
        self._tool_events: deque[ToolCallSummary] = deque(maxlen=100)
        self._files_changed: set[str] = set()
        self._llm_outputs: deque[str] = deque(maxlen=5)

        # Iteration tracking
        self._iteration: int = 0
        self._start_time: float = time.monotonic()
        self._error_count: int = 0

        # Checkpoint scheduling
        self._next_checkpoint_at: int = threshold
        self._pause_requested: bool = False
        self._checkpoints_used: int = 0

        # Resolution flags (set by apply_pause_result)
        self._finalize_requested: bool = False
        self._stop_requested: bool = False
        self._extra_iterations: int = 0

    # -- AgentHook overrides ------------------------------------------------

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """Delegate to the optional callback (e.g. subagent idle-time tracking)."""
        if self._on_before_execute_tools is not None:
            await self._on_before_execute_tools(context)

    async def after_iteration(self, ctx: AgentHookContext) -> None:
        """Fast, non-blocking.  Records state and may set the pause flag."""
        self._iteration += 1
        self._record_state(ctx)
        if (
            not self._pause_requested
            and self._iteration >= self._next_checkpoint_at
            and self._checkpoints_used < self._max_checkpoints
        ):
            self._pause_requested = True
            logger.debug(
                "CheckpointHook: pause requested at iteration {} "
                "(threshold={}, checkpoints_used={})",
                self._iteration,
                self._next_checkpoint_at,
                self._checkpoints_used,
            )

    # -- pause lifecycle -----------------------------------------------------

    async def do_pause(self) -> tuple[UserAction, int]:
        """Await the broker's review and apply the result.

        Called by the runner at the **top** of the loop when
        :attr:`pause_requested` is ``True``.
        """
        snapshot = self.build_snapshot()
        action, extra = await self._broker.pause(self._task_id, snapshot)
        self.apply_pause_result(action, extra)
        return action, extra

    def apply_pause_result(self, action: UserAction, extra: int = 0) -> None:
        """Apply the review decision and update scheduling."""
        self._pause_requested = False
        if action == UserAction.CONTINUE:
            self._extra_iterations += extra
            self._checkpoints_used += 1
            self._next_checkpoint_at = self._iteration + extra + self._cooldown
        elif action == UserAction.DONE:
            self._finalize_requested = True
            self._checkpoints_used += 1
        elif action == UserAction.STOP:
            self._stop_requested = True
            self._checkpoints_used += 1
        # DETAILS is a non-terminal sub-flow; broker handles it internally.

    def build_snapshot(self) -> CheckpointSnapshot:
        """Build a frozen snapshot for policy evaluation."""
        # Run loop detector over accumulated tool events
        from nanobot.checkpoint.loop_detector import LoopDetector

        detector = LoopDetector(window=8)
        for tc in self._tool_events:
            detector.observe(tc.tool_name, tc.detail, tc.iteration)
        loop = detector.detect()

        return CheckpointSnapshot(
            total_iterations=self._iteration,
            max_iterations=self.effective_max,
            tool_calls=tuple(self._tool_events)[-20:],
            files_touched=frozenset(self._files_changed),
            last_llm_outputs=tuple(self._llm_outputs),
            error_count=self._error_count,
            loop_detected=(loop is not None),
            stuck_score=0.0,
        )

    # -- properties ----------------------------------------------------------

    @property
    def pause_requested(self) -> bool:
        return self._pause_requested

    @property
    def finalize_requested(self) -> bool:
        return self._finalize_requested

    @property
    def should_stop(self) -> bool:
        return self._stop_requested

    @property
    def effective_max(self) -> int:
        return self._base_max + self._extra_iterations

    @property
    def finalize_prompt(self) -> str:
        return self.FINALIZE_PROMPT

    # -- internal helpers ----------------------------------------------------

    def _record_state(self, ctx: AgentHookContext) -> None:
        """Extract tool events, file changes, LLM outputs, and errors from the context."""
        # Record tool events
        for event in ctx.tool_events:
            args = event.get("arguments", {})
            file_path = args.get("path") or args.get("file_path")
            if file_path:
                self._files_changed.add(file_path)
            self._tool_events.append(ToolCallSummary(
                tool_name=event["name"],
                detail=event["detail"],
                file_path=file_path,
                iteration=self._iteration,
            ))
            if event.get("status") == "error":
                self._error_count += 1

        # Record LLM output
        if ctx.final_content is not None:
            self._llm_outputs.append(ctx.final_content)
