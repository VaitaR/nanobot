"""Checkpoint broker — orchestrates pause → snapshot → review → resume."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.checkpoint.policy import CheckpointAction
from nanobot.checkpoint.user_action import UserAction

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus
    from nanobot.checkpoint.policy import ReviewPolicy
    from nanobot.checkpoint.snapshot import CheckpointSnapshot


class CheckpointBroker:
    """Coordinates checkpoint pauses between the runner and the user.

    The runner calls :meth:`pause` which blocks (await) until a review
    decision is made.  In Phase 2 the review is automatic (policy) with
    text-only escalation; Phase 3 adds Telegram inline-keyboard callbacks.
    """

    def __init__(
        self,
        policy: ReviewPolicy,
        bus: MessageBus,
        origin: dict[str, Any],
    ) -> None:
        self._policy = policy
        self._bus = bus
        self._origin = origin
        self._pending_events: dict[str, asyncio.Event] = {}
        self._pending_results: dict[str, tuple[UserAction, int]] = {}
        self._on_status: Callable[[str, str], Any] | None = None

    # -- public API ----------------------------------------------------------

    async def pause(
        self,
        task_id: str,
        snapshot: CheckpointSnapshot,
    ) -> tuple[UserAction, int]:
        """Block until a review decision is available.

        Returns ``(action, extra_iterations)``.
        """
        event = asyncio.Event()
        self._pending_events[task_id] = event

        decision = self._policy.evaluate(snapshot)

        if decision.action == CheckpointAction.AUTO_CONTINUE:
            logger.info(
                "Checkpoint [{}]: auto-continue (confidence={:.1f}): {}",
                task_id,
                decision.confidence,
                decision.reason,
            )
            del self._pending_events[task_id]
            # Grant remaining iterations as extra budget
            remaining = max(0, snapshot.max_iterations - snapshot.total_iterations)
            return (UserAction.CONTINUE, remaining)

        # ── Escalate ──────────────────────────────────────────────────────
        logger.info(
            "Checkpoint [{}]: escalating — {}",
            task_id,
            decision.reason,
        )
        await self._send_escalation(snapshot, decision.reason)

        try:
            await asyncio.wait_for(event.wait(), timeout=self._policy.review_timeout)
            action, extra = self._pending_results.pop(
                task_id, (UserAction.CONTINUE, max(0, snapshot.max_iterations - snapshot.total_iterations)),
            )
            # Handle DETAILS sub-flow: re-send info, re-wait
            if action == UserAction.DETAILS:
                await self._send_details(snapshot)
                event.clear()
                try:
                    await asyncio.wait_for(event.wait(), timeout=self._policy.review_timeout)
                    action, extra = self._pending_results.pop(
                        task_id, (UserAction.CONTINUE, max(0, snapshot.max_iterations - snapshot.total_iterations)),
                    )
                except asyncio.TimeoutError:
                    await self._send_timeout_notification(snapshot)
                    return (UserAction.CONTINUE, max(0, snapshot.max_iterations - snapshot.total_iterations))
            return (action, extra)
        except asyncio.TimeoutError:
            logger.warning("Checkpoint [{}]: review timeout — auto-continuing", task_id)
            await self._send_timeout_notification(snapshot)
            return (UserAction.CONTINUE, max(0, snapshot.max_iterations - snapshot.total_iterations))
        finally:
            self._pending_events.pop(task_id, None)
            self._pending_results.pop(task_id, None)

    def resolve_checkpoint(
        self,
        task_id: str,
        action: UserAction,
        param: int | None = None,
    ) -> bool:
        """Resolve a pending checkpoint (Phase 3 Telegram callback entry-point)."""
        event = self._pending_events.get(task_id)
        if event is None or event.is_set():
            return False
        extra = param if param is not None else 0
        self._pending_results[task_id] = (action, extra)
        event.set()
        logger.info("Checkpoint [{}]: resolved with action={}", task_id, action.value)
        return True

    # -- internal helpers ----------------------------------------------------

    async def _send_escalation(self, snapshot: CheckpointSnapshot, reason: str) -> None:
        """Send a text-only escalation alert via the outbound bus."""
        recent_tools = ", ".join(
            f"{tc.tool_name}" for tc in snapshot.tool_calls[-5:]
        ) or "(none)"
        alert = (
            f"⚠️ Checkpoint review needed\n\n"
            f"Iteration: {snapshot.total_iterations}/{snapshot.max_iterations}\n"
            f"Files touched: {len(snapshot.files_touched) or 'none'}\n"
            f"Errors: {snapshot.error_count}\n"
            f"Recent tools: {recent_tools}\n\n"
            f"Reason: {reason}\n\n"
            f"The subagent is paused. Auto-continuing in "
            f"{self._policy.review_timeout}s if no response."
        )
        await self._bus.publish_outbound(OutboundMessage(
            channel=self._origin.get("channel", "cli"),
            chat_id=self._origin.get("chat_id", "direct"),
            content=alert,
        ))

    async def _send_details(self, snapshot: CheckpointSnapshot) -> None:
        """Send a detailed snapshot summary (DETAILS sub-flow)."""
        tool_lines = "\n".join(
            f"  {i + 1}. {tc.tool_name}: {tc.detail}"
            for i, tc in enumerate(snapshot.tool_calls[-10:])
        )
        file_lines = "\n".join(f"  - {f}" for f in sorted(snapshot.files_touched)[:10])
        details = (
            f"📋 Checkpoint details\n\n"
            f"Iteration: {snapshot.total_iterations}/{snapshot.max_iterations}\n"
            f"Errors: {snapshot.error_count}\n\n"
            f"Tool calls (last 10):\n{tool_lines or '  (none)'}\n\n"
            f"Files touched:\n{file_lines or '  (none)'}\n\n"
            f"Last LLM output: {snapshot.last_llm_outputs[-1] if snapshot.last_llm_outputs else '(none)'}\n\n"
            f"Respond to continue, stop, or mark done."
        )
        await self._bus.publish_outbound(OutboundMessage(
            channel=self._origin.get("channel", "cli"),
            chat_id=self._origin.get("chat_id", "direct"),
            content=details,
        ))

    async def _send_timeout_notification(self, snapshot: CheckpointSnapshot) -> None:
        """Notify that the review timeout expired and the subagent is auto-continuing."""
        notification = (
            f"⏰ Checkpoint auto-continued\n\n"
            f"No response within {self._policy.review_timeout}s. "
            f"Subagent resuming at iteration {snapshot.total_iterations}/{snapshot.max_iterations}."
        )
        await self._bus.publish_outbound(OutboundMessage(
            channel=self._origin.get("channel", "cli"),
            chat_id=self._origin.get("chat_id", "direct"),
            content=notification,
        ))
