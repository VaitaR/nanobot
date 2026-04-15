"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

from nanobot.heartbeat.drain import (
    HEARTBEAT_SYSTEM_PROMPT,
    HEARTBEAT_TOOL,
    collect_pending,
    mark_delivered,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class HeartbeatService:
    """Periodic heartbeat: LLM decides skip/run/review, then executes if needed."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        interval_s: int = 30 * 60,
        enabled: bool = True,
        timezone: str | None = None,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.on_execute = on_execute
        self.on_notify = on_notify
        # Set as attribute AFTER construction, NOT in __init__ signature.
        self.on_tick_report: Callable[[str], Coroutine[Any, Any, None]] | None = None
        self.interval_s = interval_s
        self.enabled = enabled
        self.timezone = timezone
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Review helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _fetch_pending_review() -> list[dict[str, str]]:
        """Query completed delegations needing review."""
        try:
            from nanobot.agent.task_lifecycle import query_pending_review
            return await query_pending_review()
        except Exception:
            return []

    @staticmethod
    async def _reap_stale_delegations() -> list[dict[str, str | bool]]:
        """Reap stale delegated tasks (F-007)."""
        try:
            from nanobot.agent.task_lifecycle import reap_stale_delegations

            return await reap_stale_delegations(timeout_s=2 * 60 * 60)
        except Exception:
            return []

    @staticmethod
    async def _process_reviews(review_decisions: list[dict[str, str]]) -> list[dict]:
        """Process review decisions: close tasks or mark failures.

        Returns list of processed results for reporting.
        """
        if not review_decisions:
            return []
        from nanobot.agent.task_lifecycle import close_task, mark_task_delegation_failure

        results: list[dict] = []
        for dec in review_decisions:
            tid, verdict, note = dec.get("task_id", ""), dec.get("verdict", ""), dec.get("note", "")
            if not tid:
                continue
            if verdict == "done" and note:
                # Prefix heartbeat source and ensure evidence marker (F-015 fix)
                if not any(m in note.lower() for m in ("checked:", "verified:", "confirmed:", "tested:", "ran:", "output:")):
                    note = f"verified: [heartbeat-review] {note}"
                else:
                    note = f"[heartbeat-review] {note}"
                ok = await close_task(tid, note)
                logger.info("Heartbeat: closed task {} as {}", tid[:20], "done" if ok else "failed")
                results.append({"task_id": tid, "verdict": verdict, "ok": ok})
            elif verdict == "failed":
                await mark_task_delegation_failure(tid, note or "heartbeat review: marked failed")
                logger.info("Heartbeat: marked task {} as failed", tid[:20])
                results.append({"task_id": tid, "verdict": verdict, "ok": True})
        return results

    # ------------------------------------------------------------------
    # Phase 1: Decide
    # ------------------------------------------------------------------

    async def _decide(
        self, content: str, pending_review: list[dict[str, str]] | None = None,
    ) -> tuple[str, str, list[dict[str, str]]]:
        """Ask LLM to decide skip/run/review. Returns (action, tasks, review_decisions)."""
        from nanobot.utils.helpers import current_time_str

        user_content = (
            f"Current Time: {current_time_str(self.timezone)}\n\n"
            "Review the following HEARTBEAT.md and decide whether there are active tasks.\n\n"
            f"{content}"
        )
        if pending_review:
            user_content += (
                "\n\n## Pending Review — completed subagent tasks\n"
                "Decide: mark done (success) or failed.\n"
                + "\n".join(f"- **{t['id']}**: {t['title']}" for t in pending_review)
            )

        response = await self.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": HEARTBEAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tools=HEARTBEAT_TOOL,
            model=self.model,
        )
        if not response.has_tool_calls:
            return "skip", "", []
        args = response.tool_calls[0].arguments
        review = args.get("review_decision", [])
        return args.get("action", "skip"), args.get("tasks", ""), review if isinstance(review, list) else []

    # ------------------------------------------------------------------
    # Notification drain
    # ------------------------------------------------------------------

    async def _deliver_pending(self) -> None:
        """Deliver pending proactive notifications and mark them sent."""
        if not self.on_notify:
            return
        try:
            pending = collect_pending()
            ids = [n["id"] for n in pending]
            for n in pending:
                try:
                    await self.on_notify(n.get("content", ""))
                except Exception:
                    logger.warning("drain: delivery failed for {}", n.get("id"))
            if ids:
                mark_delivered(ids)
        except Exception:
            logger.exception("drain: unexpected error")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Heartbeat started (every {}s)", self.interval_s)

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

    # ------------------------------------------------------------------
    # Tick (periodic)
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        from nanobot.utils.evaluator import evaluate_response

        content = self._read_heartbeat_file()
        if not content:
            return

        logger.info("Heartbeat: checking for tasks...")
        if self.on_tick_report:
            await self.on_tick_report("start", "", None)

        try:
            stale_reaped = await self._reap_stale_delegations()
            if stale_reaped:
                logger.warning(
                    "Heartbeat reaper: processed {} stale delegated task(s)",
                    len(stale_reaped),
                )

            pending_review = await self._fetch_pending_review()
            if pending_review:
                logger.info("Heartbeat: {} tasks pending review", len(pending_review))

            action, tasks, review_decisions = await self._decide(content, pending_review or None)
            await self._process_reviews(review_decisions)

            if action == "review" and not review_decisions:
                logger.info("Heartbeat: review action but no decisions")
                return

            if action == "run" and self.on_execute:
                logger.info("Heartbeat: tasks found, executing...")
                try:
                    response = await asyncio.wait_for(self.on_execute(tasks), timeout=600)
                except asyncio.TimeoutError:
                    logger.error("Heartbeat: on_execute timed out")
                    response = None
                if response:
                    should_notify = await evaluate_response(response, tasks, self.provider, self.model)
                    if should_notify and self.on_notify:
                        await self.on_notify(response)
            else:
                logger.info("Heartbeat: nothing to report")

            if self.on_tick_report:
                await self.on_tick_report(action, tasks, review_decisions)
        except Exception:
            logger.exception("Heartbeat execution failed")

        await self._deliver_pending()

    # ------------------------------------------------------------------
    # Trigger (manual)
    # ------------------------------------------------------------------

    async def trigger_now(self) -> dict:
        """Manually trigger heartbeat, return structured result."""
        if self.on_tick_report:
            await self.on_tick_report("start", "", None)

        content = self._read_heartbeat_file()
        if not content:
            r = {"action": "skip", "tasks": "", "result": "HEARTBEAT.md missing", "review_decisions": []}
            if self.on_tick_report:
                await self.on_tick_report("skip", "", None)
            return r

        stale_reaped = await self._reap_stale_delegations()
        if stale_reaped:
            logger.warning(
                "Heartbeat reaper(trigger_now): processed {} stale delegated task(s)",
                len(stale_reaped),
            )

        action, tasks, review_decisions = await self._decide(
            content, await self._fetch_pending_review() or None,
        )
        closed = await self._process_reviews(review_decisions)
        if closed:
            if self.on_tick_report:
                await self.on_tick_report("review", tasks, review_decisions)
            return {"action": "review", "tasks": tasks, "result": "", "review_decisions": closed}

        if action != "run" or not self.on_execute:
            if self.on_tick_report:
                await self.on_tick_report(action, tasks, None)
            return {"action": action, "tasks": tasks, "result": "", "review_decisions": []}

        try:
            result = await asyncio.wait_for(self.on_execute(tasks), timeout=600)
        except asyncio.TimeoutError:
            logger.error("Heartbeat trigger_now: timed out")
            result = None
        if self.on_tick_report:
            await self.on_tick_report("run", tasks, None)
        await self._deliver_pending()
        return {"action": "run", "tasks": tasks, "result": result or "", "review_decisions": []}
