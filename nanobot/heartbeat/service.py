"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "description": "Report heartbeat decision after reviewing tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["skip", "run", "review"],
                        "description": (
                            "skip = nothing to do, "
                            "run = has active tasks to spawn/work on, "
                            "review = subagent completed a task, decide done/failed"
                        ),
                    },
                    "tasks": {
                        "type": "string",
                        "description": "Natural-language summary of active tasks (required for run/review)",
                    },
                    "review_decision": {
                        "type": "array",
                        "description": "For review action: list of {task_id, verdict, note}. verdict: done or failed.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "verdict": {"type": "string", "enum": ["done", "failed"]},
                                "note": {"type": "string"},
                            },
                            "required": ["task_id", "verdict", "note"],
                        },
                    },
                },
                "required": ["action"],
            },
        },
    }
]


class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.

    Phase 1 (decision): reads HEARTBEAT.md and asks the LLM — via a virtual
    tool call — whether there are active tasks.  This avoids free-text parsing
    and the unreliable HEARTBEAT_OK token.

    Phase 2 (execution): only triggered when Phase 1 returns ``run``.  The
    ``on_execute`` callback runs the task through the full agent loop and
    returns the result to deliver.
    """

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
        # on_tick_report is set as an attribute AFTER construction, not a constructor param.
        # Do NOT add it to __init__ signature — set it as: heartbeat.on_tick_report = fn
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

    async def _decide(
        self,
        content: str,
        pending_review: list[dict[str, str]] | None = None,
    ) -> tuple[str, str, list[dict[str, str]]]:
        """Phase 1: ask LLM to decide skip/run/review via virtual tool call.

        Returns (action, tasks, review_decisions).
        review_decisions is a list of dicts with task_id, verdict, note.
        """
        from nanobot.utils.helpers import current_time_str

        user_content = (
            f"Current Time: {current_time_str(self.timezone)}\n\n"
            "Review the following HEARTBEAT.md and decide whether there are active tasks.\n\n"
            f"{content}"
        )

        if pending_review:
            review_block = "\n\n## Pending Review — subagent-delegated tasks that completed\n"
            review_block += "These tasks were delegated to subagents which have finished. "
            review_block += "Review the task descriptions and decide: mark as done (success) or failed.\n\n"
            for t in pending_review:
                review_block += f"- **{t['id']}**: {t['title']}\n"
            user_content += review_block

        response = await self.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": (
                    "You are a heartbeat agent. Call the heartbeat tool to report your decision.\n\n"
                    "Actions:\n"
                    "- skip: nothing to do\n"
                    "- run: there are active tasks to work on (spawn subagent, do research, etc.)\n"
                    "- review: a subagent completed a delegated task and you need to decide done/failed\n\n"
                    "For review action, include review_decision array with task_id, verdict (done/failed), "
                    "and note (validation evidence). Only mark done if the subagent likely succeeded "
                    "(tests passed, files created, etc.). If uncertain, mark failed."
                )},
                {"role": "user", "content": user_content},
            ],
            tools=_HEARTBEAT_TOOL,
            model=self.model,
        )

        if not response.has_tool_calls:
            return "skip", "", []

        args = response.tool_calls[0].arguments
        action = args.get("action", "skip")
        tasks = args.get("tasks", "")
        review_decisions = args.get("review_decision", [])
        if not isinstance(review_decisions, list):
            review_decisions = []
        return action, tasks, review_decisions

    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Heartbeat started (every {}s)", self.interval_s)

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

    async def _tick(self) -> None:
        """Execute a single heartbeat tick."""
        from nanobot.utils.evaluator import evaluate_response

        content = self._read_heartbeat_file()
        if not content:
            logger.debug("Heartbeat: HEARTBEAT.md missing or empty")
            return

        logger.info("Heartbeat: checking for tasks...")

        # Send start ping
        if self.on_tick_report:
            await self.on_tick_report("start", "", None)

        try:
            # Check for completed delegations that need review
            pending_review: list[dict[str, str]] = []
            try:
                from nanobot.agent.task_lifecycle import query_pending_review
                pending_review = await query_pending_review()
            except Exception as exc:
                logger.warning("Heartbeat: delegation review check failed: {}", exc)

            if pending_review:
                logger.info("Heartbeat: {} tasks pending review", len(pending_review))

            action, tasks, review_decisions = await self._decide(content, pending_review or None)

            # Handle review decisions first
            if review_decisions:
                from nanobot.agent.task_lifecycle import close_task
                for dec in review_decisions:
                    tid = dec.get("task_id", "")
                    verdict = dec.get("verdict", "")
                    note = dec.get("note", "")
                    if not tid:
                        continue
                    if verdict == "done" and note:
                        ok = await close_task(tid, note)
                        logger.info(
                            "Heartbeat: closed task {} as done: {}",
                            tid[:20], "ok" if ok else "failed",
                        )
                    elif verdict == "failed":
                        from nanobot.agent.task_lifecycle import mark_task_delegation_failure
                        await mark_task_delegation_failure(tid, note or "heartbeat review: marked failed")
                        logger.info("Heartbeat: marked task {} as failed", tid[:20])

            if action == "review" and not review_decisions:
                logger.info("Heartbeat: review action but no decisions — skipping")
                return

            if action != "run":
                logger.info("Heartbeat: OK (nothing to report)")
            else:
                logger.info("Heartbeat: tasks found, executing...")
                if self.on_execute:
                    response = await self.on_execute(tasks)

                    if response:
                        should_notify = await evaluate_response(
                            response, tasks, self.provider, self.model,
                        )
                        if should_notify and self.on_notify:
                            logger.info("Heartbeat: completed, delivering response")
                            await self.on_notify(response)
                        else:
                            logger.info("Heartbeat: silenced by post-run evaluation")

            # Send brief tick report to user
            if self.on_tick_report:
                await self.on_tick_report(action, tasks, review_decisions)
        except Exception:
            logger.exception("Heartbeat execution failed")

    async def trigger_now(self) -> dict:
        """Manually trigger a heartbeat and return a structured result.

        Returns a dict with:
            - ``action``: ``"skip"``, ``"run"``, or ``"review"``
            - ``tasks``: natural-language summary (empty for skip)
            - ``result``: execution response (only for run)
            - ``review_decisions``: list of review decisions
        """
        # Send start ping
        if self.on_tick_report:
            await self.on_tick_report("start", "", None)

        content = self._read_heartbeat_file()
        if not content:
            result = {"action": "skip", "tasks": "", "result": "HEARTBEAT.md missing or empty", "review_decisions": []}
            if self.on_tick_report:
                await self.on_tick_report("skip", "", None)
            return result

        pending_review: list[dict[str, str]] = []
        try:
            from nanobot.agent.task_lifecycle import query_pending_review
            pending_review = await query_pending_review()
        except Exception:
            pass

        action, tasks, review_decisions = await self._decide(content, pending_review or None)

        # Handle review decisions
        if review_decisions:
            from nanobot.agent.task_lifecycle import close_task
            closed = []
            for dec in review_decisions:
                tid = dec.get("task_id", "")
                verdict = dec.get("verdict", "")
                note = dec.get("note", "")
                if not tid:
                    continue
                if verdict == "done" and note:
                    ok = await close_task(tid, note)
                    closed.append({"task_id": tid, "verdict": verdict, "ok": ok})
                elif verdict == "failed":
                    from nanobot.agent.task_lifecycle import mark_task_delegation_failure
                    await mark_task_delegation_failure(tid, note or "heartbeat review: marked failed")
                    closed.append({"task_id": tid, "verdict": verdict, "ok": True})
            if closed:
                if self.on_tick_report:
                    await self.on_tick_report("review", tasks, review_decisions)
                return {"action": "review", "tasks": tasks, "result": "", "review_decisions": closed}

        if action != "run" or not self.on_execute:
            if self.on_tick_report:
                await self.on_tick_report(action, tasks, None)
            return {"action": action, "tasks": tasks, "result": "", "review_decisions": []}

        result = await self.on_execute(tasks)
        if self.on_tick_report:
            await self.on_tick_report("run", tasks, None)
        return {"action": "run", "tasks": tasks, "result": result or "", "review_decisions": []}
