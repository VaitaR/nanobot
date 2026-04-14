"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
import json
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


def _emit_heartbeat_event(event_type: str, data: dict[str, Any] | None = None) -> None:
    """Best-effort write to heartbeat.jsonl session log.  Never raises."""
    try:
        from nanobot_workspace.observability.session_writer import write_event

        write_event(event_type, data)
    except Exception:
        pass


class HeartbeatService:
    """Periodic heartbeat: LLM decides skip/run/review, then executes if needed."""

    # Memory reindex interval: every N ticks (~hours at 30-min intervals).
    _MEMORY_REINDEX_INTERVAL = 2

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        is_user_active: Callable[[], bool] | None = None,
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
        self.is_user_active = is_user_active
        self.interval_s = interval_s
        self.enabled = enabled
        self.timezone = timezone
        self._running = False
        self._task: asyncio.Task | None = None
        self._tick_count: int = 0
        self._boredom_tasks: set[asyncio.Task] = set()
        self._boredom_loop: asyncio.AbstractEventLoop | None = None
        self._spawn_boredom_delegation_cb: Callable[[str, str, str], Coroutine[Any, Any, str]] | None = None

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
        _emit_heartbeat_event("heartbeat.started", {"interval_s": self.interval_s})

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                if self._running:
                    await self._tick()
                await asyncio.sleep(self.interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

    # ------------------------------------------------------------------
    # Tick (periodic)
    # ------------------------------------------------------------------

    def _maybe_reindex_memory(self) -> None:
        """Run incremental memory embedding reindex (non-critical).

        Called every ``_MEMORY_REINDEX_INTERVAL`` ticks.  From cache this
        takes <1 s.  On failure, the agent still works without embeddings.
        """
        try:
            from nanobot_workspace.memory.search import reindex_incremental

            stats = reindex_incremental(self.workspace)
            if stats:
                logger.info(
                    "Memory reindex: embedded={}, skipped={}, errors={}",
                    stats.get("embedded", 0),
                    stats.get("skipped", 0),
                    stats.get("errors", 0),
                )
        except Exception:
            logger.debug("Memory reindex skipped (module unavailable)")

    def _collect_boredom_task_snapshot(self) -> list[dict[str, Any]]:
        """Load the active workspace task snapshot for boredom decisions."""
        try:
            from nanobot_workspace.tasks.store import TaskStore

            return [
                {
                    "id": task.id,
                    "status": task.status,
                    "title": task.title,
                    "updated": task.updated,
                    "blocked_reason": task.blocked_reason,
                    "source": task.source,
                }
                for task in TaskStore(self.workspace).list_tasks("active")
            ]
        except Exception as e:
            logger.debug("Boredom task snapshot unavailable: {}", e)
            return []

    def set_spawn_callback(
        self,
        cb: Callable[[str, str, str], Coroutine[Any, Any, str]] | None,
    ) -> None:
        """Override the boredom delegation spawn callback."""
        self._spawn_boredom_delegation_cb = cb

    @staticmethod
    def _serialize_boredom_result(result: Any) -> dict[str, Any]:
        data = {
            "state_before": getattr(result, "state_before", "?"),
            "state_after": getattr(result, "state_after", "?"),
            "action_taken": getattr(result, "action_taken", "?"),
            "ideas_accepted": getattr(result, "ideas_accepted", 0),
            "ideas_generated": getattr(result, "ideas_generated", 0),
            "next_action": getattr(result, "next_action", None),
            "error": getattr(result, "error", None),
            "duration_ms": getattr(result, "duration_ms", 0),
            "open_tasks": getattr(result, "open_tasks", 0),
        }
        initiative = getattr(result, "initiative", None)
        if initiative is not None:
            idea = getattr(initiative, "idea", initiative)
            data["initiative"] = {
                "title": getattr(idea, "title", ""),
                "description": getattr(idea, "description", ""),
                "category": getattr(idea, "category", ""),
                "target_paths": getattr(idea, "target_paths", []),
                "acceptance_criteria": getattr(idea, "estimated_checks", []),
            }
        return data

    @staticmethod
    def _load_boredom_hooks() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        from nanobot_workspace.proactive.boredom.heartbeat_integration import (
            apply_boredom_delegation_result,
            boredom_tick,
            materialize_boredom_initiative,
            register_boredom_delegation_spawner,
        )

        return boredom_tick, materialize_boredom_initiative, register_boredom_delegation_spawner, apply_boredom_delegation_result

    async def _spawn_boredom_delegation(self, task: str, label: str, executors: list[str]) -> str:
        """Spawn the boredom idea-generation delegation using the configured fallback chain."""
        if self._spawn_boredom_delegation_cb is None:
            raise RuntimeError("boredom delegation spawn callback is not configured")
        last_error: Exception | None = None
        for executor in executors:
            try:
                return await self._spawn_boredom_delegation_cb(task, label, executor)
            except Exception as exc:
                last_error = exc
                logger.warning("Boredom delegation spawn failed for {}: {}", executor, exc)
        raise RuntimeError(f"boredom delegation spawn failed for {executors}") from last_error

    def _spawn_boredom_delegation_from_thread(self, task: str, label: str, executors: list[str]) -> str:
        """Bridge boredom_tick's worker thread to the runtime event loop."""
        if self._boredom_loop is None:
            raise RuntimeError("heartbeat loop unavailable for boredom delegation")
        future = asyncio.run_coroutine_threadsafe(
            self._spawn_boredom_delegation(task, label, executors),
            self._boredom_loop,
        )
        return future.result(timeout=30)

    @staticmethod
    def _boredom_state_path(workspace: Path) -> Path:
        return workspace / "data" / "boredom_state.json"

    def _notify_boredom_completion(self, success: bool, reason: str | None = None) -> None:
        """Best-effort boredom completion notification for direct on_execute delegation."""
        from nanobot_workspace.proactive.boredom.models import BoredomMode
        from nanobot_workspace.proactive.boredom.orchestrator import BoredomOrchestrator
        from nanobot_workspace.proactive.boredom.store import BoredomStore

        store = BoredomStore(self._boredom_state_path(self.workspace))
        orchestrator = BoredomOrchestrator(store=store)
        state = orchestrator.get_state()
        if state.mode != BoredomMode.BORED_DELEGATING:
            return

        if success:
            # Do not reset state here. `_process_boredom_delegations()` runs on the
            # next heartbeat tick, reads the delegation output, triages candidates,
            # materializes a task, and then resets state. The 35-minute staleness
            # timeout remains the safety net if that processing never completes.
            logger.debug("Boredom delegation completed successfully; deferring state reset to next tick")
            return

        fail_hook = getattr(orchestrator, "fail_safe_execution", None)
        if callable(fail_hook):
            fail_hook(reason=reason or "delegation_failed")
            return

        state.mode = BoredomMode.IDLE_COUNTING
        state.empty_tick_count = 0
        state.active_initiative = None
        store.save(state)

    def _process_boredom_delegation_payload(self, initiative: dict[str, Any]) -> dict[str, Any] | None:
        """Load, apply, and materialize a completed boredom delegation result."""
        from nanobot_workspace.proactive.boredom.heartbeat_integration import (
            BoredomTickResult,
            apply_boredom_delegation_result,
            boredom_delegation_output_path,
            materialize_boredom_initiative,
        )

        delegation_id = initiative.get("initiative_id") or initiative.get("runtime_task_id")
        if not delegation_id:
            logger.warning("Boredom delegation finished without delegation identifier")
            return None

        output_path = boredom_delegation_output_path(delegation_id, workspace_root=self.workspace)
        if not output_path.exists():
            logger.warning("Boredom delegation output missing: {}", output_path)
            return None

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        transition = apply_boredom_delegation_result(
            delegation_id,
            payload,
            workspace_root=self.workspace,
        )
        if transition is None:
            logger.info("Boredom delegation {} did not match an active initiative", delegation_id)
            return None

        logger.info(
            "Boredom delegation {} applied: {} -> {}",
            delegation_id,
            transition.previous_mode.value,
            transition.current_mode.value,
        )
        if transition.initiative is None or getattr(transition.initiative, "idea", None) is None:
            logger.info("Boredom delegation {} produced no accepted initiative", delegation_id)
            return None

        result = BoredomTickResult(
            timestamp="",
            state_before=transition.previous_mode.value,
            state_after=transition.current_mode.value,
            open_tasks=0,
            ideas_generated=len(payload.get("candidates", [])) if isinstance(payload.get("candidates"), list) else 0,
            ideas_accepted=1,
            action_taken="apply_delegation_result",
            duration_ms=0,
            next_action=transition.next_action,
            initiative=transition.initiative,
        )
        created = materialize_boredom_initiative(result, workspace_root=self.workspace)
        if created is None:
            logger.info("Boredom delegation {} accepted ideas but created no task", delegation_id)
            return None

        logger.info(
            "Boredom delegation {} created task {} for '{}'",
            delegation_id,
            created.get("task_id"),
            created.get("title", ""),
        )
        return created

    async def _process_boredom_delegations(self) -> None:
        """Apply completed boredom delegation outputs back into workspace state."""
        try:
            (
                _boredom_tick,
                _materialize,
                _register_spawner,
                apply_boredom_delegation_result,
            ) = self._load_boredom_hooks()
            from nanobot_workspace.proactive.boredom.heartbeat_integration import (
                boredom_delegation_output_path,
            )
            from nanobot_workspace.proactive.boredom.models import BoredomMode
            from nanobot_workspace.proactive.boredom.orchestrator import BoredomOrchestrator
            from nanobot_workspace.proactive.boredom.store import BoredomStore
        except (ImportError, ModuleNotFoundError) as exc:
            logger.debug("Boredom delegation check skipped: {}", exc)
            return

        try:
            orchestrator = BoredomOrchestrator(store=BoredomStore(self._boredom_state_path(self.workspace)))
            state = await asyncio.to_thread(orchestrator.get_state)
            initiative = state.active_initiative
            if state.mode != BoredomMode.BORED_DELEGATING or initiative is None:
                return
            if not initiative.task_id:
                return

            output_path = boredom_delegation_output_path(initiative.task_id, workspace_root=self.workspace)
            if not output_path.exists():
                logger.debug("Boredom delegation {} waiting for output", initiative.task_id)
                return

            payload = await asyncio.to_thread(output_path.read_text, encoding="utf-8")
            data = json.loads(payload)
            transition = await asyncio.to_thread(
                apply_boredom_delegation_result,
                initiative.task_id,
                data,
                orchestrator=orchestrator,
                workspace_root=self.workspace,
            )
            if transition is not None:
                logger.info(
                    "Boredom delegation applied: {} -> {}",
                    transition.previous_mode.value,
                    transition.current_mode.value,
                )
                # Materialize the initiative into a workspace task
                if transition.initiative and transition.initiative.idea:
                    try:
                        await asyncio.to_thread(
                            _materialize,
                            transition,
                            workspace_root=self.workspace,
                        )
                        await asyncio.to_thread(orchestrator.complete_safe_execution)
                        logger.info(
                            "Boredom: materialized initiative '{}'",
                            transition.initiative.idea.title,
                        )
                    except Exception as mat_err:
                        logger.warning("Boredom: materialization failed: {}", mat_err)
                else:
                    logger.info("Boredom delegation applied but no idea accepted")
        except Exception as e:
            logger.exception("boredom_delegation_check_failed: {}", e)

    async def _run_boredom_tick(self) -> None:
        """Run boredom engine tick in-process with the runtime provider stack."""
        try:
            boredom_tick, materialize_boredom_initiative, register_boredom_delegation_spawner, _apply_boredom = self._load_boredom_hooks()
        except Exception as e:
            logger.debug("Boredom tick skipped: {}", e)
            return

        self._boredom_loop = asyncio.get_running_loop()
        register_boredom_delegation_spawner(self._spawn_boredom_delegation_from_thread)
        snapshot = self._collect_boredom_task_snapshot()
        try:
            result = await asyncio.to_thread(
                boredom_tick,
                snapshot,
                workspace_root=self.workspace,
            )
            if result.next_action != "wait_for_delegation" and result.initiative:
                initiative_data = await asyncio.to_thread(
                    materialize_boredom_initiative,
                    result,
                    workspace_root=self.workspace,
                )
            else:
                initiative_data = None
            if initiative_data is not None:
                task = asyncio.create_task(self._delegate_boredom_initiative(initiative_data))
                self._boredom_tasks.add(task)
                task.add_done_callback(self._boredom_tasks.discard)
            data = self._serialize_boredom_result(result)
            state = data.get("state_after", "?")
            action = data.get("action_taken", "?")
            ideas = data.get("ideas_accepted", 0)
            logger.info(
                "Boredom tick: state={}, action={}, ideas={}",
                state, action, ideas,
            )
            _emit_heartbeat_event("heartbeat.boredom_tick", data)
        except Exception as e:
            logger.debug("Boredom tick skipped: {}", e)

    async def _delegate_boredom_initiative(self, initiative: dict[str, Any]) -> None:
        """Delegate a SAFE boredom initiative to on_execute, non-blocking."""
        if initiative.get("risk_class") != "safe":
            return
        if not self.on_execute:
            return
        try:
            from nanobot_workspace.proactive.boredom.git_ops import (
                find_dirty_overlap,
                list_dirty_paths,
            )

            target_paths = initiative.get("target_paths", [])
            dirty = await asyncio.to_thread(list_dirty_paths, self.workspace)
            overlaps = find_dirty_overlap(target_paths, dirty)
            if overlaps:
                logger.info("Boredom: skip delegation, dirty overlap: {}", overlaps)
                return
        except Exception as e:
            logger.debug("Boredom delegation: overlap check failed: {}", e)
            return
        title = initiative.get("title", "boredom initiative")
        category = initiative.get("category", "")
        target_str = ", ".join(initiative.get("target_paths", []))
        description = initiative.get("description", "")
        checks = "; ".join(initiative.get("acceptance_criteria", []))
        prompt = (
            f"Boredom initiative: {title}\n"
            f"Category: {category}\n"
            f"Target paths: {target_str}\n"
            f"Description: {description}\n"
            f"Acceptance criteria: {checks}"
        ).strip()
        logger.info("Boredom: delegating initiative '{}' to subagent", title)
        _emit_heartbeat_event(
            "heartbeat.boredom_delegated",
            {
                "title": title,
                "category": category,
                "source": initiative.get("source", "boredom"),
                "task_id": initiative.get("task_id"),
                "correlation_id": initiative.get("correlation_id") or initiative.get("task_id"),
                "action_taken": "delegated",
            },
        )
        try:
            await asyncio.wait_for(self.on_execute(prompt), timeout=600)
            try:
                await asyncio.to_thread(self._process_boredom_delegation_payload, initiative)
            except Exception as e:
                logger.warning("Boredom delegation result processing failed: {}", e)
            try:
                await asyncio.to_thread(self._notify_boredom_completion, True, None)
            except Exception as e:
                logger.debug("Boredom completion notification failed after success: {}", e)
        except asyncio.TimeoutError:
            logger.warning("Boredom: initiative delegation timed out: {}", title)
            try:
                await asyncio.to_thread(self._notify_boredom_completion, False, "timeout")
            except Exception as e:
                logger.debug("Boredom completion notification failed after timeout: {}", e)
        except Exception as e:
            logger.warning("Boredom: initiative delegation failed: {}", e)
            try:
                await asyncio.to_thread(self._notify_boredom_completion, False, str(e))
            except Exception as notify_error:
                logger.debug("Boredom completion notification failed after error: {}", notify_error)

    async def _run_tick(self) -> dict[str, Any]:
        from nanobot.utils.evaluator import evaluate_response

        # --- Maintenance: incremental memory reindex ---
        self._tick_count += 1
        if self._tick_count % self._MEMORY_REINDEX_INTERVAL == 0:
            self._maybe_reindex_memory()

        # --- Boredom engine tick (always runs, independent of LLM decision) ---
        await self._process_boredom_delegations()
        await self._run_boredom_tick()

        content = self._read_heartbeat_file()
        if not content:
            return {"action": "skip", "tasks": "", "result": "HEARTBEAT.md missing", "review_decisions": []}

        logger.info("Heartbeat: checking for tasks...")
        _emit_heartbeat_event("heartbeat.checking")
        if self.is_user_active and self.is_user_active():
            logger.info("Heartbeat: user session active, skipping LLM call")
            _emit_heartbeat_event("heartbeat.nothing_to_report")
            return {"action": "skip", "tasks": "", "result": "", "review_decisions": []}

        pending_review = await self._fetch_pending_review()
        if pending_review:
            logger.info("Heartbeat: {} tasks pending review", len(pending_review))

        action, tasks, review_decisions = await self._decide(content, pending_review or None)
        closed_reviews = await self._process_reviews(review_decisions)

        if action == "review" and not closed_reviews:
            logger.info("Heartbeat: review action but no decisions")
            return {"action": "review", "tasks": tasks, "result": "", "review_decisions": []}

        if action == "run" and self.on_execute:
            logger.info("Heartbeat: tasks found, executing...")
            _emit_heartbeat_event("heartbeat.executing", {"tasks_preview": tasks[:200]})
            try:
                response = await asyncio.wait_for(self.on_execute(tasks), timeout=600)
            except asyncio.TimeoutError:
                logger.error("Heartbeat: on_execute timed out")
                response = None
            if response:
                should_notify = await evaluate_response(response, tasks, self.provider, self.model)
                if should_notify and self.on_notify:
                    await self.on_notify(response)
            return {"action": "run", "tasks": tasks, "result": response or "", "review_decisions": closed_reviews}

        logger.info("Heartbeat: nothing to report")
        _emit_heartbeat_event("heartbeat.nothing_to_report")
        return {"action": action, "tasks": tasks, "result": "", "review_decisions": closed_reviews}

    async def _tick(self) -> None:
        if self.on_tick_report:
            await self.on_tick_report("start", "", None)

        try:
            result = await self._run_tick()
        except Exception:
            logger.exception("Heartbeat execution failed")
            return
        finally:
            await self._deliver_pending()

        if self.on_tick_report:
            await self.on_tick_report(result["action"], result["tasks"], result["review_decisions"] or None)

    # ------------------------------------------------------------------
    # Trigger (manual)
    # ------------------------------------------------------------------

    async def trigger_now(self) -> dict:
        """Manually trigger heartbeat, return structured result."""
        if self.on_tick_report:
            await self.on_tick_report("start", "", None)
        try:
            result = await self._run_tick()
        finally:
            await self._deliver_pending()

        if self.on_tick_report:
            await self.on_tick_report(result["action"], result["tasks"], result["review_decisions"] or None)
        return result
