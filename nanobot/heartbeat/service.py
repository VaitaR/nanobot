"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger
from nanobot_workspace.core.peak_hours import is_zai_peak

from nanobot.heartbeat.boredom_prompt_context import build_boredom_prompt_context
from nanobot.heartbeat.drain import (
    HEARTBEAT_SYSTEM_PROMPT,
    HEARTBEAT_TOOL,
    collect_pending,
    mark_delivered,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


_PRIORITY_ORDER: dict[str, int] = {"critical": 0, "high": 1, "normal": 2, "low": 3}
_COMPLEXITY_ORDER: dict[str, int] = {"s": 0, "m": 1, "l": 2, "xl": 3}

_ZAI_PEAK_HEARTBEAT_INSTRUCTION = (
    "\n\nZAI peak hours active — model is unstable, minimize tool calls, "
    "do not spawn claude-zai subagents"
)


def _emit_heartbeat_event(event_type: str, data: dict[str, Any] | None = None) -> None:
    """Best-effort write to heartbeat.jsonl session log.  Never raises."""
    try:
        from nanobot_workspace.observability.session_writer import write_event

        write_event(event_type, data)
    except Exception:
        pass


def _refresh_health_state(workspace: Path) -> dict[str, Any] | None:
    """Rebuild the compact workspace health snapshot."""
    try:
        from nanobot_workspace.observability import build_current_state

        return build_current_state(workspace)
    except Exception:
        return None


def _health_tick_context(state: dict[str, Any] | None) -> str:
    """Render a compact health section for the Phase 1 heartbeat prompt."""
    if not state:
        return ""
    overall = state.get("overall", {})
    levels = state.get("levels", {})
    l0 = levels.get("l0", {})
    l1 = levels.get("l1", {})
    lines = [
        "Current health snapshot:",
        f"- overall: {overall.get('status', 'unknown')} — {overall.get('summary', 'no summary')}",
        f"- l0: alive={l0.get('process_alive')} uptime_s={l0.get('uptime_s')} disk_free_gb={l0.get('disk_free_gb')}",
        f"- l1: heartbeat={l1.get('last_heartbeat_at')} queue={l1.get('task_queue_depth')} last_error={(l1.get('last_error') or {}).get('category', 'none')}",
    ]
    return "\n\n" + "\n".join(lines)


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
        self._spawn_boredom_delegation_cb: (
            Callable[[str, str, str], Coroutine[Any, Any, str]] | None
        ) = None
        self._spawn_reviewer_cb: Callable[[str, str, str], Coroutine[Any, Any, str]] | None = None
        self._active_review_task_ids: set[str] = set()  # tasks with reviewer currently running

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
    async def _fetch_tasks_in_review() -> list[dict[str, str]]:
        """Query tasks with status 'review' awaiting independent verification."""
        try:
            from nanobot.agent.task_lifecycle import query_tasks_in_review

            return await query_tasks_in_review()
        except Exception:
            return []

    async def _spawn_reviewer(self, task_info: dict[str, str]) -> None:
        """Spawn a glm-turbo reviewer subagent for a task in 'review' status."""
        task_id = task_info.get("id", "")
        title = task_info.get("title", "")
        if not task_id or self._spawn_reviewer_cb is None:
            return

        # Skip if a reviewer is already running for this task (avoids duplicates across ticks)
        if task_id in self._active_review_task_ids:
            logger.debug("Heartbeat: reviewer already active for task {}, skipping", task_id)
            return

        reviewer_skill_path = self.workspace / "skills" / "reviewer" / "SKILL.md"
        skill_instructions = ""
        if reviewer_skill_path.exists():
            try:
                skill_instructions = reviewer_skill_path.read_text(encoding="utf-8")
            except Exception:
                pass

        task_store_path = self.workspace / "tasks" / f"{task_id}.md"
        task_file_hint = ""
        if task_store_path.exists():
            task_file_hint = f"\nTask file: {task_store_path}"

        prompt = (
            f"{skill_instructions}\n\n"
            "## Reviewer constraints\n"
            "- Use executor glm-turbo behavior only; do not request another executor.\n"
            "- Start with deterministic checks: ruff, format check, pytest, then import/runtime integration checks.\n"
            "- If checks fail due to trivial issues in allowed files, fix and re-run them.\n"
            "- If review fails, return the task to open with a precise reason and add a review-failed history event.\n"
            "- If review passes, mark the task done with evidence from checks and AC verification.\n\n"
            f"## Task to review\n"
            f"Task ID: {task_id}\n"
            f"Title: {title}{task_file_hint}\n\n"
            "Follow the reviewer skill instructions above to verify and close or reopen this task."
        ).strip()
        label = f"review: {task_id}"
        logger.info("Heartbeat: spawning reviewer for task {}", task_id)
        try:
            await self._spawn_reviewer_cb(prompt, label, "glm-turbo")
            self._active_review_task_ids.add(task_id)
        except Exception as exc:
            logger.warning("Heartbeat: reviewer spawn failed for {}: {}", task_id, exc)
            # Roll task back to 'review' so next tick retries
            try:
                from nanobot.agent.task_lifecycle import _run_cli
                await _run_cli(
                    f"update {task_id!r} --status review --reason 'reviewer spawn failed: {exc}'"
                )
            except Exception as rollback_err:
                logger.warning("Heartbeat: failed to rollback task {} to review: {}", task_id, rollback_err)

    @staticmethod
    async def _fetch_pending_review() -> list[dict[str, str]]:
        """Query completed delegations needing review."""
        try:
            from nanobot.agent.task_lifecycle import query_pending_review

            return await query_pending_review()
        except Exception:
            return []

    async def _process_reviews(self, review_decisions: list[dict[str, str]]) -> list[dict]:
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
            # Clear active-review tracking for resolved tasks
            self._active_review_task_ids.discard(tid)
        return results

    # ------------------------------------------------------------------
    # Phase 1: Programmatic task check + optional LLM review
    # ------------------------------------------------------------------

    def _check_actionable_tasks(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Check for open/unblocked tasks programmatically.

        Returns (actionable, incomplete) where:
        - actionable: tasks with ## Files AND ## Acceptance Criteria, sorted by
          priority (critical→low) then complexity (s→xl)
        - incomplete: tasks missing one or both sections
        """
        try:
            from nanobot_workspace.tasks.store import TaskStore

            actionable_tasks: list = []
            incomplete: list[dict[str, str]] = []
            for t in TaskStore(self.workspace).list_tasks("active"):
                if t.status == "open" and not t.blocked_reason:
                    body_lower = t.body.lower() if t.body else ""
                    has_files = "## files" in body_lower
                    has_acceptance = "## acceptance" in body_lower
                    if has_files and has_acceptance:
                        actionable_tasks.append(t)
                    else:
                        incomplete.append({"id": t.id, "title": t.title})
            actionable_tasks.sort(
                key=lambda t: (
                    _PRIORITY_ORDER.get(getattr(t, "priority", "normal"), 2),
                    _COMPLEXITY_ORDER.get(getattr(t, "complexity", "m"), 1),
                )
            )
            actionable = [{"id": t.id, "title": t.title} for t in actionable_tasks]
            return actionable, incomplete
        except Exception:
            return [], []

    async def _review_pending(
        self,
        pending_review: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Ask LLM to decide done/failed for completed subagent tasks."""
        if not pending_review:
            return []
        from nanobot.utils.helpers import current_time_str

        user_content = (
            f"Current Time: {current_time_str(self.timezone)}\n\n"
            "Review the following completed subagent tasks. "
            "Decide: mark done (success) or failed.\n"
            "Only mark done if subagent likely succeeded. If uncertain, mark failed.\n\n"
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
            return []
        args = response.tool_calls[0].arguments
        review = args.get("review_decision", [])
        return review if isinstance(review, list) else []

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

    def set_spawn_reviewer_cb(
        self,
        cb: Callable[[str, str, str], Coroutine[Any, Any, str]] | None,
    ) -> None:
        """Set the reviewer subagent spawn callback."""
        self._spawn_reviewer_cb = cb

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
    def _load_boredom_hooks() -> tuple[
        Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]
    ]:
        from nanobot_workspace.proactive.boredom.heartbeat_integration import (
            apply_boredom_delegation_result,
            boredom_tick,
            materialize_boredom_initiative,
            recover_orphaned_delegations,
            register_boredom_delegation_spawner,
        )

        return (
            boredom_tick,
            materialize_boredom_initiative,
            register_boredom_delegation_spawner,
            apply_boredom_delegation_result,
            recover_orphaned_delegations,
        )

    async def _spawn_boredom_delegation(self, task: str, label: str, executors: list[str]) -> str:
        """Spawn the boredom idea-generation delegation using the configured fallback chain."""
        from nanobot.agent.subagent import PeakHoursSpawnBlockedError

        if self._spawn_boredom_delegation_cb is None:
            raise RuntimeError("boredom delegation spawn callback is not configured")
        last_error: Exception | None = None
        for executor in executors:
            try:
                return await self._spawn_boredom_delegation_cb(task, label, executor)
            except PeakHoursSpawnBlockedError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("Boredom delegation spawn failed for {}: {}", executor, exc)
        raise RuntimeError(f"boredom delegation spawn failed for {executors}") from last_error

    def _spawn_boredom_delegation_from_thread(
        self, task: str, label: str, executors: list[str]
    ) -> str:
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

    def _is_boredom_disabled(self) -> bool:
        """Check if boredom engine is permanently disabled via state flag."""
        try:
            state_path = self._boredom_state_path(self.workspace)
            if not state_path.exists():
                return False
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return bool(data.get("disabled", False))
        except Exception:
            return False

    @staticmethod
    def _peak_hours_instruction() -> str:
        if is_zai_peak():
            return _ZAI_PEAK_HEARTBEAT_INSTRUCTION
        return ""

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
            logger.debug(
                "Boredom delegation completed successfully; deferring state reset to next tick"
            )
            return

        fail_hook = getattr(orchestrator, "fail_safe_execution", None)
        if callable(fail_hook):
            fail_hook(reason=reason or "delegation_failed")
            return

        state.mode = BoredomMode.IDLE_COUNTING
        state.empty_tick_count = 0
        state.active_initiative = None
        store.save(state)

    def _process_boredom_delegation_payload(
        self, initiative: dict[str, Any]
    ) -> dict[str, Any] | None:
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
            ideas_generated=len(payload.get("candidates", []))
            if isinstance(payload.get("candidates"), list)
            else 0,
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
                recover_orphaned_delegations,
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
            orchestrator = BoredomOrchestrator(
                store=BoredomStore(self._boredom_state_path(self.workspace))
            )
            state = await asyncio.to_thread(orchestrator.get_state)
            initiative = state.active_initiative
            if state.mode != BoredomMode.BORED_DELEGATING or initiative is None:
                # No active delegation in-flight — recover any orphaned files.
                try:
                    recovery = await asyncio.to_thread(
                        recover_orphaned_delegations,
                        workspace_root=self.workspace,
                    )
                    if recovery.files_unprocessed > 0:
                        logger.info(
                            "Boredom recovery: {} files scanned, {} unprocessed, {} candidates, {} tasks created, exhausted={}",
                            recovery.files_scanned,
                            recovery.files_unprocessed,
                            recovery.candidates_after_filter,
                            recovery.tasks_created,
                            recovery.exhausted,
                        )
                except Exception as recovery_exc:
                    logger.debug("Boredom orphaned recovery failed: {}", recovery_exc)
                return
            if not initiative.task_id:
                return

            output_path = boredom_delegation_output_path(
                initiative.task_id, workspace_root=self.workspace
            )
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
            (
                boredom_tick,
                materialize_boredom_initiative,
                register_boredom_delegation_spawner,
                _apply_boredom,
                _recover_orphaned,
            ) = self._load_boredom_hooks()
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
                state,
                action,
                ideas,
            )
            _emit_heartbeat_event("heartbeat.boredom_tick", data)

            # Chat notification on non-trivial boredom actions
            if action not in ("wait", "wait_for_delegation") and self.on_notify:
                initiative = result.initiative
                idea = getattr(initiative, "idea", None) if initiative else None
                msg = f"🧩 Boredom: {action}"
                if idea:
                    msg += f" — {idea[:80]}"
                if initiative_data is not None:
                    msg += " → delegating"
                try:
                    await self.on_notify(msg)
                except Exception:
                    logger.debug("Boredom tick notification failed")
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
        prompt += self._peak_hours_instruction()
        prompt += build_boredom_prompt_context(self.workspace)
        logger.info("Boredom: delegating initiative '{}' to subagent", title)
        if self.on_notify:
            try:
                await self.on_notify(f"🧩 Boredom: delegating '{title}' → subagent")
            except Exception:
                pass
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
            if self.on_notify:
                try:
                    await self.on_notify(f"🧩 Boredom: delegation '{title}' completed")
                except Exception:
                    pass
        except asyncio.TimeoutError:
            logger.warning("Boredom: initiative delegation timed out: {}", title)
            try:
                await asyncio.to_thread(self._notify_boredom_completion, False, "timeout")
            except Exception as e:
                logger.debug("Boredom completion notification failed after timeout: {}", e)
            if self.on_notify:
                try:
                    await self.on_notify(f"🧩 Boredom: delegation '{title}' timed out")
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Boredom: initiative delegation failed: {}", e)
            try:
                await asyncio.to_thread(self._notify_boredom_completion, False, str(e))
            except Exception as notify_error:
                logger.debug("Boredom completion notification failed after error: {}", notify_error)
            if self.on_notify:
                try:
                    await self.on_notify(f"🧩 Boredom: delegation '{title}' failed — {e}")
                except Exception:
                    pass

    async def _run_tick(self, *, force: bool = False) -> dict[str, Any]:
        from nanobot.utils.evaluator import evaluate_response

        # --- Maintenance: incremental memory reindex ---
        # Reindexing is handled by nanobot-reindex.timer (systemd) as a
        # separate process — do not run it here to avoid blocking the event loop.
        self._tick_count += 1

        # --- Boredom engine tick (runs independently of LLM decision) ---
        # CRITICAL: Permanently disabled via state flag - check before any boredom logic runs
        # See: /root/.nanobot/workspace/data/boredom_state.json -> "disabled": true
        if self._is_boredom_disabled():
            logger.debug("Boredom engine DISABLED - skipping all boredom logic")
        else:
            await self._process_boredom_delegations()
            await self._run_boredom_tick()
        health_state = await asyncio.to_thread(_refresh_health_state, self.workspace)

        content = self._read_heartbeat_file()
        if not content:
            return {
                "action": "skip",
                "tasks": "",
                "result": "HEARTBEAT.md missing",
                "review_decisions": [],
            }

        logger.info("Heartbeat: checking for tasks...")
        _emit_heartbeat_event("heartbeat.checking")
        if not force and self.is_user_active and self.is_user_active():
            logger.info("Heartbeat: user session active, skipping LLM call")
            _emit_heartbeat_event("heartbeat.nothing_to_report")
            return {"action": "skip", "tasks": "", "result": "", "review_decisions": []}

        # --- Phase 1: programmatic task check (no LLM needed) ---
        actionable, incomplete = self._check_actionable_tasks()
        if incomplete:
            logger.info(
                "Heartbeat: {} incomplete tasks (missing ## Files/## Acceptance Criteria): {}",
                len(incomplete),
                [t["id"] for t in incomplete],
            )
        if actionable:
            logger.info("Heartbeat: {} actionable tasks found, delegating...", len(actionable))
            _emit_heartbeat_event("heartbeat.executing", {"tasks_preview": str(actionable)[:200]})
            tasks_str = (
                content + _health_tick_context(health_state) + self._peak_hours_instruction()
            )
            if self.on_execute:
                try:
                    response = await asyncio.wait_for(self.on_execute(tasks_str), timeout=600)
                except asyncio.TimeoutError:
                    logger.error("Heartbeat: on_execute timed out")
                    response = None
                if response:
                    should_notify = await evaluate_response(
                        response, tasks_str, self.provider, self.model
                    )
                    if should_notify and self.on_notify:
                        await self.on_notify(response)
            return {"action": "run", "tasks": str(actionable), "result": "", "review_decisions": []}

        # --- Review gate: spawn reviewer for tasks awaiting independent verification ---
        # Done AFTER any LLM call to avoid rate-limit contention on shared executor.
        tasks_in_review = await self._fetch_tasks_in_review()
        # Prune stale entries from active-review tracking
        review_ids_now = {t["id"] for t in tasks_in_review}
        self._active_review_task_ids &= review_ids_now
        if tasks_in_review:
            logger.info(
                "Heartbeat: {} task(s) in review status, spawning reviewer(s)", len(tasks_in_review)
            )
            _emit_heartbeat_event(
                "heartbeat.review_gate", {"tasks": [t["id"] for t in tasks_in_review]}
            )
            # Brief delay to avoid rate-limit collision with heartbeat's own LLM call
            await asyncio.sleep(5)
            for task_info in tasks_in_review:
                await self._spawn_reviewer(task_info)

        # --- Review pending subagent results (LLM needed) ---
        pending_review = await self._fetch_pending_review()
        review_decisions = []
        closed_reviews: list[dict] = []
        if pending_review:
            logger.info("Heartbeat: {} tasks pending review", len(pending_review))
            review_decisions = await self._review_pending(pending_review)
            closed_reviews = await self._process_reviews(review_decisions)

        logger.info("Heartbeat: nothing to report")
        _emit_heartbeat_event("heartbeat.nothing_to_report")
        return {"action": "skip", "tasks": "", "result": "", "review_decisions": closed_reviews}

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
            await self.on_tick_report(
                result["action"], result["tasks"], result["review_decisions"] or None
            )

    # ------------------------------------------------------------------
    # Trigger (manual)
    # ------------------------------------------------------------------

    async def trigger_now(self) -> dict:
        """Manually trigger heartbeat, return structured result.

        Skips the is_user_active() check — the user explicitly requested a tick.
        """
        if self.on_tick_report:
            await self.on_tick_report("start", "", None)
        try:
            result = await self._run_tick(force=True)
        finally:
            await self._deliver_pending()

        if self.on_tick_report:
            await self.on_tick_report(
                result["action"], result["tasks"], result["review_decisions"] or None
            )
        return result
