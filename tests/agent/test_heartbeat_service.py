import asyncio
from types import SimpleNamespace

import pytest

# Body text that makes a task "complete" (has ## Files + ## Acceptance Criteria)
_COMPLETE_BODY = "## Description\ntest\n\n## Files\nfoo.py\n\n## Acceptance Criteria\n- passes"

from nanobot.heartbeat.service import HeartbeatService
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class DummyProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


@pytest.fixture(autouse=True)
def stub_pending_review(monkeypatch) -> None:
    async def _empty_pending_review() -> list[dict[str, str]]:
        return []

    monkeypatch.setattr(HeartbeatService, "_fetch_pending_review", staticmethod(_empty_pending_review))


@pytest.fixture(autouse=True)
def stub_tasks_in_review(monkeypatch, request) -> None:
    """Prevent real subprocess calls to nanobot-tasks in non-review-gate tests."""
    if "review_gate" in request.node.name or "spawn_reviewer" in request.node.name:
        return

    async def _empty_tasks_in_review() -> list[dict[str, str]]:
        return []

    monkeypatch.setattr(HeartbeatService, "_fetch_tasks_in_review", staticmethod(_empty_tasks_in_review))


@pytest.fixture(autouse=True)
def stub_boredom_runtime_for_non_boredom_tests(monkeypatch, request) -> None:
    if "boredom" in request.node.name:
        return

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(HeartbeatService, "_run_boredom_tick", _noop)
    monkeypatch.setattr(HeartbeatService, "_process_boredom_delegations", _noop)


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path) -> None:
    provider = DummyProvider([])

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        interval_s=9999,
        enabled=True,
    )

    await service.start()
    first_task = service._task
    await service.start()

    assert service._task is first_task

    service.stop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_check_actionable_tasks_finds_open_unblocked(tmp_path, monkeypatch) -> None:
    """_check_actionable_tasks returns open/unblocked tasks only."""
    fake_tasks = [
        SimpleNamespace(id="t1", status="open", title="Fix bug", blocked_reason=None, body=_COMPLETE_BODY),
        SimpleNamespace(id="t2", status="blocked", title="Old audit", blocked_reason="user decision", body=""),
        SimpleNamespace(id="t3", status="open", title="Also open", blocked_reason="", body=_COMPLETE_BODY),
    ]

    def _fake_list(self, scope):
        return fake_tasks

    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", _fake_list)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
    )

    actionable, incomplete = service._check_actionable_tasks()
    assert len(actionable) == 2
    assert len(incomplete) == 0
    assert actionable[0]["id"] == "t1"
    assert actionable[1]["id"] == "t3"


@pytest.mark.asyncio
async def test_check_actionable_tasks_sorted_by_priority_then_complexity(tmp_path, monkeypatch) -> None:
    """Actionable tasks are sorted: critical > high > normal > low, then s < m < l < xl."""
    fake_tasks = [
        SimpleNamespace(id="t_low", status="open", title="Low prio", blocked_reason=None, body=_COMPLETE_BODY, priority="low", complexity="m"),
        SimpleNamespace(id="t_high", status="open", title="High prio", blocked_reason=None, body=_COMPLETE_BODY, priority="high", complexity="m"),
        SimpleNamespace(id="t_normal_l", status="open", title="Normal prio L", blocked_reason=None, body=_COMPLETE_BODY, priority="normal", complexity="l"),
        SimpleNamespace(id="t_normal_s", status="open", title="Normal prio S", blocked_reason=None, body=_COMPLETE_BODY, priority="normal", complexity="s"),
        SimpleNamespace(id="t_critical", status="open", title="Critical", blocked_reason=None, body=_COMPLETE_BODY, priority="critical", complexity="xl"),
    ]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    service = HeartbeatService(workspace=tmp_path, provider=DummyProvider([]), model="test-model")
    actionable, incomplete = service._check_actionable_tasks()

    ids = [t["id"] for t in actionable]
    assert ids == ["t_critical", "t_high", "t_normal_s", "t_normal_l", "t_low"]
    assert incomplete == []


@pytest.mark.asyncio
async def test_check_actionable_tasks_missing_priority_uses_normal_default(tmp_path, monkeypatch) -> None:
    """Tasks without priority/complexity attributes sort as 'normal'/'m'."""
    fake_tasks = [
        SimpleNamespace(id="t_no_attr", status="open", title="No attrs", blocked_reason=None, body=_COMPLETE_BODY),
        SimpleNamespace(id="t_high", status="open", title="High", blocked_reason=None, body=_COMPLETE_BODY, priority="high", complexity="m"),
    ]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    service = HeartbeatService(workspace=tmp_path, provider=DummyProvider([]), model="test-model")
    actionable, _ = service._check_actionable_tasks()

    assert [t["id"] for t in actionable] == ["t_high", "t_no_attr"]


@pytest.mark.asyncio
async def test_check_actionable_tasks_ties_preserve_input_order(tmp_path, monkeypatch) -> None:
    """Tasks with identical priority+complexity preserve their relative input order."""
    fake_tasks = [
        SimpleNamespace(id="t1", status="open", title="First", blocked_reason=None, body=_COMPLETE_BODY, priority="normal", complexity="m"),
        SimpleNamespace(id="t2", status="open", title="Second", blocked_reason=None, body=_COMPLETE_BODY, priority="normal", complexity="m"),
        SimpleNamespace(id="t3", status="open", title="Third", blocked_reason=None, body=_COMPLETE_BODY, priority="normal", complexity="m"),
    ]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    service = HeartbeatService(workspace=tmp_path, provider=DummyProvider([]), model="test-model")
    actionable, _ = service._check_actionable_tasks()

    assert [t["id"] for t in actionable] == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_check_actionable_tasks_returns_empty_when_no_tasks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: [])

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
    )

    assert service._check_actionable_tasks() == ([], [])


@pytest.mark.asyncio
async def test_trigger_now_executes_when_actionable_tasks_exist(tmp_path, monkeypatch) -> None:
    """With open/unblocked tasks, trigger_now runs Phase 2 without LLM."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    fake_tasks = [SimpleNamespace(id="t1", status="open", title="Fix bug", blocked_reason=None, body=_COMPLETE_BODY)]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    called_with: list[str] = []

    async def _on_execute(tasks: str) -> str:
        called_with.append(tasks)
        return "done"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),  # no LLM calls needed
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    result = await service.trigger_now()
    assert result["action"] == "run"
    assert result["result"] == ""
    assert len(called_with) == 1
    assert "do thing" in called_with[0]


@pytest.mark.asyncio
async def test_trigger_now_skips_when_no_actionable_tasks(tmp_path, monkeypatch) -> None:
    """With no open/unblocked tasks, trigger_now returns skip without LLM."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: [])

    async def _on_execute(tasks: str) -> str:
        raise AssertionError("should not execute when no actionable tasks")

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    result = await service.trigger_now()
    assert result["action"] == "skip"


@pytest.mark.asyncio
async def test_tick_notifies_when_evaluator_says_yes(tmp_path, monkeypatch) -> None:
    """Actionable tasks -> Phase 2 execute -> evaluate=notify -> on_notify called."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check deployments", encoding="utf-8")

    fake_tasks = [SimpleNamespace(id="t1", status="open", title="Fix bug", blocked_reason=None, body=_COMPLETE_BODY)]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "deployment failed on staging"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    async def _eval_notify(*a, **kw):
        return True

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _eval_notify)

    await service._tick()
    assert len(executed) == 1
    assert "check deployments" in executed[0]
    assert notified == ["deployment failed on staging"]


@pytest.mark.asyncio
async def test_tick_suppresses_when_evaluator_says_no(tmp_path, monkeypatch) -> None:
    """Actionable tasks -> Phase 2 execute -> evaluate=silent -> on_notify NOT called."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check status", encoding="utf-8")

    fake_tasks = [SimpleNamespace(id="t1", status="open", title="Fix bug", blocked_reason=None, body=_COMPLETE_BODY)]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "everything is fine, no issues"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    async def _eval_silent(*a, **kw):
        return False

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _eval_silent)

    await service._tick()
    assert len(executed) == 1
    assert "check status" in executed[0]
    assert notified == []


@pytest.mark.asyncio
async def test_run_boredom_tick_calls_workspace_in_process(tmp_path, monkeypatch) -> None:
    provider = DummyProvider([])
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    captured: dict[str, object] = {}
    events: list[tuple[str, dict | None]] = []

    async def _fake_spawn(task: str, label: str, executor: str) -> str:
        captured["spawn_task"] = task
        captured["spawn_label"] = label
        captured["spawn_executor"] = executor
        return "runtime-123"

    def _fake_boredom_tick(task_snapshot, **kwargs):
        captured["snapshot"] = task_snapshot
        captured["workspace_root"] = kwargs["workspace_root"]
        return SimpleNamespace(
            state_before="idle_counting",
            state_after="bored_delegating",
            action_taken="spawn_delegation",
            ideas_accepted=0,
            ideas_generated=0,
            next_action="wait_for_delegation",
            error=None,
            duration_ms=12,
            open_tasks=len(task_snapshot),
            initiative=None,
        )

    def _fake_materialize(result, *, workspace_root):
        captured["materialized_workspace"] = workspace_root
        captured["materialized_result"] = result
        return None

    monkeypatch.setattr(
        HeartbeatService,
        "_load_boredom_hooks",
        staticmethod(lambda: (_fake_boredom_tick, _fake_materialize, lambda cb: captured.setdefault("registered_spawner", cb), object())),
    )
    service.set_spawn_callback(_fake_spawn)

    async def _direct_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _direct_to_thread)
    monkeypatch.setattr(
        "nanobot.heartbeat.service._emit_heartbeat_event",
        lambda event_type, data=None: events.append((event_type, data)),
    )

    await service._run_boredom_tick()
    await asyncio.sleep(0)

    assert captured["snapshot"] == []
    assert captured["workspace_root"] == tmp_path
    assert callable(captured["registered_spawner"])
    # When next_action="wait_for_delegation", materialize is skipped (delegation still in-flight)
    assert "materialized_workspace" not in captured
    assert provider.calls == 0
    assert events == [
        (
            "heartbeat.boredom_tick",
            {
                "state_before": "idle_counting",
                "state_after": "bored_delegating",
                "action_taken": "spawn_delegation",
                "ideas_accepted": 0,
                "ideas_generated": 0,
                "next_action": "wait_for_delegation",
                "error": None,
                "duration_ms": 12,
                "open_tasks": 0,
            },
        )
    ]


@pytest.mark.asyncio
async def test_run_boredom_tick_does_not_block_on_background_spawn(tmp_path, monkeypatch) -> None:
    provider = DummyProvider([])
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    def _fake_boredom_tick(task_snapshot, **kwargs):
        return SimpleNamespace(
            state_before="idle_counting",
            state_after="bored_delegating",
            action_taken="spawn_delegation",
            ideas_accepted=0,
            ideas_generated=0,
            next_action="wait_for_delegation",
            error=None,
            duration_ms=5,
            open_tasks=len(task_snapshot),
            initiative=None,
        )

    monkeypatch.setattr(
        HeartbeatService,
        "_load_boredom_hooks",
        staticmethod(lambda: (_fake_boredom_tick, lambda result, *, workspace_root: None, lambda cb: None, object())),
    )

    async def _direct_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _direct_to_thread)

    start = asyncio.get_running_loop().time()
    await service._run_boredom_tick()
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_review_pending_returns_empty_when_no_tool_call(tmp_path) -> None:
    """_review_pending returns [] when LLM response has no tool calls."""
    provider = DummyProvider([LLMResponse(content="no tool call", tool_calls=[])])
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    review = await service._review_pending([{"id": "t1", "title": "Test task"}])
    assert review == []


@pytest.mark.asyncio
async def test_review_pending_retries_transient_error_then_succeeds(tmp_path, monkeypatch) -> None:
    """_review_pending retries on transient LLM errors."""
    provider = DummyProvider([
        LLMResponse(content="429 rate limit", finish_reason="error"),
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"review_decision": [{"task_id": "t1", "verdict": "done", "note": "success"}]},
                )
            ],
        ),
    ])

    delays: list[int] = []

    async def _fake_sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    review = await service._review_pending([{"id": "t1", "title": "Test task"}])

    assert len(review) == 1
    assert review[0]["verdict"] == "done"
    assert provider.calls == 2
    assert delays == [1]


@pytest.mark.asyncio
async def test_review_pending_prompt_includes_current_time(tmp_path) -> None:
    """_review_pending user prompt must contain current time for urgency judgment."""

    captured_messages: list[dict] = []

    class CapturingProvider(LLMProvider):
        async def chat(self, *, messages=None, **kwargs) -> LLMResponse:
            if messages:
                captured_messages.extend(messages)
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="hb_1", name="heartbeat",
                        arguments={"review_decision": []},
                    )
                ],
            )

        def get_default_model(self) -> str:
            return "test-model"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=CapturingProvider(),
        model="test-model",
    )

    await service._review_pending([{"id": "t1", "title": "Check servers at 10:00"}])

    user_msg = captured_messages[1]
    assert user_msg["role"] == "user"
    assert "Current Time:" in user_msg["content"]


@pytest.mark.asyncio
async def test_trigger_now_returns_structured_result(tmp_path, monkeypatch) -> None:
    """trigger_now always returns a dict with action, tasks, and result keys."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    # --- skip path (no actionable tasks) ---
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: [])
    service_skip = HeartbeatService(
        workspace=tmp_path, provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
    )
    result_skip = await service_skip.trigger_now()
    assert isinstance(result_skip, dict)
    assert result_skip["action"] == "skip"
    assert result_skip["result"] == ""

    # --- run path (actionable tasks exist) ---
    fake_tasks = [SimpleNamespace(id="t1", status="open", title="fix bug", blocked_reason=None, body=_COMPLETE_BODY)]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    async def _exec(tasks: str) -> str:
        return "fixed"

    service_run = HeartbeatService(
        workspace=tmp_path, provider=DummyProvider([]),
        model="openai/gpt-4o-mini", on_execute=_exec,
    )
    result_run = await service_run.trigger_now()
    assert result_run["action"] == "run"
    assert result_run["result"] == ""

    # --- missing HEARTBEAT.md ---
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    service_empty = HeartbeatService(
        workspace=empty_dir, provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
    )
    result_empty = await service_empty.trigger_now()
    assert result_empty["action"] == "skip"
    assert "missing" in result_empty["result"]


@pytest.mark.asyncio
async def test_trigger_now_notifies_when_evaluator_says_yes(tmp_path, monkeypatch) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check deployments", encoding="utf-8")

    fake_tasks = [SimpleNamespace(id="t1", status="open", title="fix bug", blocked_reason=None, body=_COMPLETE_BODY)]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        return "deployment failed on staging"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    async def _eval_notify(*a, **kw):
        return True

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _eval_notify)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    result = await service.trigger_now()
    assert result["action"] == "run"
    assert result["result"] == ""
    assert notified == ["deployment failed on staging"]


@pytest.mark.asyncio
async def test_trigger_now_injects_zai_peak_instruction_into_execute_prompt(tmp_path, monkeypatch) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check status", encoding="utf-8")

    fake_tasks = [SimpleNamespace(id="t1", status="open", title="fix bug", blocked_reason=None, body=_COMPLETE_BODY)]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    prompts: list[str] = []

    async def _on_execute(tasks: str) -> str:
        prompts.append(tasks)
        return "done"

    async def _eval_silent(*a, **kw):
        return False

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _eval_silent)
    monkeypatch.setattr("nanobot.heartbeat.service.is_zai_peak", lambda: True)
    monkeypatch.setattr("nanobot.heartbeat.service._refresh_health_state", lambda workspace: None)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    await service.trigger_now()

    assert len(prompts) == 1
    assert "ZAI peak hours active — model is unstable, minimize tool calls, do not spawn claude-zai subagents" in prompts[0]


@pytest.mark.asyncio
async def test_trigger_now_omits_zai_peak_instruction_off_peak(tmp_path, monkeypatch) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check status", encoding="utf-8")

    fake_tasks = [SimpleNamespace(id="t1", status="open", title="fix bug", blocked_reason=None, body=_COMPLETE_BODY)]
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: fake_tasks)

    prompts: list[str] = []

    async def _on_execute(tasks: str) -> str:
        prompts.append(tasks)
        return "done"

    async def _eval_silent(*a, **kw):
        return False

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _eval_silent)
    monkeypatch.setattr("nanobot.heartbeat.service.is_zai_peak", lambda: False)
    monkeypatch.setattr("nanobot.heartbeat.service._refresh_health_state", lambda workspace: None)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    await service.trigger_now()

    assert len(prompts) == 1
    assert "do not spawn claude-zai subagents" not in prompts[0]


@pytest.mark.asyncio
async def test_trigger_now_skips_when_user_is_active(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider([])

    async def _on_execute(tasks: str) -> str:
        raise AssertionError("trigger_now should not execute while the user is active")

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        is_user_active=lambda: True,
    )

    result = await service.trigger_now()
    assert result == {"action": "skip", "tasks": "", "result": "", "review_decisions": []}
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_process_boredom_delegations_updates_state_with_results(tmp_path, monkeypatch) -> None:
    from nanobot_workspace.proactive.boredom.models import (
        BoredomInitiative,
        BoredomMode,
        BoredomState,
        RiskClass,
    )
    from nanobot_workspace.proactive.boredom.orchestrator import BoredomOrchestrator
    from nanobot_workspace.proactive.boredom.store import BoredomStore

    provider = DummyProvider([])
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test-model")
    state_path = tmp_path / "data" / "boredom_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    orchestrator = BoredomOrchestrator(store=BoredomStore(state_path))
    orchestrator._store.save(  # noqa: SLF001
        BoredomState(
            mode=BoredomMode.BORED_DELEGATING,
            active_initiative=BoredomInitiative(
                idea=None,
                risk_class=RiskClass.SAFE,
                status="pending",
                task_id="boredom_1",
                runtime_task_id="runtime-123",
            ),
        )
    )
    output_dir = tmp_path / "data" / "boredom_delegations"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "boredom_1.json").write_text(
        '{"candidates":[{"title":"Add boredom tests","category":"tests","target_paths":["tests/test_x.py"],'
        '"description":"Add coverage","why_safe":"tests only","estimated_checks":["pytest tests/"],'
        '"novelty_basis":"new","risk_class":"safe"}]}',
        encoding="utf-8",
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "task_delivery.jsonl").write_text(
        '{"task_id":"runtime-123","success":true,"summary":"done"}\n',
        encoding="utf-8",
    )

    async def _direct_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _direct_to_thread)

    await service._process_boredom_delegations()

    state = orchestrator.get_state()
    # After successful materialization, _process_boredom_delegations resets to idle_counting
    assert state.mode.value == "idle_counting"


@pytest.mark.asyncio
async def test_boredom_spawn_callback_uses_runtime_fallback_chain(tmp_path) -> None:
    provider = DummyProvider([])
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test-model")
    attempts: list[str] = []

    async def _fallback(task: str, label: str, executor: str) -> str:
        attempts.append(executor)
        if executor != "openrouter":
            raise RuntimeError(f"{executor} unavailable")
        return "runtime-1"

    service.set_spawn_callback(_fallback)
    task_id = await service._spawn_boredom_delegation("task body", "boredom-idea-generation", ["glm-5.1", "codex-5.4", "openrouter"])

    assert task_id == "runtime-1"
    assert attempts == ["glm-5.1", "codex-5.4", "openrouter"]


@pytest.mark.asyncio
async def test_boredom_spawn_block_does_not_fallback_to_next_executor(tmp_path) -> None:
    from nanobot.agent.subagent import PeakHoursSpawnBlockedError

    provider = DummyProvider([])
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test-model")
    attempts: list[str] = []

    async def _blocked(task: str, label: str, executor: str) -> str:
        attempts.append(executor)
        if executor == "claude-zai":
            raise PeakHoursSpawnBlockedError("blocked by ZAI peak")
        return "runtime-1"

    service.set_spawn_callback(_blocked)

    with pytest.raises(PeakHoursSpawnBlockedError):
        await service._spawn_boredom_delegation(
            "task body",
            "boredom-idea-generation",
            ["claude-zai", "openrouter"],
        )

    assert attempts == ["claude-zai"]


# ---------------------------------------------------------------------------
# Boredom delegation tests
# ---------------------------------------------------------------------------

_SAFE_INITIATIVE = {
    "title": "Add doc tests",
    "description": "Document the boredom module",
    "category": "docs",
    "target_paths": ["docs/boredom.md"],
    "acceptance_criteria": ["ruff check passes"],
    "risk_class": "safe",
    "source": "boredom",
    "task_id": "TASK-9",
    "correlation_id": "boredom_123",
}


@pytest.mark.asyncio
async def test_delegate_boredom_initiative_calls_on_execute_for_safe_initiative(
    tmp_path, monkeypatch
) -> None:
    """SAFE initiative with no dirty overlap delegates to on_execute."""
    import sys
    import types

    # Inject a lightweight git_ops stub so the inline import inside the method resolves.
    git_ops_stub = types.ModuleType("nanobot_workspace.proactive.boredom.git_ops")
    git_ops_stub.list_dirty_paths = lambda _root: []
    git_ops_stub.find_dirty_overlap = lambda targets, dirty: []
    for name in [
        "nanobot_workspace",
        "nanobot_workspace.proactive",
        "nanobot_workspace.proactive.boredom",
        "nanobot_workspace.proactive.boredom.git_ops",
    ]:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nanobot_workspace.proactive.boredom.git_ops", git_ops_stub)

    executed: list[str] = []

    async def _on_execute(prompt: str) -> str:
        executed.append(prompt)
        return "done"

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _fake_to_thread)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
        on_execute=_on_execute,
    )

    await service._delegate_boredom_initiative(_SAFE_INITIATIVE)

    assert len(executed) == 1
    assert "Add doc tests" in executed[0]
    assert "docs" in executed[0]


@pytest.mark.asyncio
async def test_delegate_boredom_initiative_skips_needs_review(tmp_path) -> None:
    """NEEDS_REVIEW initiative is not delegated regardless of other conditions."""
    executed: list[str] = []

    async def _on_execute(prompt: str) -> str:
        executed.append(prompt)
        return "done"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
        on_execute=_on_execute,
    )

    await service._delegate_boredom_initiative({**_SAFE_INITIATIVE, "risk_class": "needs_review"})

    assert executed == []


@pytest.mark.asyncio
async def test_delegate_boredom_initiative_skips_on_dirty_overlap(
    tmp_path, monkeypatch
) -> None:
    """SAFE initiative with dirty overlap in target paths is not delegated."""
    import sys
    import types

    git_ops_stub = types.ModuleType("nanobot_workspace.proactive.boredom.git_ops")
    git_ops_stub.list_dirty_paths = lambda _root: ["docs/boredom.md"]
    git_ops_stub.find_dirty_overlap = lambda targets, dirty: [p for p in dirty if p in targets]
    for name in [
        "nanobot_workspace",
        "nanobot_workspace.proactive",
        "nanobot_workspace.proactive.boredom",
        "nanobot_workspace.proactive.boredom.git_ops",
    ]:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nanobot_workspace.proactive.boredom.git_ops", git_ops_stub)

    executed: list[str] = []

    async def _on_execute(prompt: str) -> str:
        executed.append(prompt)
        return "done"

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _fake_to_thread)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
        on_execute=_on_execute,
    )

    await service._delegate_boredom_initiative(_SAFE_INITIATIVE)

    assert executed == []


@pytest.mark.asyncio
async def test_delegate_boredom_initiative_emits_correlation_metadata(tmp_path, monkeypatch) -> None:
    events: list[tuple[str, dict | None]] = []

    async def _on_execute(prompt: str) -> str:
        return prompt

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    import sys
    import types

    git_ops_stub = types.ModuleType("nanobot_workspace.proactive.boredom.git_ops")
    git_ops_stub.list_dirty_paths = lambda _root: []
    git_ops_stub.find_dirty_overlap = lambda targets, dirty: []
    for name in [
        "nanobot_workspace",
        "nanobot_workspace.proactive",
        "nanobot_workspace.proactive.boredom",
        "nanobot_workspace.proactive.boredom.git_ops",
    ]:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nanobot_workspace.proactive.boredom.git_ops", git_ops_stub)
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _fake_to_thread)
    monkeypatch.setattr(
        "nanobot.heartbeat.service._emit_heartbeat_event",
        lambda event_type, data=None: events.append((event_type, data)),
    )

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
        on_execute=_on_execute,
    )

    await service._delegate_boredom_initiative(_SAFE_INITIATIVE)

    assert events == [
        (
            "heartbeat.boredom_delegated",
            {
                "title": "Add doc tests",
                "category": "docs",
                "source": "boredom",
                "task_id": "TASK-9",
                "correlation_id": "boredom_123",
                "action_taken": "delegated",
            },
        )
    ]


@pytest.mark.asyncio
async def test_delegate_boredom_initiative_processes_delegation_output(tmp_path, monkeypatch) -> None:
    import json
    import sys
    import types

    git_ops_stub = types.ModuleType("nanobot_workspace.proactive.boredom.git_ops")
    git_ops_stub.list_dirty_paths = lambda _root: []
    git_ops_stub.find_dirty_overlap = lambda targets, dirty: []
    for name in [
        "nanobot_workspace",
        "nanobot_workspace.proactive",
        "nanobot_workspace.proactive.boredom",
        "nanobot_workspace.proactive.boredom.git_ops",
    ]:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nanobot_workspace.proactive.boredom.git_ops", git_ops_stub)

    async def _on_execute(prompt: str) -> str:
        return prompt

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    class _Mode:
        def __init__(self, value: str) -> None:
            self.value = value

    transition = SimpleNamespace(
        previous_mode=_Mode("bored_delegating"),
        current_mode=_Mode("executing_safe"),
        next_action=None,
        initiative=SimpleNamespace(idea=SimpleNamespace(title="Accepted idea")),
    )
    applied: list[tuple[str, dict, object]] = []
    materialized: list[tuple[str, object]] = []
    completions: list[tuple[bool, str | None]] = []

    def _fake_apply(delegation_id: str, payload: dict, *, workspace_root):
        applied.append((delegation_id, payload, workspace_root))
        return transition

    def _fake_materialize(result, *, workspace_root):
        materialized.append((result.initiative.idea.title, workspace_root))
        return {"task_id": "task-1", "title": "Accepted idea"}

    boredom_module = types.ModuleType("nanobot_workspace.proactive.boredom.heartbeat_integration")
    boredom_module.BoredomTickResult = lambda **kwargs: SimpleNamespace(**kwargs)
    boredom_module.apply_boredom_delegation_result = _fake_apply
    boredom_module.boredom_delegation_output_path = (
        lambda delegation_id, *, workspace_root: workspace_root / "data" / "boredom_delegations" / f"{delegation_id}.json"
    )
    boredom_module.materialize_boredom_initiative = _fake_materialize
    monkeypatch.setitem(sys.modules, "nanobot_workspace.proactive.boredom.heartbeat_integration", boredom_module)
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _fake_to_thread)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
        on_execute=_on_execute,
    )
    monkeypatch.setattr(service, "_notify_boredom_completion", lambda success, reason=None: completions.append((success, reason)))

    output_path = tmp_path / "data" / "boredom_delegations" / "boredom_123.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"candidates": [{"title": "Accepted idea"}]}), encoding="utf-8")

    await service._delegate_boredom_initiative({**_SAFE_INITIATIVE, "initiative_id": "boredom_123"})

    assert applied == [("boredom_123", {"candidates": [{"title": "Accepted idea"}]}, tmp_path)]
    assert materialized == [("Accepted idea", tmp_path)]
    assert completions == [(True, None)]


@pytest.mark.asyncio
async def test_delegate_boredom_initiative_notifies_success_when_result_processing_fails(
    tmp_path, monkeypatch
) -> None:
    import sys
    import types

    git_ops_stub = types.ModuleType("nanobot_workspace.proactive.boredom.git_ops")
    git_ops_stub.list_dirty_paths = lambda _root: []
    git_ops_stub.find_dirty_overlap = lambda targets, dirty: []
    for name in [
        "nanobot_workspace",
        "nanobot_workspace.proactive",
        "nanobot_workspace.proactive.boredom",
        "nanobot_workspace.proactive.boredom.git_ops",
    ]:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nanobot_workspace.proactive.boredom.git_ops", git_ops_stub)

    async def _on_execute(prompt: str) -> str:
        return prompt

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    completions: list[tuple[bool, str | None]] = []
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _fake_to_thread)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
        on_execute=_on_execute,
    )
    monkeypatch.setattr(service, "_process_boredom_delegation_payload", lambda initiative: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(service, "_notify_boredom_completion", lambda success, reason=None: completions.append((success, reason)))

    await service._delegate_boredom_initiative(_SAFE_INITIATIVE)

    assert completions == [(True, None)]


# ------------------------------------------------------------------
# Review gate tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_gate_spawns_reviewer_for_tasks_in_review(tmp_path, monkeypatch) -> None:
    """_run_tick spawns a reviewer subagent for each task with status='review'."""
    (tmp_path / "HEARTBEAT.md").write_text("# Heartbeat", encoding="utf-8")
    (tmp_path / "skills" / "reviewer").mkdir(parents=True)
    (tmp_path / "skills" / "reviewer" / "SKILL.md").write_text("# Reviewer instructions", encoding="utf-8")

    review_tasks = [{"id": "20260415T120000_review_me", "title": "Review me"}]

    async def _fake_tasks_in_review() -> list[dict[str, str]]:
        return review_tasks

    monkeypatch.setattr(HeartbeatService, "_fetch_tasks_in_review", staticmethod(_fake_tasks_in_review))
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: [])

    spawned: list[tuple[str, str, str]] = []

    async def _fake_spawn_cb(prompt: str, label: str, executor: str) -> str:
        spawned.append((prompt, label, executor))
        return "spawned"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
    )
    service._spawn_reviewer_cb = _fake_spawn_cb

    await service._run_tick()

    assert len(spawned) == 1
    prompt, label, executor = spawned[0]
    assert "20260415T120000_review_me" in prompt
    assert "review: 20260415T120000_review_me" == label
    assert executor == "glm-turbo"
    assert "Reviewer instructions" in prompt


@pytest.mark.asyncio
async def test_review_gate_skips_spawn_when_no_tasks_in_review(tmp_path, monkeypatch) -> None:
    """_run_tick does not spawn reviewers when no tasks are in 'review' status."""
    (tmp_path / "HEARTBEAT.md").write_text("# Heartbeat", encoding="utf-8")

    async def _empty_tasks_in_review() -> list[dict[str, str]]:
        return []

    monkeypatch.setattr(HeartbeatService, "_fetch_tasks_in_review", staticmethod(_empty_tasks_in_review))
    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.list_tasks", lambda self, scope: [])

    spawned: list[tuple] = []

    async def _fake_spawn_cb(prompt: str, label: str, executor: str) -> str:
        spawned.append((prompt, label, executor))
        return "spawned"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
    )
    service._spawn_reviewer_cb = _fake_spawn_cb

    await service._run_tick()

    assert spawned == []


@pytest.mark.asyncio
async def test_spawn_reviewer_skips_without_delegation_cb(tmp_path) -> None:
    """_spawn_reviewer is a no-op when _spawn_reviewer_cb is None."""
    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
    )
    # _spawn_reviewer_cb defaults to None; should not raise
    await service._spawn_reviewer({"id": "20260415T120000_no_cb", "title": "No callback"})


@pytest.mark.asyncio
async def test_spawn_reviewer_includes_skill_instructions_when_skill_file_exists(tmp_path) -> None:
    """_spawn_reviewer includes SKILL.md content in the prompt."""
    skill_dir = tmp_path / "skills" / "reviewer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Custom review skill\nRun ruff first.", encoding="utf-8")

    spawned_prompts: list[str] = []

    async def _fake_spawn_cb(prompt: str, label: str, executor: str) -> str:
        spawned_prompts.append(prompt)
        return "spawned"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
    )
    service._spawn_reviewer_cb = _fake_spawn_cb

    await service._spawn_reviewer({"id": "20260415T120000_skill_test", "title": "Skill test"})

    assert len(spawned_prompts) == 1
    assert "Custom review skill" in spawned_prompts[0]
    assert "Run ruff first." in spawned_prompts[0]
    assert "20260415T120000_skill_test" in spawned_prompts[0]


@pytest.mark.asyncio
async def test_spawn_reviewer_gracefully_handles_missing_skill_file(tmp_path) -> None:
    """_spawn_reviewer still spawns even if SKILL.md doesn't exist."""
    spawned: list[tuple] = []

    async def _fake_spawn_cb(prompt: str, label: str, executor: str) -> str:
        spawned.append((prompt, label, executor))
        return "spawned"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider([]),
        model="test-model",
    )
    service._spawn_reviewer_cb = _fake_spawn_cb

    await service._spawn_reviewer({"id": "20260415T120000_no_skill", "title": "No skill file"})

    assert len(spawned) == 1
    assert spawned[0][2] == "glm-turbo"
