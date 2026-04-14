import asyncio
from types import SimpleNamespace

import pytest

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
async def test_decide_returns_skip_when_no_tool_call(tmp_path) -> None:
    provider = DummyProvider([LLMResponse(content="no tool call", tool_calls=[])])
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    action, tasks, review_decisions = await service._decide("heartbeat content")
    assert action == "skip"
    assert tasks == ""
    assert review_decisions == []


@pytest.mark.asyncio
async def test_trigger_now_executes_when_decision_is_run(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check open tasks"},
                )
            ],
        )
    ])

    called_with: list[str] = []

    async def _on_execute(tasks: str) -> str:
        called_with.append(tasks)
        return "done"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    result = await service.trigger_now()
    assert result["action"] == "run"
    assert result["result"] == "done"
    assert called_with == ["check open tasks"]


@pytest.mark.asyncio
async def test_trigger_now_returns_none_when_decision_is_skip(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "skip"},
                )
            ],
        )
    ])

    async def _on_execute(tasks: str) -> str:
        return tasks

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    result = await service.trigger_now()
    assert result["action"] == "skip"


@pytest.mark.asyncio
async def test_tick_notifies_when_evaluator_says_yes(tmp_path, monkeypatch) -> None:
    """Phase 1 run -> Phase 2 execute -> Phase 3 evaluate=notify -> on_notify called."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check deployments", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check deployments"},
                )
            ],
        ),
    ])

    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "deployment failed on staging"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    async def _eval_notify(*a, **kw):
        return True

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _eval_notify)

    await service._tick()
    assert executed == ["check deployments"]
    assert notified == ["deployment failed on staging"]


@pytest.mark.asyncio
async def test_tick_suppresses_when_evaluator_says_no(tmp_path, monkeypatch) -> None:
    """Phase 1 run -> Phase 2 execute -> Phase 3 evaluate=silent -> on_notify NOT called."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check status", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check status"},
                )
            ],
        ),
    ])

    executed: list[str] = []
    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "everything is fine, no issues"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    async def _eval_silent(*a, **kw):
        return False

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _eval_silent)

    await service._tick()
    assert executed == ["check status"]
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
async def test_decide_retries_transient_error_then_succeeds(tmp_path, monkeypatch) -> None:
    provider = DummyProvider([
        LLMResponse(content="429 rate limit", finish_reason="error"),
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check open tasks"},
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

    action, tasks, review_decisions = await service._decide("heartbeat content")

    assert action == "run"
    assert tasks == "check open tasks"
    assert review_decisions == []
    assert provider.calls == 2
    assert delays == [1]


@pytest.mark.asyncio
async def test_decide_prompt_includes_current_time(tmp_path) -> None:
    """Phase 1 user prompt must contain current time so the LLM can judge task urgency."""

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
                        arguments={"action": "skip"},
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

    await service._decide("- [ ] check servers at 10:00 UTC")

    user_msg = captured_messages[1]
    assert user_msg["role"] == "user"
    assert "Current Time:" in user_msg["content"]


@pytest.mark.asyncio
async def test_trigger_now_returns_structured_result(tmp_path) -> None:
    """trigger_now always returns a dict with action, tasks, and result keys."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    # --- skip path ---
    provider_skip = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(id="hb_1", name="heartbeat",
                                arguments={"action": "skip"}),
            ],
        )
    ])
    service_skip = HeartbeatService(
        workspace=tmp_path, provider=provider_skip,
        model="openai/gpt-4o-mini",
    )
    result_skip = await service_skip.trigger_now()
    assert isinstance(result_skip, dict)
    assert result_skip["action"] == "skip"
    assert result_skip["result"] == ""

    # --- run path ---
    provider_run = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(id="hb_1", name="heartbeat",
                                arguments={"action": "run", "tasks": "fix bug"}),
            ],
        )
    ])
    async def _exec(tasks: str) -> str:
        return "fixed"

    service_run = HeartbeatService(
        workspace=tmp_path, provider=provider_run,
        model="openai/gpt-4o-mini", on_execute=_exec,
    )
    result_run = await service_run.trigger_now()
    assert result_run["action"] == "run"
    assert result_run["tasks"] == "fix bug"
    assert result_run["result"] == "fixed"

    # --- missing HEARTBEAT.md ---
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    service_empty = HeartbeatService(
        workspace=empty_dir, provider=provider_skip,
        model="openai/gpt-4o-mini",
    )
    result_empty = await service_empty.trigger_now()
    assert result_empty["action"] == "skip"
    assert "missing" in result_empty["result"]


@pytest.mark.asyncio
async def test_trigger_now_notifies_when_evaluator_says_yes(tmp_path, monkeypatch) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check deployments", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check deployments"},
                )
            ],
        ),
    ])

    notified: list[str] = []

    async def _on_execute(tasks: str) -> str:
        assert tasks == "check deployments"
        return "deployment failed on staging"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    async def _eval_notify(*a, **kw):
        return True

    monkeypatch.setattr("nanobot.utils.evaluator.evaluate_response", _eval_notify)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
    )

    result = await service.trigger_now()
    assert result["action"] == "run"
    assert result["result"] == "deployment failed on staging"
    assert notified == ["deployment failed on staging"]


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
