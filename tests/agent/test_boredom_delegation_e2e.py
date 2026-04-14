"""End-to-end tests for boredom delegation pipeline.

Validates the full flow:
  idle_counting -> bored -> bored_delegating (spawn idea-gen)
  -> idea-gen completes -> delegation result applied -> task created
  -> state reset to idle_counting

Tests use real workspace boredom modules (models, store, orchestrator, service)
and mock only the runtime layer (HeartbeatService._load_boredom_hooks).
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.heartbeat.service import HeartbeatService
from nanobot.providers.base import LLMProvider, LLMResponse


# ---------------------------------------------------------------------------
# Workspace imports (available via PYTHONPATH)
# ---------------------------------------------------------------------------

from nanobot_workspace.proactive.boredom.models import (
    BoredomInitiative,
    BoredomMode,
    BoredomState,
    IdeaCandidate,
    RiskClass,
)
from nanobot_workspace.proactive.boredom.store import BoredomStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


def _make_idea(title: str = "Test audit", category: str = "tests") -> IdeaCandidate:
    return IdeaCandidate(
        title=title,
        category=category,
        target_paths=["tests/test_foo.py"],
        description="Add test coverage",
        why_safe="tests only",
        estimated_checks=["pytest tests/test_foo.py"],
        novelty_basis="new",
        risk_class=RiskClass.SAFE,
    )


def _make_initiative(idea: IdeaCandidate | None = None, task_id: str = "boredom_test123", runtime_task_id: str | None = None) -> BoredomInitiative:
    return BoredomInitiative(
        idea=idea,
        risk_class=RiskClass.SAFE,
        status="pending" if idea is None else "executing",
        task_id=task_id,
        runtime_task_id=runtime_task_id,
    )


def _write_state(tmp_path: Path, mode: BoredomMode, initiative: BoredomInitiative | None = None) -> BoredomStore:
    state_path = tmp_path / "data" / "boredom_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    store = BoredomStore(state_path)
    store.save(BoredomState(mode=mode, active_initiative=initiative))
    return store


def _write_delegation_output(tmp_path: Path, delegation_id: str = "boredom_test123", candidates: list[dict] | None = None) -> Path:
    if candidates is None:
        candidates = [
            {
                "title": "Test audit",
                "category": "tests",
                "target_paths": ["tests/test_foo.py"],
                "description": "Add test coverage",
                "why_safe": "tests only",
                "estimated_checks": ["pytest tests/test_foo.py"],
                "novelty_basis": "new",
                "risk_class": "safe",
            }
        ]
    output_dir = tmp_path / "data" / "boredom_delegations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{delegation_id}.json"
    output_path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    return output_path


def _write_delivery_log(tmp_path: Path, runtime_task_id: str = "runtime-test-123", success: bool = True) -> Path:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / "task_delivery.jsonl"
    log_path.write_text(
        json.dumps({"task_id": runtime_task_id, "success": success, "summary": "done"}),
        encoding="utf-8",
    )
    return log_path


async def _sync_to_thread(func, /, *args, **kwargs):
    """Replacement for asyncio.to_thread — runs synchronously for testing."""
    return func(*args, **kwargs)


def _stub_git_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a minimal git_ops stub so inline imports resolve."""
    git_ops = types.ModuleType("nanobot_workspace.proactive.boredom.git_ops")
    git_ops.list_dirty_paths = lambda _root: []
    git_ops.find_dirty_overlap = lambda _targets, _dirty: []
    for name in [
        "nanobot_workspace",
        "nanobot_workspace.proactive",
        "nanobot_workspace.proactive.boredom",
        "nanobot_workspace.proactive.boredom.git_ops",
    ]:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nanobot_workspace.proactive.boredom.git_ops", git_ops)


def _make_spawn_delegation_result(delegation_id: str = "boredom_test123") -> SimpleNamespace:
    """Build a fake BoredomTickResult from boredom_tick() returning spawn_delegation."""
    return SimpleNamespace(
        state_before="idle_counting",
        state_after="bored_delegating",
        action_taken="spawn_delegation",
        ideas_accepted=0,
        ideas_generated=0,
        next_action="wait_for_delegation",
        error=None,
        duration_ms=5,
        open_tasks=0,
        initiative=None,
    )


def _make_skip_result() -> SimpleNamespace:
    """Build a fake BoredomTickResult from boredom_tick() returning skip."""
    return SimpleNamespace(
        state_before="idle_counting",
        state_after="idle_counting",
        action_taken="skip",
        ideas_accepted=0,
        ideas_generated=0,
        next_action=None,
        error=None,
        duration_ms=1,
        open_tasks=0,
        initiative=None,
    )


# ---------------------------------------------------------------------------
# Test 1: _process_boredom_delegations applies results and creates task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_delegations_applies_result_and_creates_task(tmp_path, monkeypatch):
    """When state is bored_delegating and delegation output exists:
    1. apply_boredom_delegation_result should be called
    2. materialize_boredom_initiative should be called to create a task
    3. state should advance to executing_safe
    """
    from nanobot_workspace.proactive.boredom.heartbeat_integration import (
        apply_boredom_delegation_result,
        boredom_delegation_output_path,
        materialize_boredom_initiative,
    )

    idea = _make_idea()
    initiative = _make_initiative(idea=idea, runtime_task_id="runtime-test-123")
    store = _write_state(tmp_path, BoredomMode.BORED_DELEGATING, initiative)
    _write_delegation_output(tmp_path, "boredom_test123")
    _write_delivery_log(tmp_path)

    apply_calls: list[str] = []
    materialize_calls: list[object] = []

    original_apply = apply_boredom_delegation_result
    original_materialize = materialize_boredom_initiative

    def _tracked_apply(did, payload, **kw):
        apply_calls.append(did)
        # Preserve orchestrator kwarg from caller
        kw.setdefault("workspace_root", tmp_path)
        return original_apply(did, payload, **kw)

    def _tracked_materialize(result, **kw):
        materialize_calls.append(result)
        return {"task_id": "TASK-E2E-1", "title": "Test audit"}

    hooks = (
        lambda *a, **kw: _make_spawn_delegation_result(),  # boredom_tick
        _tracked_materialize,  # materialize
        lambda cb: None,  # register_spawner
        _tracked_apply,  # apply
    )

    monkeypatch.setattr(HeartbeatService, "_load_boredom_hooks", staticmethod(lambda: hooks))
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _sync_to_thread)

    provider = DummyProvider()
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test-model")

    await service._process_boredom_delegations()

    # apply should have been called with the delegation_id
    assert "boredom_test123" in apply_calls, f"apply should have been called, got: {apply_calls}"

    # State should be executing_safe (from triage)
    state = store.load()
    assert state.mode == BoredomMode.EXECUTING_SAFE
    assert state.active_initiative is not None
    assert state.active_initiative.idea is not None

    # BUG: _process_boredom_delegations does NOT call materialize_boredom_initiative
    # After the fix, this should pass. Before the fix, materialize_calls is empty.
    assert len(materialize_calls) > 0, (
        "BUG: _process_boredom_delegations applied results but did not materialize a task. "
        "materialize_boredom_initiative must be called after apply to create a workspace task."
    )


# ---------------------------------------------------------------------------
# Test 2: _delegate_boredom_initiative processes payload and creates task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_initiative_processes_payload_and_creates_task(tmp_path, monkeypatch):
    """Full _delegate_boredom_initiative flow:
    1. on_execute runs (subagent executes)
    2. _process_boredom_delegation_payload applies + materializes
    3. _notify_boredom_completion fires after processing
    """
    _stub_git_ops(monkeypatch)

    delegation_id = "boredom_456"
    _write_delegation_output(tmp_path, delegation_id)

    order: list[str] = []

    # Monkey-patch _process_boredom_delegation_payload to track it was called
    def _tracked_process_payload(initiative):
        order.append("process_payload")
        # Simulate successful processing
        return {"task_id": "TASK-E2E-2", "title": "Payload task"}

    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _sync_to_thread)

    async def _on_execute(prompt: str) -> str:
        order.append("execute")
        return "done"

    provider = DummyProvider()
    service = HeartbeatService(
        workspace=tmp_path, provider=provider, model="test-model", on_execute=_on_execute,
    )

    original_notify = service._notify_boredom_completion

    def _tracked_notify(success, reason=None):
        order.append(f"completion_success={success}")
    monkeypatch.setattr(service, "_notify_boredom_completion", _tracked_notify)
    monkeypatch.setattr(service, "_process_boredom_delegation_payload", _tracked_process_payload)

    initiative_data = {
        "title": "Payload task",
        "description": "test",
        "category": "tests",
        "target_paths": ["tests/test_foo.py"],
        "acceptance_criteria": ["pytest"],
        "risk_class": "safe",
        "source": "boredom",
        "task_id": "TASK-E2E-2",
        "correlation_id": delegation_id,
        "initiative_id": delegation_id,
    }

    await service._delegate_boredom_initiative(initiative_data)

    # Verify order: execute -> process_payload -> completion
    assert order[0] == "execute", f"Expected execute first, got: {order}"
    assert "process_payload" in order, f"Expected process_payload in order, got: {order}"
    assert "completion_success=True" in order, f"Expected completion in order, got: {order}"

    # Verify process_payload came before completion
    payload_idx = order.index("process_payload")
    completion_idx = order.index("completion_success=True")
    assert payload_idx < completion_idx, "process_payload should happen before completion"


# ---------------------------------------------------------------------------
# Test 3: Completion callback does not prevent result processing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_callback_does_not_block_result_processing(tmp_path, monkeypatch):
    """Even if completion callback fires first (e.g., subagent completion callback),
    _process_boredom_delegations on the next tick should still work.
    """
    from nanobot_workspace.proactive.boredom.heartbeat_integration import (
        apply_boredom_delegation_result,
    )

    # State is still bored_delegating (callback hasn't fired yet)
    idea = _make_idea("Callback test")
    initiative = _make_initiative(idea=idea, task_id="boredom_789", runtime_task_id="runtime-789")
    store = _write_state(tmp_path, BoredomMode.BORED_DELEGATING, initiative)
    _write_delegation_output(tmp_path, "boredom_789")
    _write_delivery_log(tmp_path, "runtime-789")

    apply_calls: list[str] = []

    def _tracked_apply(did, payload, **kw):
        apply_calls.append(did)
        kw.setdefault("workspace_root", tmp_path)
        return apply_boredom_delegation_result(did, payload, **kw)

    hooks = (
        lambda *a, **kw: _make_skip_result(),  # boredom_tick
        lambda **kw: None,  # materialize
        lambda cb: None,  # register_spawner
        _tracked_apply,  # apply
    )

    monkeypatch.setattr(HeartbeatService, "_load_boredom_hooks", staticmethod(lambda: hooks))
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _sync_to_thread)

    provider = DummyProvider()
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test-model")

    await service._process_boredom_delegations()

    assert "boredom_789" in apply_calls
    state = store.load()
    assert state.mode == BoredomMode.EXECUTING_SAFE


# ---------------------------------------------------------------------------
# Test 4: No double processing after state reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_double_processing_after_state_reset(tmp_path, monkeypatch):
    """After callback resets state to idle_counting, delegation should not be re-processed."""
    from nanobot_workspace.proactive.boredom.heartbeat_integration import (
        apply_boredom_delegation_result,
    )

    _write_state(tmp_path, BoredomMode.IDLE_COUNTING)
    _write_delegation_output(tmp_path)
    _write_delivery_log(tmp_path)

    apply_calls: list[str] = []

    def _tracked_apply(did, payload, **kw):
        apply_calls.append(did)
        return None

    hooks = (
        lambda *a, **kw: _make_skip_result(),
        lambda **kw: None,
        lambda cb: None,
        _tracked_apply,
    )

    monkeypatch.setattr(HeartbeatService, "_load_boredom_hooks", staticmethod(lambda: hooks))
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _sync_to_thread)

    provider = DummyProvider()
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test-model")

    await service._process_boredom_delegations()

    assert apply_calls == [], f"Should not process when state is idle_counting, got: {apply_calls}"


# ---------------------------------------------------------------------------
# Test 5: Missing delivery log — graceful handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_delivery_log_no_crash(tmp_path, monkeypatch):
    """_process_boredom_delegations handles missing delivery log gracefully."""
    initiative = _make_initiative(task_id="boredom_nodel", runtime_task_id="runtime-nodel")
    store = _write_state(tmp_path, BoredomMode.BORED_DELEGATING, initiative)
    _write_delegation_output(tmp_path, "boredom_nodel")
    # No delivery log created

    hooks = (
        lambda *a, **kw: _make_skip_result(),
        lambda **kw: None,
        lambda cb: None,
        lambda *a, **kw: None,
    )

    monkeypatch.setattr(HeartbeatService, "_load_boredom_hooks", staticmethod(lambda: hooks))
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _sync_to_thread)

    provider = DummyProvider()
    service = HeartbeatService(workspace=tmp_path, provider=provider, model="test-model")

    # Should not raise
    await service._process_boredom_delegations()

    # State unchanged (no delivery record)
    state = store.load()
    assert state.mode == BoredomMode.BORED_DELEGATING


# ---------------------------------------------------------------------------
# Test 6: Timeout in _delegate_boredom_initiative fires failure callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_timeout_fires_failure_callback(tmp_path, monkeypatch):
    """When on_execute times out, completion callback fires with success=False."""
    _stub_git_ops(monkeypatch)

    completions: list[tuple[bool, str | None]] = []

    async def _slow_execute(prompt: str) -> str:
        await asyncio.sleep(10)
        return "done"

    provider = DummyProvider()
    service = HeartbeatService(
        workspace=tmp_path, provider=provider, model="test-model", on_execute=_slow_execute,
    )
    monkeypatch.setattr(service, "_notify_boredom_completion", lambda s, r=None: completions.append((s, r)))

    initiative_data = {
        **_SAFE_INITIATIVE_DICT,
    }

    # Use a very short timeout to trigger TimeoutError
    with monkeypatch.context() as m:
        import nanobot.heartbeat.service as svc_mod
        original_wait_for = asyncio.wait_for

        async def _short_timeout(coro, timeout=None):
            return await original_wait_for(coro, timeout=0.01)

        m.setattr(asyncio, "wait_for", _short_timeout)
        await service._delegate_boredom_initiative(initiative_data)

    assert len(completions) == 1
    assert completions[0] == (False, "timeout")


# ---------------------------------------------------------------------------
# Test 7: Exception in _delegate_boredom_initiative fires failure callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delegate_exception_fires_failure_callback(tmp_path, monkeypatch):
    """When on_execute raises, completion callback fires with success=False."""
    _stub_git_ops(monkeypatch)

    completions: list[tuple[bool, str | None]] = []

    async def _failing_execute(prompt: str) -> str:
        raise RuntimeError("subagent crashed")

    provider = DummyProvider()
    service = HeartbeatService(
        workspace=tmp_path, provider=provider, model="test-model", on_execute=_failing_execute,
    )
    monkeypatch.setattr(service, "_notify_boredom_completion", lambda s, r=None: completions.append((s, r)))
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _sync_to_thread)

    await service._delegate_boredom_initiative(_SAFE_INITIATIVE_DICT)

    assert len(completions) == 1
    assert completions[0][0] is False
    assert "subagent crashed" in completions[0][1]


# ---------------------------------------------------------------------------
# Test 8: Result processing failure does not prevent completion callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_processing_failure_does_not_prevent_completion(tmp_path, monkeypatch):
    """If _process_boredom_delegation_payload raises, completion still fires."""
    _stub_git_ops(monkeypatch)

    completions: list[tuple[bool, str | None]] = []

    async def _on_execute(prompt: str) -> str:
        return "done"

    provider = DummyProvider()
    service = HeartbeatService(
        workspace=tmp_path, provider=provider, model="test-model", on_execute=_on_execute,
    )
    monkeypatch.setattr(service, "_notify_boredom_completion", lambda s, r=None: completions.append((s, r)))
    monkeypatch.setattr(
        service,
        "_process_boredom_delegation_payload",
        lambda initiative: (_ for _ in ()).throw(RuntimeError("payload processing failed")),
    )
    monkeypatch.setattr("nanobot.heartbeat.service.asyncio.to_thread", _sync_to_thread)

    await service._delegate_boredom_initiative(_SAFE_INITIATIVE_DICT)

    # Completion should still fire with success=True
    assert len(completions) == 1
    assert completions[0] == (True, None)


# Shared test data
_SAFE_INITIATIVE_DICT = {
    "title": "Add doc tests",
    "description": "Document the boredom module",
    "category": "docs",
    "target_paths": ["docs/boredom.md"],
    "acceptance_criteria": ["ruff check passes"],
    "risk_class": "safe",
    "source": "boredom",
    "task_id": "TASK-9",
    "correlation_id": "boredom_123",
    "initiative_id": "boredom_123",
}
