from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.spawn_status import SpawnStatusTool
from nanobot.bus.queue import MessageBus
from nanobot.heartbeat.service import HeartbeatService
from nanobot.providers.base import LLMProvider, LLMResponse


class DummyProvider(LLMProvider):
    async def chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


@pytest.fixture(autouse=True)
def stub_heartbeat_background(monkeypatch) -> None:
    async def _empty_tasks() -> list[dict[str, str]]:
        return []

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(HeartbeatService, "_fetch_pending_review", staticmethod(_empty_tasks))
    monkeypatch.setattr(HeartbeatService, "_fetch_tasks_in_review", staticmethod(_empty_tasks))
    monkeypatch.setattr(HeartbeatService, "_run_boredom_tick", _noop)
    monkeypatch.setattr(HeartbeatService, "_process_boredom_delegations", _noop)


def _make_service(tmp_path, *, subagent_manager=None) -> HeartbeatService:
    return HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test-model",
        subagent_manager=subagent_manager,
    )


def test_subagent_get_active_tasks_exposes_workspace_task_id(tmp_path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    manager = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())

    class PendingTask:
        def done(self) -> bool:
            return False

    manager._running_tasks["spawn-1"] = PendingTask()
    manager._task_info["spawn-1"] = {
        "label": "delegate task",
        "started_at": 0.0,
        "executor": "codex",
        "session_key": "telegram:1",
        "origin": {"channel": "telegram"},
        "workspace_task_id": "task-123",
    }

    active = manager.get_active_tasks()

    assert len(active) == 1
    assert active[0]["workspace_task_id"] == "task-123"


@pytest.mark.asyncio
async def test_spawn_status_shows_workspace_task_id() -> None:
    tool = SpawnStatusTool(
        get_active_tasks=lambda: [
            {
                "session_key": "heartbeat:boredom",
                "elapsed_seconds": 12.3,
                "workspace_task_id": "task-123",
            }
        ]
    )

    result = await tool.execute()

    assert "task=task-123" in result


def test_detect_stale_in_progress_recovers_only_tasks_without_active_spawn(tmp_path, monkeypatch) -> None:
    updated: list[tuple[str, str, str]] = []
    fake_tasks = [
        SimpleNamespace(id="task-active", status="in_progress", validation_note=""),
        SimpleNamespace(id="task-review", status="in_progress", validation_note="checked: pytest"),
        SimpleNamespace(id="task-open", status="in_progress", validation_note=""),
        SimpleNamespace(id="task-ignored", status="open", validation_note=""),
    ]

    monkeypatch.setattr(
        "nanobot_workspace.tasks.store.TaskStore.list_tasks",
        lambda self, scope: fake_tasks,
    )

    def _fake_update_status(self, task_id: str, new_status: str, reason: str = ""):
        updated.append((task_id, new_status, reason))
        return SimpleNamespace(id=task_id, status=new_status)

    monkeypatch.setattr("nanobot_workspace.tasks.store.TaskStore.update_status", _fake_update_status)

    subagent_manager = SimpleNamespace(
        get_active_tasks=lambda: [{"workspace_task_id": "task-active"}]
    )
    service = _make_service(tmp_path, subagent_manager=subagent_manager)

    fixed = service._detect_stale_in_progress()

    assert fixed == 2
    assert ("task-review", "review", "") in updated
    assert (
        "task-open",
        "open",
        "stale: no active spawn found, auto-recovered",
    ) in updated
    assert all(task_id != "task-active" for task_id, _, _ in updated)


def test_detect_stale_in_progress_returns_zero_on_store_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "nanobot_workspace.tasks.store.TaskStore.list_tasks",
        lambda self, scope: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    service = _make_service(tmp_path)

    assert service._detect_stale_in_progress() == 0


@pytest.mark.asyncio
async def test_run_tick_detects_stale_tasks_before_actionable_check(tmp_path, monkeypatch) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] check queue", encoding="utf-8")
    call_order: list[str] = []

    def _fake_detect(self) -> int:
        call_order.append("detect")
        return 1

    def _fake_check(self):
        call_order.append("check")
        return [], []

    monkeypatch.setattr(HeartbeatService, "_detect_stale_in_progress", _fake_detect)
    monkeypatch.setattr(HeartbeatService, "_check_actionable_tasks", _fake_check)
    monkeypatch.setattr("nanobot.heartbeat.service._refresh_health_state", lambda workspace: None)
    monkeypatch.setattr("nanobot.heartbeat.service._emit_heartbeat_event", lambda *args, **kwargs: None)

    service = _make_service(tmp_path)

    result = await service._run_tick(force=True)

    assert result["action"] == "skip"
    assert call_order[:2] == ["detect", "check"]
