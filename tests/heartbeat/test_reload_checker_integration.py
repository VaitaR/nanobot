"""Tests for ReloadChecker integration into HeartbeatService._run_tick()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.heartbeat.service import HeartbeatService
from nanobot.providers.base import LLMProvider, LLMResponse


class DummyProvider(LLMProvider):
    async def chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


@pytest.fixture(autouse=True)
def stub_heartbeat_background(monkeypatch) -> None:
    """Suppress boredom, review, and pending-review logic for focused tests."""

    async def _empty_tasks() -> list[dict[str, str]]:
        return []

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(HeartbeatService, "_fetch_pending_review", staticmethod(_empty_tasks))
    monkeypatch.setattr(HeartbeatService, "_fetch_tasks_in_review", staticmethod(_empty_tasks))
    monkeypatch.setattr(HeartbeatService, "_run_boredom_tick", _noop)
    monkeypatch.setattr(HeartbeatService, "_process_boredom_delegations", _noop)


def _make_service(tmp_path: Path) -> HeartbeatService:
    return HeartbeatService(
        workspace=tmp_path,
        provider=DummyProvider(),
        model="test-model",
    )


@pytest.mark.asyncio
async def test_reload_checker_is_optional_and_tick_proceeds(tmp_path, monkeypatch) -> None:
    """When no reload checker is configured, _run_tick proceeds as usual."""
    (tmp_path / "HEARTBEAT.md").write_text("# Heartbeat\n- [ ] something", encoding="utf-8")
    monkeypatch.setattr("nanobot.heartbeat.service._refresh_health_state", lambda workspace: None)
    monkeypatch.setattr("nanobot.heartbeat.service._emit_heartbeat_event", lambda *a, **kw: None)

    service = _make_service(tmp_path)
    assert service._reload_checker is None

    result = await service._run_tick(force=True)

    assert result["action"] == "skip"
    assert result["result"] != "reload triggered"


@pytest.mark.asyncio
async def test_reload_checker_false_allows_tick_work(tmp_path, monkeypatch) -> None:
    """When reload_checker.check() returns False, the rest of the tick still runs."""
    (tmp_path / "HEARTBEAT.md").write_text("# Heartbeat\n- [ ] something", encoding="utf-8")
    monkeypatch.setattr("nanobot.heartbeat.service._refresh_health_state", lambda workspace: None)
    monkeypatch.setattr("nanobot.heartbeat.service._emit_heartbeat_event", lambda *a, **kw: None)

    service = _make_service(tmp_path)
    mock_checker = MagicMock()
    mock_checker.check.return_value = False
    service.set_reload_checker(mock_checker)

    initial_count = service._tick_count
    result = await service._run_tick(force=True)

    mock_checker.check.assert_called_once_with()
    assert service._tick_count == initial_count + 1
    assert result["action"] == "skip"
    assert result["result"] != "reload triggered"


@pytest.mark.asyncio
async def test_reload_checker_true_short_circuits_tick(tmp_path, monkeypatch) -> None:
    """When reload_checker.check() returns True, tick exits before other work."""
    monkeypatch.setattr("nanobot.heartbeat.service._emit_heartbeat_event", lambda *a, **kw: None)

    service = _make_service(tmp_path)
    mock_checker = MagicMock()
    mock_checker.check.return_value = True
    service.set_reload_checker(mock_checker)

    initial_count = service._tick_count
    result = await service._run_tick(force=True)

    mock_checker.check.assert_called_once_with()
    assert service._tick_count == initial_count
    assert result == {
        "action": "skip",
        "tasks": "",
        "result": "reload triggered",
        "review_decisions": [],
    }


@pytest.mark.asyncio
async def test_reload_checker_true_skips_rest_of_tick(tmp_path, monkeypatch) -> None:
    """Reload-triggered ticks must not perform stale detection or file reads."""
    monkeypatch.setattr("nanobot.heartbeat.service._emit_heartbeat_event", lambda *a, **kw: None)

    stale_called = False
    heartbeat_read = False

    def _fake_detect(self) -> int:
        nonlocal stale_called
        stale_called = True
        return 0

    def _fake_read(self) -> str | None:
        nonlocal heartbeat_read
        heartbeat_read = True
        return "# Heartbeat"

    monkeypatch.setattr(HeartbeatService, "_detect_stale_in_progress", _fake_detect)
    monkeypatch.setattr(HeartbeatService, "_read_heartbeat_file", _fake_read)

    service = _make_service(tmp_path)
    mock_checker = MagicMock()
    mock_checker.check.return_value = True
    service.set_reload_checker(mock_checker)

    await service._run_tick(force=True)

    assert stale_called is False
    assert heartbeat_read is False


@pytest.mark.asyncio
async def test_set_reload_checker_stores_checker(tmp_path) -> None:
    """set_reload_checker stores and clears the checker instance."""
    service = _make_service(tmp_path)
    assert service._reload_checker is None

    checker = MagicMock()
    service.set_reload_checker(checker)
    assert service._reload_checker is checker

    service.set_reload_checker(None)
    assert service._reload_checker is None
