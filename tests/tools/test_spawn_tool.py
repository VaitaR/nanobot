"""Tests for the SpawnTool post-check auto-closure hook."""

from __future__ import annotations

from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.subagent import PeakHoursSpawnBlockedError
from nanobot.agent.tools.spawn import SpawnTool


@pytest.fixture
def manager(tmp_path):
    """Return a minimal subagent manager stub for SpawnTool tests."""
    return SimpleNamespace(spawn=AsyncMock(return_value="spawned"), workspace=tmp_path)


@pytest.mark.asyncio
async def test_spawn_tool_starts_post_check_after_successful_spawn(manager) -> None:
    tool = SpawnTool(manager)
    tool._schedule_post_check = MagicMock()  # type: ignore[method-assign]

    result = await tool.execute(
        task="Implement follow-up for 20260415T120000_auto_close_task",
        label="Review task 20260415T120000_auto_close_task",
        skip_validation=True,
    )

    assert result == "spawned"
    tool._schedule_post_check.assert_called_once_with(  # type: ignore[attr-defined]
        task_id="20260415T120000_auto_close_task",
        timeout_s=1800,
    )


@pytest.mark.asyncio
async def test_spawn_tool_skips_post_check_on_failed_spawn(manager) -> None:
    tool = SpawnTool(manager)
    tool._schedule_post_check = MagicMock()  # type: ignore[method-assign]
    manager.spawn.side_effect = PeakHoursSpawnBlockedError("spawn blocked during peak hours")

    result = await tool.execute(
        task="Review task 20260415T120000_auto_close_task", skip_validation=True
    )

    assert result == "spawn blocked during peak hours"
    tool._schedule_post_check.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_post_check_loop_moves_to_review_when_validation_note_appears(
    manager, monkeypatch
) -> None:
    """When subagent sets validation_note, task moves to 'review' (not auto-closed)."""
    tool = SpawnTool(manager)
    task = SimpleNamespace(
        status="open", validation_note="verified: checked src/example.py", source="auto"
    )
    tool._load_workspace_task = AsyncMock(return_value=task)  # type: ignore[method-assign]
    mark_success_mock = AsyncMock(return_value=True)

    monkeypatch.setattr("nanobot.agent.tools.spawn.mark_task_delegation_success", mark_success_mock)

    await tool._run_post_check_loop(task_id="20260415T120000_auto_close_task", timeout_s=1)

    mark_success_mock.assert_awaited_once_with("20260415T120000_auto_close_task")


@pytest.mark.asyncio
async def test_post_check_loop_exits_immediately_when_already_in_review(
    manager, monkeypatch
) -> None:
    """When task is already in 'review' status, loop exits without calling mark_task_delegation_success."""
    tool = SpawnTool(manager)
    task = SimpleNamespace(status="review", validation_note="verified: checked src/example.py")
    tool._load_workspace_task = AsyncMock(return_value=task)  # type: ignore[method-assign]
    mark_success_mock = AsyncMock(return_value=True)

    monkeypatch.setattr("nanobot.agent.tools.spawn.mark_task_delegation_success", mark_success_mock)

    await tool._run_post_check_loop(task_id="20260415T120000_auto_close_task", timeout_s=1)

    mark_success_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_check_loop_does_not_close_without_validation_note(manager, monkeypatch) -> None:
    tool = SpawnTool(manager)
    task = SimpleNamespace(status="open", validation_note="", source="auto")
    tool._load_workspace_task = AsyncMock(side_effect=chain([task], repeat(None)))  # type: ignore[method-assign]
    mark_success_mock = AsyncMock(return_value=True)

    monkeypatch.setattr("nanobot.agent.tools.spawn.mark_task_delegation_success", mark_success_mock)
    monkeypatch.setattr("nanobot.agent.tools.spawn.asyncio.sleep", AsyncMock())

    await tool._run_post_check_loop(task_id="20260415T120000_auto_close_task", timeout_s=1)

    mark_success_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# _validate_task_packet and pre-flight validation tests
# ---------------------------------------------------------------------------

_GOOD_TASK = """\
## Scope
Implement X so that Y works correctly. What the agent must cover: write tests.
What the agent must not do: touch unrelated modules.

## Acceptance Criteria
- [ ] All tests pass
- [ ] Ruff reports no errors
"""

_THIN_TASK = "Just do the thing quickly."


def test_validate_good_packet_passes(manager) -> None:
    tool = SpawnTool(manager)
    missing = tool._validate_task_packet(_GOOD_TASK)
    assert missing == []


def test_validate_thin_packet_rejected(manager) -> None:
    tool = SpawnTool(manager)
    missing = tool._validate_task_packet(_THIN_TASK)
    assert "Scope" in missing
    assert "Acceptance Criteria" in missing


@pytest.mark.asyncio
async def test_execute_rejects_thin_packet(manager) -> None:
    tool = SpawnTool(manager)
    result = await tool.execute(task=_THIN_TASK)
    assert "rejected" in result
    assert "Scope" in result
    assert "Acceptance Criteria" in result
    manager.spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_passes_good_packet(manager) -> None:
    tool = SpawnTool(manager)
    tool._schedule_post_check = MagicMock()  # type: ignore[method-assign]
    result = await tool.execute(task=_GOOD_TASK)
    assert result == "spawned"
    manager.spawn.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_skip_validation_bypasses_check(manager) -> None:
    tool = SpawnTool(manager)
    tool._schedule_post_check = MagicMock()  # type: ignore[method-assign]
    result = await tool.execute(task=_THIN_TASK, skip_validation=True)
    assert result == "spawned"
    manager.spawn.assert_awaited_once()


def test_validate_logs_warning_for_short_packet(manager, caplog) -> None:
    import logging

    tool = SpawnTool(manager)
    with caplog.at_level(logging.WARNING, logger="nanobot.agent.tools.spawn"):
        tool._validate_task_packet("short")
    assert any("thin task packet" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _load_task_file_body and ID → file auto-loading tests
# ---------------------------------------------------------------------------

_GOOD_TASK_FILE_BODY = """\
## Scope
Refactor the spawn tool to auto-load task files by ID.
What the agent must cover: new method, tests, ruff-clean.
What the agent must not do: change heartbeat or task store.

## Acceptance Criteria
- [ ] File body is loaded when task string contains a short ID
- [ ] Validation runs on the combined content
"""

_THIN_TASK_FILE_BODY = "Just a thin placeholder with no structure."


def test_load_task_file_body_found(tmp_path, monkeypatch) -> None:
    """When the .md file exists, its content is returned."""
    tasks_dir = tmp_path / ".nanobot" / "workspace" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "20260416T214721.md").write_text(_GOOD_TASK_FILE_BODY, encoding="utf-8")

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / ".nanobot")  # noqa: PT008
    # Path.home() returns tmp_path/.nanobot, so the tasks dir is at
    # tmp_path/.nanobot/.nanobot/workspace/tasks — adjust:
    monkeypatch.setattr("nanobot.agent.tools.spawn.Path.home", lambda: tmp_path)

    result = SpawnTool._load_task_file_body("20260416T214721")
    assert result is not None
    assert "## Scope" in result
    assert "Refactor the spawn tool" in result


def test_load_task_file_body_missing(tmp_path, monkeypatch) -> None:
    """When the .md file does not exist, returns None."""
    monkeypatch.setattr("nanobot.agent.tools.spawn.Path.home", lambda: tmp_path)
    result = SpawnTool._load_task_file_body("99999999T000000")
    assert result is None


def test_load_task_file_body_missing_logs_warning(tmp_path, monkeypatch, caplog) -> None:
    """A warning is logged when the file is not found."""
    import logging

    monkeypatch.setattr("nanobot.agent.tools.spawn.Path.home", lambda: tmp_path)
    with caplog.at_level(logging.WARNING, logger="nanobot.agent.tools.spawn"):
        SpawnTool._load_task_file_body("99999999T000000")
    assert any("task file not found" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_execute_loads_task_file_and_passes_validation(tmp_path, monkeypatch) -> None:
    """A thin task string with a short ID resolves to the full file body and passes validation."""
    # Set up the fake home so tasks dir is at tmp_path/.nanobot/workspace/tasks/
    workspace_tasks = tmp_path / ".nanobot" / "workspace" / "tasks"
    workspace_tasks.mkdir(parents=True)
    (workspace_tasks / "20260416T214721.md").write_text(_GOOD_TASK_FILE_BODY, encoding="utf-8")

    monkeypatch.setattr("nanobot.agent.tools.spawn.Path.home", lambda: tmp_path)

    manager = SimpleNamespace(spawn=AsyncMock(return_value="spawned"), workspace=tmp_path)
    tool = SpawnTool(manager)
    tool._schedule_post_check = MagicMock()  # type: ignore[method-assign]

    # Thin string referencing the task ID — without file loading this would be rejected
    thin_task = "Work on 20260416T214721"
    result = await tool.execute(task=thin_task)

    assert result == "spawned"
    # The spawned task should contain the file body content
    call_args = manager.spawn.call_args
    spawned_task = call_args.kwargs.get("task") or call_args[1].get("task") or call_args[0][0]
    assert "## Scope" in spawned_task
    assert "Refactor the spawn tool" in spawned_task
    assert thin_task in spawned_task


@pytest.mark.asyncio
async def test_execute_file_not_found_proceeds_with_original(tmp_path, monkeypatch) -> None:
    """When no file matches the short ID, proceed with the original thin task string."""
    monkeypatch.setattr("nanobot.agent.tools.spawn.Path.home", lambda: tmp_path)

    manager = SimpleNamespace(spawn=AsyncMock(return_value="spawned"), workspace=tmp_path)
    tool = SpawnTool(manager)

    # This thin task contains a short ID but no file exists — should be rejected by validation
    thin_task = "Do 20260416T214721 now"
    result = await tool.execute(task=thin_task)
    assert "rejected" in result
    assert "Scope" in result


@pytest.mark.asyncio
async def test_execute_no_short_id_skips_file_loading(tmp_path, monkeypatch) -> None:
    """A task string without a short ID goes through without file loading."""
    monkeypatch.setattr("nanobot.agent.tools.spawn.Path.home", lambda: tmp_path)

    manager = SimpleNamespace(spawn=AsyncMock(return_value="spawned"), workspace=tmp_path)
    tool = SpawnTool(manager)
    tool._schedule_post_check = MagicMock()  # type: ignore[method-assign]

    result = await tool.execute(task=_GOOD_TASK)
    assert result == "spawned"
    call_args = manager.spawn.call_args
    spawned_task = call_args.kwargs.get("task") or call_args[1].get("task") or call_args[0][0]
    assert spawned_task == _GOOD_TASK


def test_load_task_file_body_io_error(tmp_path, monkeypatch) -> None:
    """An OSError (e.g. permission denied) returns None without raising."""
    monkeypatch.setattr("nanobot.agent.tools.spawn.Path.home", lambda: tmp_path)
    tasks_dir = tmp_path / ".nanobot" / "workspace" / "tasks"
    tasks_dir.mkdir(parents=True)
    bad_file = tasks_dir / "20260416T214721.md"
    bad_file.write_text("content", encoding="utf-8")

    # Make read_text raise OSError
    def _raising_read_text(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.read_text", _raising_read_text)

    result = SpawnTool._load_task_file_body("20260416T214721")
    assert result is None
