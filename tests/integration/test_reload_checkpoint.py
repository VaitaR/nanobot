"""Tests for ReloadChecker checkpoint/resume mechanism."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from nanobot.agent.reload import ReloadChecker, ReloadState

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_marker(workspace: Path, *, reason: str = "deploy", task_id: str | None = None) -> None:
    from datetime import UTC, datetime

    data: dict = {"reason": reason, "timestamp": datetime.now(UTC).isoformat()}
    if task_id:
        data["task_id"] = task_id
    (workspace / ".pending-reload").write_text(json.dumps(data), encoding="utf-8")


def _make_checker(workspace: Path) -> ReloadChecker:
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent._active_tasks = {}
    subagents = MagicMock()
    subagents.get_running_count.return_value = 0
    return ReloadChecker(workspace=workspace, agent_loop=agent, subagent_manager=subagents)


# ── write_checkpoint ──────────────────────────────────────────────────────────


class TestWriteCheckpoint:
    def test_creates_checkpoint_file(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.write_checkpoint(["task-1", "task-2"])
        assert checker.checkpoint.exists()

    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.write_checkpoint(["task-abc", "task-xyz"])
        data = json.loads(checker.checkpoint.read_text(encoding="utf-8"))
        assert data["task_ids"] == ["task-abc", "task-xyz"]

    def test_empty_list_round_trip(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.write_checkpoint([])
        data = json.loads(checker.checkpoint.read_text(encoding="utf-8"))
        assert data["task_ids"] == []

    def test_checkpoint_has_timestamp(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.write_checkpoint(["t1"])
        data = json.loads(checker.checkpoint.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert data["timestamp"]

    def test_overwrite_replaces_previous(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.write_checkpoint(["old-task"])
        checker.write_checkpoint(["new-task"])
        result = checker.resume_from_checkpoint()
        assert result == ["new-task"]

    def test_resume_consumes_checkpoint_file(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.write_checkpoint(["task-1"])

        assert checker.resume_from_checkpoint() == ["task-1"]
        assert not checker.checkpoint.exists()

    def test_atomic_no_tmp_file_left_on_success(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.write_checkpoint(["t1"])
        tmp_files = list(tmp_path.glob(".reload-checkpoint-*"))
        assert tmp_files == []  # temp file renamed away


# ── resume_from_checkpoint ────────────────────────────────────────────────────


class TestResumeFromCheckpoint:
    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        assert checker.resume_from_checkpoint() == []

    def test_returns_task_ids_after_write(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.write_checkpoint(["task-1", "task-2"])
        assert checker.resume_from_checkpoint() == ["task-1", "task-2"]

    def test_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.checkpoint.write_text("not valid json{{", encoding="utf-8")
        assert checker.resume_from_checkpoint() == []

    def test_missing_task_ids_key_returns_empty(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.checkpoint.write_text('{"other": "data"}', encoding="utf-8")
        assert checker.resume_from_checkpoint() == []

    def test_wrong_type_for_task_ids_returns_empty(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.checkpoint.write_text('{"task_ids": "not-a-list"}', encoding="utf-8")
        assert checker.resume_from_checkpoint() == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.checkpoint.write_text("", encoding="utf-8")
        assert checker.resume_from_checkpoint() == []

    def test_null_task_ids_returns_empty(self, tmp_path: Path) -> None:
        checker = _make_checker(tmp_path)
        checker.checkpoint.write_text('{"task_ids": null}', encoding="utf-8")
        assert checker.resume_from_checkpoint() == []


# ── check() sequencing ────────────────────────────────────────────────────────


class TestCheckSequencing:
    def test_checkpoint_written_before_execute_reload(self, tmp_path: Path) -> None:
        """write_checkpoint must be called before execute_reload."""
        _make_marker(tmp_path, task_id="task-99")
        checker = _make_checker(tmp_path)

        call_order: list[str] = []

        def fake_write(task_ids: list[str]) -> None:
            call_order.append("checkpoint")

        def fake_exec(state: ReloadState) -> None:
            call_order.append("execute")

        with (
            patch.object(checker, "write_checkpoint", side_effect=fake_write),
            patch.object(checker, "execute_reload", side_effect=fake_exec),
        ):
            checker.check()

        assert call_order == ["checkpoint", "execute"]

    def test_checkpoint_receives_task_id_from_state(self, tmp_path: Path) -> None:
        """write_checkpoint is called with the task_id from the reload marker."""
        _make_marker(tmp_path, task_id="task-42")
        checker = _make_checker(tmp_path)

        captured: list[list[str]] = []

        def fake_write(task_ids: list[str]) -> None:
            captured.append(task_ids)

        with (
            patch.object(checker, "write_checkpoint", side_effect=fake_write),
            patch.object(checker, "execute_reload"),
        ):
            checker.check()

        assert captured == [["task-42"]]

    def test_checkpoint_receives_empty_list_when_no_task_id(self, tmp_path: Path) -> None:
        """write_checkpoint called with [] when marker has no task_id."""
        _make_marker(tmp_path)  # no task_id
        checker = _make_checker(tmp_path)

        captured: list[list[str]] = []

        def fake_write(task_ids: list[str]) -> None:
            captured.append(task_ids)

        with (
            patch.object(checker, "write_checkpoint", side_effect=fake_write),
            patch.object(checker, "execute_reload"),
        ):
            checker.check()

        assert captured == [[]]

    def test_checkpoint_not_written_when_reload_deferred(self, tmp_path: Path) -> None:
        """If safety checks fail, checkpoint is not written."""
        from unittest.mock import MagicMock

        _make_marker(tmp_path, task_id="task-7")
        agent = MagicMock()
        agent._active_tasks = {"s": [MagicMock()]}  # active task → deferred
        subagents = MagicMock()
        subagents.get_running_count.return_value = 0
        checker = ReloadChecker(workspace=tmp_path, agent_loop=agent, subagent_manager=subagents)

        with patch.object(checker, "write_checkpoint") as mock_cp:
            result = checker.check()

        assert result is False
        mock_cp.assert_not_called()

    def test_full_check_writes_checkpoint_file(self, tmp_path: Path) -> None:
        """Integration: check() creates a real checkpoint file with the task_id."""
        _make_marker(tmp_path, task_id="task-full")
        checker = _make_checker(tmp_path)

        with patch.object(checker, "execute_reload"):
            checker.check()

        assert checker.checkpoint.exists()
        assert checker.resume_from_checkpoint() == ["task-full"]
