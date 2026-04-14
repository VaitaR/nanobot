"""Adversarial test suite for ReloadChecker safe reload mechanism.

Tests cover 8 scenarios ensuring restart never breaks active work:
1. Concurrent subagent - reload waits for running subagent
2. Active message - reload defers during Telegram message processing
3. Heartbeat mid-tick - reload defers, triggers on next tick when clear
4. Multiple reload flags - exactly one reload when two flags written simultaneously
5. Broken code reload - graceful recovery via clear_markers on startup
6. Stale reload timeout - 2h+ stale marker triggers warning, no restart
7. Reload during reload - no infinite restart loop
8. Normal flow - full happy-path: marker → safe check → restart
"""
from __future__ import annotations

import json
import os
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from nanobot.agent.reload import ReloadChecker, ReloadState

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_marker(
    workspace: Path,
    *,
    reason: str = "test deploy",
    timestamp: str | None = None,
    **extra,
) -> Path:
    """Write .pending-reload marker to workspace directory."""
    ts = timestamp or datetime.now(UTC).isoformat()
    data = {"reason": reason, "timestamp": ts, **extra}
    marker = workspace / ".pending-reload"
    marker.write_text(json.dumps(data), encoding="utf-8")
    return marker


def _make_checker(
    workspace: Path,
    *,
    running: int = 0,
    active: int = 0,
) -> ReloadChecker:
    """Create ReloadChecker with mocked dependencies."""
    subagents = MagicMock()
    subagents.get_running_count.return_value = running

    agent = MagicMock()
    # _active_tasks is dict[str, list[asyncio.Task]]; sum of lengths = active count
    if active > 0:
        agent._active_tasks = {"session": [MagicMock()] * active}
    else:
        agent._active_tasks = {}

    return ReloadChecker(workspace=workspace, agent_loop=agent, subagent_manager=subagents)


# ── Scenario 1: Concurrent subagent ───────────────────────────────────────────


class TestConcurrentSubagent:
    """Spawn subagent A (long-running); reload flag appears → reload must wait for A."""

    def test_reload_deferred_while_subagent_running(self, tmp_path: Path) -> None:
        """check() returns False and does not execute reload when subagent count > 0."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=1)

        with patch.object(checker, "execute_reload") as mock_exec:
            result = checker.check()

        assert result is False
        mock_exec.assert_not_called()
        assert checker.pending_reload.exists()  # marker untouched — deferral

    def test_reload_proceeds_when_subagent_finishes(self, tmp_path: Path) -> None:
        """After subagent count drops to zero, check() triggers reload."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=0, active=0)

        with patch.object(checker, "execute_reload") as mock_exec:
            result = checker.check()

        assert result is True
        mock_exec.assert_called_once()


# ── Scenario 2: Active message processing ─────────────────────────────────────


class TestActiveMessage:
    """Reload flag appears while Telegram message is being processed → deferred."""

    def test_reload_deferred_while_active_tasks(self, tmp_path: Path) -> None:
        """check() returns False and does not execute reload when active_tasks_count > 0."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=0, active=2)

        with patch.object(checker, "execute_reload") as mock_exec:
            result = checker.check()

        assert result is False
        mock_exec.assert_not_called()
        assert checker.pending_reload.exists()  # marker untouched — deferral

    def test_reload_allowed_when_no_active_tasks(self, tmp_path: Path) -> None:
        """After message completes (tasks drain), check() proceeds with reload."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=0, active=0)

        with patch.object(checker, "execute_reload") as mock_exec:
            result = checker.check()

        assert result is True
        mock_exec.assert_called_once()


# ── Scenario 3: Heartbeat mid-tick ────────────────────────────────────────────


class TestHeartbeatMidTick:
    """Reload flag written while heartbeat is mid-tick → defers to next tick."""

    def test_reload_deferred_during_execute_phase(self, tmp_path: Path) -> None:
        """Both subagents running and tasks active (mid-tick) → deferred."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=1, active=1)

        with patch.object(checker, "execute_reload") as mock_exec:
            result = checker.check()

        assert result is False
        mock_exec.assert_not_called()

    def test_reload_triggers_on_next_tick_when_clear(self, tmp_path: Path) -> None:
        """First tick defers (subagent busy); second tick triggers reload (all idle)."""
        _make_marker(tmp_path)

        subagents = MagicMock()
        call_count = [0]

        def running_side_effect() -> int:
            call_count[0] += 1
            return 1 if call_count[0] < 2 else 0  # busy on tick 1, idle on tick 2

        subagents.get_running_count.side_effect = running_side_effect
        agent = MagicMock()
        agent._active_tasks = {}
        checker = ReloadChecker(workspace=tmp_path, agent_loop=agent, subagent_manager=subagents)

        with patch.object(checker, "execute_reload") as mock_exec:
            result1 = checker.check()  # tick 1: subagent running → deferred
            # .pending-reload still present (check returned False, no rename)
            result2 = checker.check()  # tick 2: subagent done → reload

        assert result1 is False
        assert result2 is True
        mock_exec.assert_called_once()


# ── Scenario 4: Multiple reload flags ─────────────────────────────────────────


class TestMultipleReloadFlags:
    """Two subagents finish simultaneously, both write .pending-reload → one reload."""

    def test_only_one_reload_when_marker_already_consumed(self, tmp_path: Path) -> None:
        """First check consumes marker; second check finds no .pending-reload."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=0, active=0)

        reload_count = [0]

        def count_exec(state: ReloadState) -> None:
            reload_count[0] += 1

        with patch.object(checker, "execute_reload", side_effect=count_exec):
            result1 = checker.check()
            result2 = checker.check()  # no new marker written

        assert result1 is True
        assert result2 is False
        assert reload_count[0] == 1

    def test_rename_atomic_prevents_double_restart(self, tmp_path: Path) -> None:
        """After reload: .pending-reload gone, .reloading exists → prevents second exec."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=0, active=0)

        with patch.object(checker, "execute_reload") as mock_exec:
            checker.check()

        assert not checker.pending_reload.exists()
        assert checker.reloading.exists()
        mock_exec.assert_called_once()


# ── Scenario 5: Broken code reload ────────────────────────────────────────────


class TestBrokenCodeReload:
    """Subagent writes bad code → gateway respawns → stale markers cleared on startup."""

    def test_clear_markers_removes_reloading_on_startup(self, tmp_path: Path) -> None:
        """Leftover .reloading (from crashed previous process) is cleaned up."""
        (tmp_path / ".reloading").write_text('{"reason": "broken code"}', encoding="utf-8")
        checker = _make_checker(tmp_path)

        checker.clear_markers()

        assert not checker.reloading.exists()
        assert not checker.pending_reload.exists()

    def test_clear_markers_removes_both_files(self, tmp_path: Path) -> None:
        """clear_markers removes both .pending-reload and .reloading."""
        _make_marker(tmp_path)
        (tmp_path / ".reloading").write_text('{"reason": "old"}', encoding="utf-8")
        checker = _make_checker(tmp_path)

        checker.clear_markers()

        assert not checker.pending_reload.exists()
        assert not checker.reloading.exists()

    def test_new_reload_proceeds_despite_leftover_reloading(self, tmp_path: Path) -> None:
        """If .reloading exists from prior crash, prepare_reload overwrites it atomically."""
        _make_marker(tmp_path, reason="post-crash deploy")
        (tmp_path / ".reloading").write_text('{"reason": "leftover"}', encoding="utf-8")
        checker = _make_checker(tmp_path, running=0, active=0)

        with patch.object(checker, "execute_reload") as mock_exec:
            result = checker.check()

        assert result is True
        mock_exec.assert_called_once()
        assert checker.reloading.exists()
        assert not checker.pending_reload.exists()


# ── Scenario 6: Stale reload timeout ──────────────────────────────────────────


class TestStaleReloadTimeout:
    """.pending-reload sits for 2h+ with no safe window → warning logged, no restart."""

    def test_stale_marker_logs_warning_and_blocks_reload(self, tmp_path: Path) -> None:
        """Marker >2h old triggers check_age warning and check() returns False without reloading."""
        stale_ts = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
        _make_marker(tmp_path, timestamp=stale_ts)
        checker = _make_checker(tmp_path, running=0, active=0)

        # Verify check_age correctly identifies the stale marker
        state = checker.read_pending()
        assert state is not None
        is_stale, warning = checker.check_age(state)
        assert is_stale is True
        assert warning is not None
        assert "reload pending for" in warning

        # Verify check() refuses to reload and preserves the marker
        with patch.object(checker, "execute_reload") as mock_exec:
            result = checker.check()

        assert result is False
        mock_exec.assert_not_called()
        assert checker.pending_reload.exists()  # marker preserved for manual inspection

    def test_check_age_detects_stale_after_two_hours(self, tmp_path: Path) -> None:
        """check_age returns (True, warning_message) for markers ≥2h old."""
        stale_ts = (datetime.now(UTC) - timedelta(hours=2, minutes=1)).isoformat()
        state = ReloadState(reason="test", timestamp=stale_ts)
        checker = _make_checker(tmp_path)

        is_stale, warning = checker.check_age(state)

        assert is_stale is True
        assert warning is not None
        assert "reload pending for" in warning

    def test_check_age_not_stale_under_two_hours(self, tmp_path: Path) -> None:
        """Marker under 2h old is not considered stale."""
        fresh_ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        state = ReloadState(reason="test", timestamp=fresh_ts)
        checker = _make_checker(tmp_path)

        is_stale, _ = checker.check_age(state)

        assert is_stale is False

    def test_check_age_no_timestamp_not_stale(self, tmp_path: Path) -> None:
        """Marker with no timestamp is treated as non-stale (unknown age)."""
        state = ReloadState(reason="test", timestamp="")
        checker = _make_checker(tmp_path)

        is_stale, warning = checker.check_age(state)

        assert is_stale is False
        assert warning is None


# ── Scenario 7: Reload during reload ──────────────────────────────────────────


class TestReloadDuringReload:
    """Restart in progress, another .pending-reload written → no infinite loop."""

    def test_second_check_finds_no_marker_after_first_reload(self, tmp_path: Path) -> None:
        """First reload renames marker → second check returns False (no infinite loop)."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=0, active=0)

        with patch.object(checker, "execute_reload"):
            first = checker.check()
            second = checker.check()  # .pending-reload is gone (renamed to .reloading)

        assert first is True
        assert second is False

    def test_new_marker_after_restart_is_independent_request(self, tmp_path: Path) -> None:
        """A new .pending-reload written after first restart is a separate deploy, not a loop."""
        _make_marker(tmp_path)
        checker = _make_checker(tmp_path, running=0, active=0)
        exec_count = [0]

        def count_exec(state: ReloadState) -> None:
            exec_count[0] += 1

        with patch.object(checker, "execute_reload", side_effect=count_exec):
            checker.check()  # first deploy
            # Simulate: systemd respawns, new subagent writes another marker
            _make_marker(tmp_path, reason="second deploy")
            checker.check()  # second independent deploy

        assert exec_count[0] == 2  # two distinct restarts, not an infinite loop

    def test_pending_reload_not_present_means_no_action(self, tmp_path: Path) -> None:
        """Without .pending-reload, check() is always a no-op."""
        checker = _make_checker(tmp_path, running=0, active=0)

        with patch.object(checker, "execute_reload") as mock_exec:
            for _ in range(5):
                result = checker.check()
                assert result is False

        mock_exec.assert_not_called()


# ── Scenario 8: Normal flow ────────────────────────────────────────────────────


class TestNormalFlow:
    """Subagent writes code → lint/test pass → reload flag → heartbeat → safe restart."""

    def test_full_reload_flow_when_safe(self, tmp_path: Path) -> None:
        """Happy path: marker present, no subagents, no tasks → reload triggered once."""
        _make_marker(
            tmp_path,
            reason="new feature deployed",
            subagent_id="sa-001",
            task_id="task-42",
            files_modified=["nanobot/agent/foo.py"],
        )
        checker = _make_checker(tmp_path, running=0, active=0)

        with patch.object(checker, "execute_reload") as mock_exec:
            result = checker.check()

        assert result is True
        mock_exec.assert_called_once()
        called_state: ReloadState = mock_exec.call_args[0][0]
        assert called_state.reason == "new feature deployed"
        assert called_state.subagent_id == "sa-001"
        assert "nanobot/agent/foo.py" in called_state.files_modified

    def test_execute_reload_sends_sigterm(self, tmp_path: Path) -> None:
        """execute_reload sends SIGTERM to the current process."""
        state = ReloadState(
            reason="update widget",
            files_modified=["nanobot/widget.py"],
            timestamp=datetime.now(UTC).isoformat(),
        )
        checker = _make_checker(tmp_path)

        with patch("os.kill") as mock_kill:
            checker.execute_reload(state)

        mock_kill.assert_called_once()
        args = mock_kill.call_args[0]
        assert args[0] == os.getpid()
        assert args[1] == signal.SIGTERM

    def test_state_transitions_pending_to_reloading(self, tmp_path: Path) -> None:
        """After reload: .pending-reload gone and .reloading holds original state."""
        _make_marker(tmp_path, reason="deploy v2")
        checker = _make_checker(tmp_path, running=0, active=0)

        with patch.object(checker, "execute_reload"):
            checker.check()

        assert not checker.pending_reload.exists()
        assert checker.reloading.exists()
        data = json.loads(checker.reloading.read_text(encoding="utf-8"))
        assert data["reason"] == "deploy v2"

    def test_read_pending_returns_none_when_no_marker(self, tmp_path: Path) -> None:
        """read_pending() returns None when .pending-reload does not exist."""
        checker = _make_checker(tmp_path)
        assert checker.read_pending() is None

    def test_read_pending_returns_state_with_all_fields(self, tmp_path: Path) -> None:
        """read_pending() correctly deserialises all ReloadState fields."""
        ts = datetime.now(UTC).isoformat()
        _make_marker(
            tmp_path,
            reason="full state test",
            timestamp=ts,
            subagent_id="sa-99",
            task_id="task-7",
            files_modified=["a.py", "b.py"],
            verification={"linted": True},
        )
        checker = _make_checker(tmp_path)

        state = checker.read_pending()

        assert state is not None
        assert state.reason == "full state test"
        assert state.subagent_id == "sa-99"
        assert state.task_id == "task-7"
        assert state.files_modified == ["a.py", "b.py"]
        assert state.verification == {"linted": True}
        assert state.timestamp == ts
