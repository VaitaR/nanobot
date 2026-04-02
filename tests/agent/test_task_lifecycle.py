"""Tests for nanobot.agent.task_lifecycle — task ID extraction and CLI bridge."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.agent.task_lifecycle import (
    _build_cmd,
    extract_task_id,
    mark_task_delegated,
    mark_task_delegation_failure,
    mark_task_delegation_success,
    query_pending_review,
)

# ── extract_task_id ──────────────────────────────────────────────────────────


class TestExtractTaskId:
    """Task ID extraction from various label formats."""

    def test_bare_task_id_in_label(self):
        label = "implement caching (20260329T140912_phase_2_add_fts5_memory_search)"
        assert extract_task_id(label) == "20260329T140912_phase_2_add_fts5_memory_search"

    def test_bare_task_id_at_end(self):
        label = "fix the bug 20260329T140912_phase_2_implement_evolution_store_and_so"
        assert extract_task_id(label) == "20260329T140912_phase_2_implement_evolution_store_and_so"

    def test_parenthetical_hint_task_full_id(self):
        label = "implement thing (task: 20260329T140912_phase_2_add_fts5_memory_search)"
        assert extract_task_id(label) == "20260329T140912_phase_2_add_fts5_memory_search"

    def test_parenthetical_hint_hb_full_id(self):
        label = "do stuff (hb: 20260329T140912_phase_2_add_fts5_memory_search)"
        assert extract_task_id(label) == "20260329T140912_phase_2_add_fts5_memory_search"

    def test_parenthetical_hint_short_id_skipped(self):
        """Short numeric refs like (hb: 152315) can't be resolved — skip them."""
        label = "do stuff (hb: 152315)"
        assert extract_task_id(label) is None

    def test_parenthetical_hint_priority_over_bare(self):
        """If label has both a hint and a bare ID, the hint wins."""
        label = "(task: 20260329T111111_alpha) also has 20260329T222222_beta"
        assert extract_task_id(label) == "20260329T111111_alpha"

    def test_no_task_id(self):
        label = "just a normal label with no task id"
        assert extract_task_id(label) is None

    def test_empty_label(self):
        assert extract_task_id("") is None
        assert extract_task_id(None) is None  # type: ignore[arg-type]

    def test_task_id_with_underscores(self):
        label = "20260329T140912_implement_evolution_store_and_solidification"
        assert extract_task_id(label) == "20260329T140912_implement_evolution_store_and_solidification"

    def test_partial_timestamp_rejected(self):
        """Must be exactly 8 digits + T + 6 digits + underscore."""
        label = "2026032T14091_short_slug"
        assert extract_task_id(label) is None

    def test_whitespace_in_parenthetical(self):
        label = "do thing ( task: 20260329T140912_add_fts5 )"
        assert extract_task_id(label) == "20260329T140912_add_fts5"

    def test_multiple_bare_ids_returns_first(self):
        label = "20260329T111111_first and 20260329T222222_second"
        assert extract_task_id(label) == "20260329T111111_first"


# ── _build_cmd ───────────────────────────────────────────────────────────────


class TestBuildCmd:
    """Verify CLI command generation."""

    def test_update_command(self):
        args = "update '20260329T140912_add_fts5' --status in_progress --reason 'delegated'"
        cmd = _build_cmd(args)
        assert cmd.startswith("cd ")
        assert "uv run nanobot-tasks" in cmd
        assert args in cmd

    def test_event_command(self):
        args = "event '20260329T140912_add_fts5' 'delegation completed successfully'"
        cmd = _build_cmd(args)
        assert "uv run nanobot-tasks event" in cmd


# ── CLI bridge (mocked subprocess) ──────────────────────────────────────────


@pytest.fixture()
def mock_create_subprocess():
    """Patch asyncio.create_subprocess_shell for CLI tests."""
    async def fake_shell(cmd, *, stdout, stderr):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
        proc.returncode = 0
        return proc

    with patch("nanobot.agent.task_lifecycle.asyncio.create_subprocess_shell", side_effect=fake_shell):
        yield


class TestMarkTaskDelegated:
    @pytest.mark.asyncio
    async def test_calls_update_and_event(self, mock_create_subprocess):
        result = await mark_task_delegated("20260329T140912_add_fts5")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_if_partial_success(self):
        """If update fails but event succeeds, still returns True."""

        call_count = 0

        async def fake_shell(cmd, *, stdout, stderr):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            # First call (update) fails, second call (event) succeeds
            if call_count == 1:
                proc.communicate = AsyncMock(return_value=(b"", b"not found"))
                proc.returncode = 1
            else:
                proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
                proc.returncode = 0
            return proc

        with patch("nanobot.agent.task_lifecycle.asyncio.create_subprocess_shell", side_effect=fake_shell):
            result = await mark_task_delegated("20260329T140912_add_fts5")
            assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_if_all_fail(self):
        async def fake_shell(cmd, *, stdout, stderr):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b"error"))
            proc.returncode = 1
            return proc

        with patch("nanobot.agent.task_lifecycle.asyncio.create_subprocess_shell", side_effect=fake_shell):
            result = await mark_task_delegated("20260329T140912_nonexistent")
            assert result is False


class TestMarkTaskDelegationSuccess:
    @pytest.mark.asyncio
    async def test_calls_event_with_progress_kind(self, mock_create_subprocess):
        result = await mark_task_delegation_success("20260329T140912_add_fts5")
        assert result is True


class TestMarkTaskDelegationFailure:
    @pytest.mark.asyncio
    async def test_includes_reason(self, mock_create_subprocess):
        result = await mark_task_delegation_failure(
            "20260329T140912_add_fts5", "tool_error during execution"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_truncates_long_reason(self, mock_create_subprocess):
        long_reason = "x" * 300
        result = await mark_task_delegation_failure("20260329T140912_add_fts5", long_reason)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_cli_error(self):
        async def fake_shell(cmd, *, stdout, stderr):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b"error"))
            proc.returncode = 1
            return proc

        with patch("nanobot.agent.task_lifecycle.asyncio.create_subprocess_shell", side_effect=fake_shell):
            result = await mark_task_delegation_failure("20260329T140912_add_fts5", "something broke")
            assert result is False


class TestCLITimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        """CLI calls that time out return False gracefully."""

        async def fake_shell(cmd, *, stdout, stderr):
            proc = AsyncMock()
            proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            proc.returncode = -1
            return proc

        with patch("nanobot.agent.task_lifecycle.asyncio.create_subprocess_shell", side_effect=fake_shell):
            assert await mark_task_delegated("20260329T140912_add_fts5") is False
            assert await mark_task_delegation_success("20260329T140912_add_fts5") is False
            assert await mark_task_delegation_failure("20260329T140912_add_fts5", "err") is False


class TestQueryPendingReview:
    @pytest.mark.asyncio
    async def test_scans_open_tasks_when_in_progress_empty(self):
        task_id = "20260329T140912_open_delegated"

        async def fake_shell(cmd, *, stdout, stderr):
            proc = AsyncMock()
            proc.returncode = 0
            if "list --status in_progress --json" in cmd:
                proc.communicate = AsyncMock(return_value=(b"[]", b""))
            elif "list --status open --json" in cmd:
                payload = [{"id": task_id, "title": "Recover delegated task", "status": "open"}]
                proc.communicate = AsyncMock(return_value=(json.dumps(payload).encode("utf-8"), b""))
            elif f"show {task_id}" in cmd:
                proc.communicate = AsyncMock(
                    return_value=(
                        b"[delegated] delegated to subagent\n[progress] delegation completed successfully\n",
                        b"",
                    )
                )
            else:
                proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("nanobot.agent.task_lifecycle.asyncio.create_subprocess_shell", side_effect=fake_shell):
            pending = await query_pending_review()

        assert pending == [{"id": task_id, "title": "Recover delegated task", "status": "open"}]
