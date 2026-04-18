"""Tests for _write_acpx_telemetry and _write_subagent_telemetry."""

from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# _write_subagent_telemetry
# ---------------------------------------------------------------------------


def test_subagent_telemetry_token_counts(tmp_path: Path) -> None:
    """Token counts must use usage kwarg, not hardcoded 0."""
    from nanobot.agent.subagent import _write_subagent_telemetry

    _write_subagent_telemetry(
        tmp_path,
        model="test-model",
        duration_ms=500,
        stop_reason="completed",
        tools=["read_file"],
        error=None,
        label="test-label",
        origin={"channel": "telegram", "chat_id": "c1"},
        usage={"prompt_tokens": 200, "completion_tokens": 80},
    )

    records = _read_jsonl(tmp_path / "memory" / "improvement-log.jsonl")
    assert len(records) == 1
    r = records[0]
    assert r["prompt_tokens"] == 200
    assert r["completion_tokens"] == 80
    assert r["total_tokens"] == 280
    assert r["usage"]["prompt_tokens"] == 200
    assert r["usage"]["completion_tokens"] == 80


def test_subagent_telemetry_zero_tokens_when_no_usage(tmp_path: Path) -> None:
    """When usage is not provided, token counts should be 0."""
    from nanobot.agent.subagent import _write_subagent_telemetry

    _write_subagent_telemetry(
        tmp_path,
        model="test-model",
        duration_ms=100,
        stop_reason="completed",
        tools=[],
        error=None,
        label="test-label",
        origin={"channel": "telegram", "chat_id": "c1"},
    )

    r = _read_jsonl(tmp_path / "memory" / "improvement-log.jsonl")[0]
    assert r["total_tokens"] == 0
    assert r["prompt_tokens"] == 0
    assert r["completion_tokens"] == 0


def test_subagent_telemetry_spawn_and_task_id(tmp_path: Path) -> None:
    """spawn_id and task_id must appear in telemetry entry."""
    from nanobot.agent.subagent import _write_subagent_telemetry

    _write_subagent_telemetry(
        tmp_path,
        model="test-model",
        duration_ms=200,
        stop_reason="completed",
        tools=[],
        error=None,
        label="test-label",
        origin={"channel": "system", "chat_id": "heartbeat"},
        spawn_id="abc12345",
        task_id="20260416T210514",
    )

    r = _read_jsonl(tmp_path / "memory" / "improvement-log.jsonl")[0]
    assert r["spawn_id"] == "abc12345"
    assert r["task_id"] == "20260416T210514"


def test_subagent_telemetry_spawn_task_id_none_when_absent(tmp_path: Path) -> None:
    """spawn_id and task_id default to None when not provided."""
    from nanobot.agent.subagent import _write_subagent_telemetry

    _write_subagent_telemetry(
        tmp_path,
        model="test-model",
        duration_ms=100,
        stop_reason="completed",
        tools=[],
        error=None,
        label="test-label",
        origin={"channel": "telegram", "chat_id": "c1"},
    )

    r = _read_jsonl(tmp_path / "memory" / "improvement-log.jsonl")[0]
    assert r["spawn_id"] is None
    assert r["task_id"] is None


# ---------------------------------------------------------------------------
# _write_acpx_telemetry
# ---------------------------------------------------------------------------


def _make_delegated_result(**kwargs):
    from nanobot.agent.execution import DelegatedResult

    defaults = dict(
        success=True,
        final_message="done",
        tool_calls=(),
        total_duration=1.5,
        error=None,
        usage={},
        spawn_id=None,
        task_id=None,
    )
    defaults.update(kwargs)
    return DelegatedResult(**defaults)


def test_acpx_telemetry_token_counts(tmp_path: Path) -> None:
    """Token counts must come from DelegatedResult.usage, not hardcoded 0."""
    from nanobot.agent.execution import _write_acpx_telemetry

    result = _make_delegated_result(
        usage={"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420},
    )
    _write_acpx_telemetry(tmp_path, "codex", result)

    r = _read_jsonl(tmp_path / "memory" / "improvement-log.jsonl")[0]
    assert r["prompt_tokens"] == 300
    assert r["completion_tokens"] == 120
    assert r["total_tokens"] == 420


def test_acpx_telemetry_spawn_task_linkage(tmp_path: Path) -> None:
    """spawn_id and task_id from DelegatedResult appear in telemetry."""
    from nanobot.agent.execution import _write_acpx_telemetry

    result = _make_delegated_result(spawn_id="sp-001", task_id="20260416T210514")
    _write_acpx_telemetry(tmp_path, "codex", result)

    r = _read_jsonl(tmp_path / "memory" / "improvement-log.jsonl")[0]
    assert r["spawn_id"] == "sp-001"
    assert r["task_id"] == "20260416T210514"


def test_acpx_telemetry_tool_statuses(tmp_path: Path) -> None:
    """tool_statuses must capture completion status for each tool call."""
    from nanobot.agent.execution import ToolCall, _write_acpx_telemetry

    tc_ok = ToolCall(name="read_file", arguments={}, status="completed")
    tc_err = ToolCall(name="exec_command", arguments={}, status="failed")
    result = _make_delegated_result(tool_calls=(tc_ok, tc_err))
    _write_acpx_telemetry(tmp_path, "codex", result)

    r = _read_jsonl(tmp_path / "memory" / "improvement-log.jsonl")[0]
    assert r["tool_statuses"] == {"read_file": "completed", "exec_command": "failed"}


def test_acpx_telemetry_tool_statuses_empty_when_no_tools(tmp_path: Path) -> None:
    """tool_statuses is an empty dict when no tool calls were made."""
    from nanobot.agent.execution import _write_acpx_telemetry

    result = _make_delegated_result()
    _write_acpx_telemetry(tmp_path, "codex", result)

    r = _read_jsonl(tmp_path / "memory" / "improvement-log.jsonl")[0]
    assert r["tool_statuses"] == {}
