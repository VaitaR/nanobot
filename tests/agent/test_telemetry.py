"""Tests for nanobot.agent.telemetry — OBS-001/002/003."""

from __future__ import annotations

import json
from pathlib import Path

from nanobot.agent.telemetry import (
    ErrorCategory,
    TelemetryCollector,
    classify_error,
)

# ---------------------------------------------------------------------------
# classify_error — OBS-002
# ---------------------------------------------------------------------------


def test_classify_error_llm_api() -> None:
    assert classify_error("rate limit exceeded") == ErrorCategory.LLM_API
    assert classify_error("429 too many requests") == ErrorCategory.LLM_API
    assert classify_error("Connection timeout") == ErrorCategory.LLM_API
    assert classify_error("api_error: invalid api key") == ErrorCategory.LLM_API


def test_classify_error_tool_execution() -> None:
    assert classify_error("command failed with exit code 1") == ErrorCategory.TOOL_EXECUTION
    assert classify_error("Permission denied") == ErrorCategory.TOOL_EXECUTION
    assert classify_error("file not found: /tmp/foo") == ErrorCategory.TOOL_EXECUTION


def test_classify_error_validation() -> None:
    assert classify_error("validation error: malformed JSON") == ErrorCategory.VALIDATION
    assert classify_error("Invalid schema") == ErrorCategory.VALIDATION


def test_classify_error_session() -> None:
    assert classify_error("session lock timeout") == ErrorCategory.SESSION
    assert classify_error("Failed to save session") == ErrorCategory.SESSION


def test_classify_error_configuration() -> None:
    assert classify_error("Missing config key") == ErrorCategory.CONFIGURATION
    assert classify_error("invalid env variable") == ErrorCategory.CONFIGURATION


def test_classify_error_unknown() -> None:
    assert classify_error("something completely unexpected") == ErrorCategory.UNKNOWN


def test_classify_error_none() -> None:
    assert classify_error(None) == ErrorCategory.UNKNOWN
    assert classify_error("") == ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# TelemetryCollector.record_turn — OBS-001/003
# ---------------------------------------------------------------------------


def _collector(tmp_path: Path) -> TelemetryCollector:
    return TelemetryCollector(tmp_path)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().strip().splitlines():
        records.append(json.loads(line))
    return records


def test_record_turn_basic(tmp_path: Path) -> None:
    """Minimal invocation writes JSONL with all expected fields."""
    tc = _collector(tmp_path)
    tc.record_turn(
        ts="2026-01-01T00:00:00Z",
        session="s1",
        channel="telegram",
        chat_id="c1",
        model="claude-3",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        duration_ms=1200,
        stop_reason="end_turn",
        error=None,
        tools_used=["read_file"],
        skills=[],
        files_touched=["/tmp/a.py"],
    )
    records = _read_jsonl(tc._path)
    assert len(records) == 1
    r = records[0]
    assert r["session"] == "s1"
    assert r["error"] is None
    assert r["tools"] == ["read_file"]
    assert r["files_touched"] == ["/tmp/a.py"]


def test_record_turn_with_request_id(tmp_path: Path) -> None:
    """OBS-001: request_id appears in output."""
    tc = _collector(tmp_path)
    tc.record_turn(
        ts="2026-01-01T00:00:00Z",
        session="s1",
        channel="telegram",
        chat_id="c1",
        model="claude-3",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        duration_ms=1200,
        stop_reason="end_turn",
        error=None,
        tools_used=[],
        skills=[],
        files_touched=[],
        request_id="abc123def456",
    )
    records = _read_jsonl(tc._path)
    assert records[0]["request_id"] == "abc123def456"


def test_record_turn_without_request_id(tmp_path: Path) -> None:
    """Backward compat: request_id defaults to None."""
    tc = _collector(tmp_path)
    tc.record_turn(
        ts="2026-01-01T00:00:00Z",
        session="s1",
        channel="telegram",
        chat_id="c1",
        model="claude-3",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        duration_ms=1200,
        stop_reason="end_turn",
        error=None,
        tools_used=[],
        skills=[],
        files_touched=[],
    )
    records = _read_jsonl(tc._path)
    assert records[0]["request_id"] is None


def test_record_turn_error_category(tmp_path: Path) -> None:
    """OBS-002: error string gets classified into error_category."""
    tc = _collector(tmp_path)
    tc.record_turn(
        ts="2026-01-01T00:00:00Z",
        session="s1",
        channel="telegram",
        chat_id="c1",
        model="claude-3",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        duration_ms=1200,
        stop_reason="error",
        error="rate limit exceeded",
        tools_used=[],
        skills=[],
        files_touched=[],
    )
    records = _read_jsonl(tc._path)
    assert records[0]["error_category"] == "llm_api"


def test_backward_compat_no_error(tmp_path: Path) -> None:
    """Record without error → error_category is None."""
    tc = _collector(tmp_path)
    tc.record_turn(
        ts="2026-01-01T00:00:00Z",
        session="s1",
        channel="telegram",
        chat_id="c1",
        model="claude-3",
        usage={"prompt_tokens": 100, "completion_tokens": 50},
        duration_ms=1200,
        stop_reason="end_turn",
        error=None,
        tools_used=[],
        skills=[],
        files_touched=[],
    )
    records = _read_jsonl(tc._path)
    assert records[0]["error_category"] is None


def test_cumulative_tokens(tmp_path: Path) -> None:
    """OBS-003: total_tokens = prompt + completion as top-level fields."""
    tc = _collector(tmp_path)
    tc.record_turn(
        ts="2026-01-01T00:00:00Z",
        session="s1",
        channel="telegram",
        chat_id="c1",
        model="claude-3",
        usage={"prompt_tokens": 300, "completion_tokens": 150},
        duration_ms=1200,
        stop_reason="end_turn",
        error=None,
        tools_used=[],
        skills=[],
        files_touched=[],
    )
    records = _read_jsonl(tc._path)
    r = records[0]
    assert r["prompt_tokens"] == 300
    assert r["completion_tokens"] == 150
    assert r["total_tokens"] == 450


def test_cumulative_tokens_missing_fields(tmp_path: Path) -> None:
    """OBS-003: graceful when usage dict lacks token fields."""
    tc = _collector(tmp_path)
    tc.record_turn(
        ts="2026-01-01T00:00:00Z",
        session="s1",
        channel="telegram",
        chat_id="c1",
        model="claude-3",
        usage={},
        duration_ms=1200,
        stop_reason="end_turn",
        error=None,
        tools_used=[],
        skills=[],
        files_touched=[],
    )
    records = _read_jsonl(tc._path)
    r = records[0]
    assert r["total_tokens"] == 0
    assert r["prompt_tokens"] == 0
    assert r["completion_tokens"] == 0


def test_skills_and_tools_deduped(tmp_path: Path) -> None:
    """Duplicate tool names / skills are deduplicated and sorted."""
    tc = _collector(tmp_path)
    tc.record_turn(
        ts="2026-01-01T00:00:00Z",
        session="s1",
        channel="telegram",
        chat_id="c1",
        model="claude-3",
        usage={},
        duration_ms=100,
        stop_reason="end_turn",
        error=None,
        tools_used=["read_file", "write_file", "read_file"],
        skills=["github", "weather", "github"],
        files_touched=[],
    )
    records = _read_jsonl(tc._path)
    r = records[0]
    assert r["tools"] == ["read_file", "write_file"]
    assert r["skills"] == ["github", "weather"]
