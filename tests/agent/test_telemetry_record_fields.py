"""Tests for telemetry record field completeness (request_id + cost)."""

from __future__ import annotations

import json
from pathlib import Path

from nanobot.agent.telemetry import TelemetryCollector


def test_record_turn_persists_request_and_cost(tmp_path: Path) -> None:
    collector = TelemetryCollector(tmp_path)

    collector.record_turn(
        ts="2026-04-15T00:00:00Z",
        session="session:test",
        channel="telegram",
        chat_id="123",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        duration_ms=123,
        stop_reason="completed",
        error=None,
        tools_used=["spawn"],
        skills=["coder"],
        files_touched=["/tmp/x.py"],
        request_id="req-123",
        estimated_cost_usd=0.001,
    )

    log_path = tmp_path / "memory" / "improvement-log.jsonl"
    row = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])

    assert row["request_id"] == "req-123"
    assert row["estimated_cost_usd"] == 0.001
    assert row["total_tokens"] == 15
