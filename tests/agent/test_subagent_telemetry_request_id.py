"""Regression tests for subagent telemetry request correlation (F-001/F-002)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from nanobot.agent.subagent import SubagentManager


class _DummyProvider:
    def get_default_model(self) -> str:
        return "dummy-model"


class _DummyBus:
    pass


def test_subagent_telemetry_includes_request_id(tmp_path: Path) -> None:
    manager = SubagentManager(provider=_DummyProvider(), workspace=tmp_path, bus=_DummyBus())

    manager._record_subagent_telemetry(
        task_id="abc123",
        model="gpt-4o-mini",
        start_time=time.monotonic(),
        stop_reason="completed",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        origin={"channel": "test", "chat_id": "chat1", "request_id": "req-xyz"},
    )

    log_path = tmp_path / "memory" / "improvement-log.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])

    assert record["session"] == "subagent:abc123"
    assert record["request_id"] == "req-xyz"
    assert record["estimated_cost_usd"] is not None
