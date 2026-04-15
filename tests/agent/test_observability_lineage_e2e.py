"""End-to-end lineage test for interactive -> spawn -> subagent telemetry (F-001/F-002)."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus


@pytest.mark.asyncio
async def test_interactive_spawn_subagent_lineage_request_id_reconstructible(tmp_path):
    from nanobot.agent.loop import AgentLoop

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 1024

    with patch("nanobot.agent.loop.ContextBuilder"), patch("nanobot.agent.loop.SessionManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path)

    # Keep the test deterministic and avoid unrelated side effects.
    loop.context.build_messages.return_value = [{"role": "user", "content": "start"}]
    fake_session = MagicMock()
    fake_session.get_history.return_value = []
    loop.sessions.get_or_create.return_value = fake_session
    loop.sessions.save = MagicMock()
    loop._save_turn = MagicMock()
    loop.signal_detector = None
    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)

    async def fake_spawn(**kwargs):
        loop.subagents._record_subagent_telemetry(
            task_id="sub-e2e",
            model="test-model",
            start_time=time.monotonic(),
            stop_reason="completed",
            usage={"prompt_tokens": 3, "completion_tokens": 2},
            origin={
                "channel": kwargs["origin_channel"],
                "chat_id": kwargs["origin_chat_id"],
                "request_id": kwargs.get("request_id"),
            },
        )
        return "sub-e2e"

    loop.subagents.spawn = AsyncMock(side_effect=fake_spawn)

    async def fake_run(spec):
        spawn_result = await spec.tools.execute("spawn", {"task": "collect evidence"})
        return AgentRunResult(
            final_content=f"spawned {spawn_result}",
            messages=list(spec.initial_messages),
            tools_used=["spawn"],
            usage={"prompt_tokens": 7, "completion_tokens": 4},
            stop_reason="completed",
            tool_events=[{"name": "spawn", "status": "ok", "detail": str(spawn_result)}],
        )

    loop.runner.run = AsyncMock(side_effect=fake_run)

    request_id = "req-lineage-e2e"
    msg = InboundMessage(
        channel="test",
        sender_id="u1",
        chat_id="c1",
        content="spawn background task",
        request_id=request_id,
    )

    response = await loop._process_message(msg)

    assert response is not None
    assert "spawned" in response.content

    log_path = tmp_path / "memory" / "improvement-log.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    by_request = [row for row in rows if row.get("request_id") == request_id]
    sessions = {row["session"] for row in by_request}

    assert msg.session_key in sessions
    assert "subagent:sub-e2e" in sessions
