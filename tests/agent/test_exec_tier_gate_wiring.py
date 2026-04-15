"""Exec tier gate wiring tests (F-016)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.shell import ExecTool


def _make_loop_with_mocks(*, workspace):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 1024

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)

    return loop


def test_agent_loop_registers_exec_tool_with_interactive_tier(tmp_path):
    loop = _make_loop_with_mocks(workspace=tmp_path)

    exec_tool = loop.tools.get("exec")
    assert isinstance(exec_tool, ExecTool)
    assert exec_tool.exec_max_tier == 3
    assert exec_tool.exec_tier_context == "interactive"


@pytest.mark.asyncio
async def test_subagent_manager_registers_exec_tool_with_subagent_tier(tmp_path):
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    manager = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
    manager._announce_result = AsyncMock()

    captured: dict[str, ExecTool] = {}

    async def fake_run(spec):
        tool = spec.tools.get("exec")
        if isinstance(tool, ExecTool):
            captured["exec"] = tool
        return AgentRunResult(final_content="done", messages=list(spec.initial_messages))

    manager.runner.run = AsyncMock(side_effect=fake_run)

    await manager._run_subagent(
        task_id="sub-tier",
        task="run checks",
        label="subagent tier check",
        origin={"channel": "test", "chat_id": "c1", "request_id": "req-tier"},
    )

    assert "exec" in captured
    assert captured["exec"].exec_max_tier == 1
    assert captured["exec"].exec_tier_context == "subagent"
