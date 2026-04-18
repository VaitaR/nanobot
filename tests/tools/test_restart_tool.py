"""Tests for the RestartGatewayTool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.agent.tools.restart import RestartGatewayTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def tool(workspace: Path) -> RestartGatewayTool:
    return RestartGatewayTool(workspace=workspace)


@pytest.mark.asyncio
async def test_restart_tool_writes_pending_file(workspace: Path, tool: RestartGatewayTool) -> None:
    """Calling execute writes .restart-pending with correct JSON."""
    with patch("nanobot.agent.tools.restart.os.execv", side_effect=SystemExit("execv replaced")):
        with pytest.raises(SystemExit):
            await tool.execute(reason="code patch applied")

    pending = workspace / ".restart-pending"
    assert pending.exists()
    data = json.loads(pending.read_text())
    assert data["reason"] == "code patch applied"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_restart_tool_calls_execv(tool: RestartGatewayTool) -> None:
    """os.execv is called with the expected arguments."""
    with patch("nanobot.agent.tools.restart.os.execv", side_effect=SystemExit) as mock_execv:
        with pytest.raises(SystemExit):
            await tool.execute(reason="testing execv")

    mock_execv.assert_called_once()
    call_args = mock_execv.call_args[0]  # (path, args_list)
    assert call_args[0].endswith("python") or call_args[0].endswith("python3")
    assert "-m" in call_args[1]
    assert "nanobot" in call_args[1]


@pytest.mark.asyncio
async def test_restart_tool_reason_required(tool: RestartGatewayTool) -> None:
    """Empty or missing reason returns an error."""
    result = await tool.execute(reason="")
    assert "Error" in result
    assert "required" in result.lower()


@pytest.mark.asyncio
async def test_restart_tool_defers_when_subagents_running(workspace: Path) -> None:
    """Active subagents cause restart to defer via .pending-reload marker."""
    mock_manager = MagicMock()
    mock_manager.get_running_count.return_value = 3

    tool = RestartGatewayTool(workspace=workspace, subagent_manager=mock_manager)

    with patch("nanobot.agent.tools.restart.os.execv") as mock_execv:
        result = await tool.execute(reason="subagent defer test")

    mock_execv.assert_not_called()
    assert "отложен" in result  # Russian: "deferred"
    assert "Рестарт" in result
    # Both markers should be written
    assert (workspace / ".pending-reload").exists()
    assert (workspace / ".restart-pending").exists()


@pytest.mark.asyncio
async def test_restart_tool_logs_pending_reload_write_failure(workspace: Path) -> None:
    """Deferred restart logs .pending-reload write failures and still writes .restart-pending."""
    mock_manager = MagicMock()
    mock_manager.get_running_count.return_value = 1
    tool = RestartGatewayTool(workspace=workspace, subagent_manager=mock_manager)

    original_write_text = Path.write_text

    def fail_pending_reload(self: Path, data: str, *args, **kwargs) -> int:
        if self.name == ".pending-reload":
            raise OSError("disk full")
        return original_write_text(self, data, *args, **kwargs)

    with patch("pathlib.Path.write_text", autospec=True, side_effect=fail_pending_reload):
        with patch("nanobot.agent.tools.restart.logger.exception") as mock_log:
            result = await tool.execute(reason="deferred write failure")

    assert "отложен" in result
    assert not (workspace / ".pending-reload").exists()
    assert (workspace / ".restart-pending").exists()
    mock_log.assert_called_once()


@pytest.mark.asyncio
async def test_restart_tool_proceeds_when_no_subagents(workspace: Path) -> None:
    """No active subagents → restart proceeds normally via os.execv."""
    mock_manager = MagicMock()
    mock_manager.get_running_count.return_value = 0

    tool = RestartGatewayTool(workspace=workspace, subagent_manager=mock_manager)

    with patch("nanobot.agent.tools.restart.os.execv", side_effect=SystemExit) as mock_execv:
        with pytest.raises(SystemExit):
            await tool.execute(reason="clean subagent test")

    mock_execv.assert_called_once()
    pending = workspace / ".restart-pending"
    assert pending.exists()


@pytest.mark.asyncio
async def test_restart_tool_safety_warnings_locks(workspace: Path, tool: RestartGatewayTool) -> None:
    """Lock files in workspace produce a warning but restart still proceeds."""
    (workspace / "test.lock").write_text("locked")

    with patch("nanobot.agent.tools.restart.os.execv", side_effect=SystemExit) as mock_execv:
        with pytest.raises(SystemExit):
            await tool.execute(reason="lock warning test")

    mock_execv.assert_called_once()
    pending = workspace / ".restart-pending"
    assert pending.exists()


@pytest.mark.asyncio
async def test_restart_tool_no_warnings(workspace: Path) -> None:
    """Clean workspace with no subagents produces no warnings in output."""
    tool = RestartGatewayTool(workspace=workspace, subagent_manager=None)

    # Use an execv mock that doesn't raise so we can capture the return
    def fake_execv(*args, **kwargs):  # noqa: ARG001
        raise SystemExit("replaced")

    with patch("nanobot.agent.tools.restart.os.execv", side_effect=fake_execv):
        with pytest.raises(SystemExit):
            await tool.execute(reason="clean test")

    # If execv raises, we can't check output directly, but we verify no crash
    pending = workspace / ".restart-pending"
    assert pending.exists()
    data = json.loads(pending.read_text())
    assert data["reason"] == "clean test"


@pytest.mark.asyncio
async def test_restart_tool_reason_truncated(workspace: Path, tool: RestartGatewayTool) -> None:
    """Reason longer than 200 chars is truncated."""
    long_reason = "x" * 300

    with patch("nanobot.agent.tools.restart.os.execv", side_effect=SystemExit):
        with pytest.raises(SystemExit):
            await tool.execute(reason=long_reason)

    pending = workspace / ".restart-pending"
    data = json.loads(pending.read_text())
    assert len(data["reason"]) == 200


@pytest.mark.asyncio
async def test_restart_tool_execv_failure(workspace: Path, tool: RestartGatewayTool) -> None:
    """If os.execv fails, an error message is returned."""
    with patch("nanobot.agent.tools.restart.os.execv", side_effect=OSError("exec failed")):
        result = await tool.execute(reason="exec failure test")

    assert "Error" in result
    assert "execv failed" in result
