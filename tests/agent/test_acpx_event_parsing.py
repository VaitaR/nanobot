"""Tests for ACPX JSON event parsing details."""

from __future__ import annotations

import json


def test_parse_acpx_json_output_captures_tool_completion_and_usage() -> None:
    from nanobot.agent.execution import _parse_acpx_json_output

    lines = [
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "content": {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "read_file",
                        "arguments": {"path": "README.md"},
                        "status": "in_progress",
                    }
                }
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8},
                    "content": {
                        "type": "tool_result",
                        "toolCallId": "call-1",
                        "status": "completed",
                        "result": "ok",
                        "durationMs": 42,
                    },
                }
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "stopReason": "end_turn",
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            },
        },
    ]

    result = _parse_acpx_json_output("\n".join(json.dumps(line) for line in lines), "", 1.0)

    assert result.success is True
    assert result.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
    }
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "read_file"
    assert tool_call.status == "completed"
    assert tool_call.duration_ms == 42


def test_parse_acpx_json_output_creates_tool_from_result_without_prior_call() -> None:
    from nanobot.agent.execution import _parse_acpx_json_output

    lines = [
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "content": {
                        "type": "tool_result",
                        "toolCallId": "call-2",
                        "toolName": "exec_command",
                        "status": "failed",
                        "error": "boom",
                    }
                }
            },
        },
        {"jsonrpc": "2.0", "id": 1, "result": {"stopReason": "turn_failed"}},
    ]

    result = _parse_acpx_json_output("\n".join(json.dumps(line) for line in lines), "", 1.0)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "exec_command"
    assert tool_call.status == "failed"
