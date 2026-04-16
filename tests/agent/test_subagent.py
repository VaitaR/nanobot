"""Tests for SubagentManager._build_subagent_prompt() memory injection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_subagent_manager(tmp_path):
    """Create a SubagentManager with a mock provider for prompt-building tests."""
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return SubagentManager(provider=provider, workspace=tmp_path, bus=bus)


class TestBuildSubagentPromptMemory:
    """Verify _build_subagent_prompt injects MEMORY.md content correctly."""

    def test_includes_memory_content_when_present(self, tmp_path):
        mgr = _make_subagent_manager(tmp_path)
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "MEMORY.md").write_text(
            "## Preferences\n- Always use pytest\n\n## Facts\n- Project uses Python 3.12\n",
            encoding="utf-8",
        )

        prompt = mgr._build_subagent_prompt()

        assert "## Memory Context" in prompt
        assert "Always use pytest" in prompt
        assert "Python 3.12" in prompt

    def test_no_memory_section_when_missing(self, tmp_path):
        mgr = _make_subagent_manager(tmp_path)
        # Do NOT create memory/MEMORY.md

        prompt = mgr._build_subagent_prompt()

        assert "## Memory Context" not in prompt

    def test_truncation_of_long_memory(self, tmp_path):
        mgr = _make_subagent_manager(tmp_path)
        (tmp_path / "memory").mkdir()
        long_content = "A" * 5000
        (tmp_path / "memory" / "MEMORY.md").write_text(long_content, encoding="utf-8")

        prompt = mgr._build_subagent_prompt()

        assert "## Memory Context" in prompt
        assert "AAA" in prompt
        assert "...(truncated)" in prompt

        # Extract just the memory section content
        memory_section_start = prompt.index("## Memory Context\n\n") + len("## Memory Context\n\n")
        memory_section_end = prompt.index("\n\n## Skills") if "\n\n## Skills" in prompt else len(prompt)
        memory_text = prompt[memory_section_start:memory_section_end]
        # Should be roughly 2000 + truncation marker
        assert len(memory_text) < 2100
        assert len(memory_text) > 2000

    def test_empty_memory_file_omits_section(self, tmp_path):
        mgr = _make_subagent_manager(tmp_path)
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "MEMORY.md").write_text("   \n  \n", encoding="utf-8")

        prompt = mgr._build_subagent_prompt()

        # Empty/whitespace-only memory should be stripped and falsy → no section
        assert "## Memory Context" not in prompt

    def test_workspace_and_skills_still_present(self, tmp_path):
        mgr = _make_subagent_manager(tmp_path)
        (tmp_path / "memory").mkdir()
        (tmp_path / "memory" / "MEMORY.md").write_text("some memory", encoding="utf-8")

        prompt = mgr._build_subagent_prompt()

        assert str(tmp_path) in prompt
        assert "## Workspace" in prompt
        assert "# Subagent" in prompt


@pytest.mark.asyncio
async def test_subagent_exec_tool_is_tier_capped(tmp_path):
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import LLMResponse
    from nanobot_workspace.agent.exec_tier_gate import DEFAULT_SUBAGENT_MAX_TIER

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done", tool_calls=[]))
    mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)
    mgr._announce_result = AsyncMock()

    captured: dict[str, object] = {}

    class FakeExecTool:
        name = "exec"
        description = ""
        parameters = {}

        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("nanobot.agent.subagent.ExecTool", FakeExecTool):
        await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

    assert captured["max_tier"] == DEFAULT_SUBAGENT_MAX_TIER
    assert captured["exec_context"] == "subagent"


@pytest.mark.asyncio
async def test_spawn_blocks_claude_zai_during_zai_peak(tmp_path):
    from nanobot.agent.subagent import PeakHoursSpawnBlockedError

    mgr = _make_subagent_manager(tmp_path)

    with patch("nanobot.agent.subagent.is_zai_peak", return_value=True):
        with pytest.raises(PeakHoursSpawnBlockedError, match="claude-zai"):
            await mgr.spawn("investigate throttling", executor="claude-zai")


@pytest.mark.asyncio
async def test_spawn_blocks_claude_native_during_claude_peak(tmp_path):
    from nanobot.agent.subagent import PeakHoursSpawnBlockedError

    mgr = _make_subagent_manager(tmp_path)

    with patch("nanobot.agent.subagent.is_claude_peak", return_value=True):
        with pytest.raises(PeakHoursSpawnBlockedError, match="claude-native"):
            await mgr.spawn("review changes", executor="claude-native")
