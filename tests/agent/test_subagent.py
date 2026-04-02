"""Tests for SubagentManager._build_subagent_prompt() memory injection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
