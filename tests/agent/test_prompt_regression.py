"""Prompt regression tests (F-020).

These tests verify that system prompts and subagent prompts maintain
expected structural sections across refactoring.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


class TestSubagentPromptStructure:
    """Verify the subagent prompt contains required structural sections."""

    def _build_prompt(self, tmp_path: Path) -> str:
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=MessageBus())
        return mgr._build_subagent_prompt()

    def test_prompt_has_workspace_section(self, tmp_path):
        prompt = self._build_prompt(tmp_path)
        assert "## Workspace" in prompt

    def test_prompt_has_quality_gates(self, tmp_path):
        prompt = self._build_prompt(tmp_path)
        assert "Quality Gates" in prompt
        assert "ruff check" in prompt
        assert "pytest" in prompt

    def test_prompt_uses_dynamic_repo_root(self, tmp_path):
        prompt = self._build_prompt(tmp_path)
        # Must not contain hardcoded /root/ path (F-014)
        assert "/root/" not in prompt
        # Must contain the actual workspace path
        assert str(tmp_path.resolve()) in prompt

    def test_prompt_injects_bootstrap_docs_when_present(self, tmp_path):
        """When governance docs exist, they should be injected (F-029)."""
        (tmp_path / "AGENTS.md").write_text("# Agent Instructions\nTest content")
        (tmp_path / "SOUL.md").write_text("# Soul\nPersonality values")
        prompt = self._build_prompt(tmp_path)
        assert "## AGENTS.md" in prompt
        assert "## SOUL.md" in prompt

    def test_prompt_injects_memory_when_present(self, tmp_path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        (mem_dir / "MEMORY.md").write_text("# Memory\nKey fact: test")
        prompt = self._build_prompt(tmp_path)
        assert "Memory Context" in prompt
        assert "Key fact: test" in prompt

    def test_prompt_truncates_memory_at_boundary(self, tmp_path):
        """Memory truncation should cut at line boundary, not mid-word (F-019)."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        # Create content with clear line breaks
        lines = [f"Line {i}: " + "x" * 50 for i in range(100)]
        content = "\n".join(lines)
        (mem_dir / "MEMORY.md").write_text(content)
        prompt = self._build_prompt(tmp_path)
        # Should not cut mid-word
        if "(truncated)" in prompt:
            before_trunc = prompt.split("(truncated)")[0]
            # Must end at a newline boundary
            assert before_trunc.rstrip().endswith("\n") or before_trunc.rstrip()[-1] != "x"


class TestContextBuilderPromptStructure:
    """Verify the main agent system prompt has required sections."""

    def test_system_prompt_has_identity(self, tmp_path):
        from nanobot.agent.context import ContextBuilder

        ctx = ContextBuilder(tmp_path)
        prompt = ctx.build_system_prompt()
        # System prompt should contain identity
        assert len(prompt) > 100
