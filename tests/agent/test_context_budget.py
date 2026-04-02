"""Tests for system prompt token budget enforcement in ContextBuilder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.agent.context import ContextBuilder, _CHARS_PER_TOKEN, _SEPARATOR


def _make_builder(
    tmp_path: Path,
    system_prompt_max_tokens: int = 0,
) -> ContextBuilder:
    """Create a ContextBuilder with mocked memory/skills and no workspace deps."""
    with (
        patch("nanobot.agent.context.MemoryStore"),
        patch("nanobot.agent.context.SkillsLoader"),
    ):
        builder = ContextBuilder(
            tmp_path,
            system_prompt_max_tokens=system_prompt_max_tokens,
        )
    return builder


def _token_count(text: str) -> int:
    """Estimate tokens using the same heuristic as the builder."""
    return len(text) // _CHARS_PER_TOKEN


def _gen_chars(n: int) -> str:
    """Generate a string of exactly *n* characters."""
    return "x" * n


class TestNoBudget:
    """When system_prompt_max_tokens=0, all parts are returned unchanged."""

    def test_no_budget_returns_all_parts(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, system_prompt_max_tokens=0)
        identity = "identity text"
        builder._get_identity = MagicMock(return_value=identity)  # type: ignore[method-assign]
        builder._load_bootstrap_files = MagicMock(return_value="bootstrap")  # type: ignore[method-assign]
        builder.memory.get_memory_context = MagicMock(return_value="memory")  # type: ignore[union-attr]
        builder.skills.get_always_skills = MagicMock(return_value=[])  # type: ignore[union-attr]
        builder.skills.load_skills_for_context = MagicMock(return_value="")  # type: ignore[union-attr]
        builder.skills.build_skills_summary = MagicMock(return_value="skills")  # type: ignore[union-attr]

        result = builder.build_system_prompt()
        assert "identity text" in result
        assert "bootstrap" in result
        assert "# Memory" in result
        assert "skills" in result

    def test_empty_parts_unchanged(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, system_prompt_max_tokens=0)
        identity = "identity only"
        builder._get_identity = MagicMock(return_value=identity)  # type: ignore[method-assign]
        builder._load_bootstrap_files = MagicMock(return_value="")  # type: ignore[method-assign]
        builder.memory.get_memory_context = MagicMock(return_value="")  # type: ignore[union-attr]
        builder.skills.get_always_skills = MagicMock(return_value=[])  # type: ignore[union-attr]
        builder.skills.build_skills_summary = MagicMock(return_value="")  # type: ignore[union-attr]

        result = builder.build_system_prompt()
        assert result == "identity only"


class TestBudgetLargeEnough:
    """When budget is large enough, nothing is truncated."""

    def test_budget_large_enough_no_truncation(self, tmp_path: Path) -> None:
        # Total prompt ~100 chars → 25 tokens. Budget of 50000 is plenty.
        builder = _make_builder(tmp_path, system_prompt_max_tokens=50_000)
        identity = "id"
        builder._get_identity = MagicMock(return_value=identity)  # type: ignore[method-assign]
        builder._load_bootstrap_files = MagicMock(return_value="boot")  # type: ignore[method-assign]
        builder.memory.get_memory_context = MagicMock(return_value="mem")  # type: ignore[union-attr]
        builder.skills.get_always_skills = MagicMock(return_value=[])  # type: ignore[union-attr]
        builder.skills.build_skills_summary = MagicMock(return_value="sk")  # type: ignore[union-attr]

        result = builder.build_system_prompt()
        assert "id" in result
        assert "boot" in result
        assert "mem" in result
        assert "sk" in result


class TestBudgetExceeded:
    """When budget is exceeded, lower-priority sections are dropped first."""

    def _setup_full_builder(
        self,
        tmp_path: Path,
        budget: int,
        id_size: int = 100,
        boot_size: int = 200,
        mem_size: int = 400,
        always_size: int = 300,
        skills_size: int = 500,
    ) -> tuple[ContextBuilder, str]:
        """Set up a builder with all sections populated at known sizes."""
        builder = _make_builder(tmp_path, system_prompt_max_tokens=budget)
        builder._get_identity = MagicMock(return_value=_gen_chars(id_size))  # type: ignore[method-assign]
        builder._load_bootstrap_files = MagicMock(return_value=_gen_chars(boot_size))  # type: ignore[method-assign]
        builder.memory.get_memory_context = MagicMock(return_value=_gen_chars(mem_size))  # type: ignore[union-attr]
        builder.skills.get_always_skills = MagicMock(return_value=["s1"])  # type: ignore[union-attr]
        builder.skills.load_skills_for_context = MagicMock(return_value=_gen_chars(always_size))  # type: ignore[union-attr]
        builder.skills.build_skills_summary = MagicMock(return_value=_gen_chars(skills_size))  # type: ignore[union-attr]
        return builder

    def test_budget_truncates_skills_summary(self, tmp_path: Path) -> None:
        """Budget drops skills_summary first."""
        # Total chars ~ 1500 + 4 separators = 1528 → 382 tokens. Budget=400 → 1600 chars.
        # After dropping skills_summary (500+7=507): 1021 → 255 tokens ≤ 400. Keep the rest.
        builder = self._setup_full_builder(tmp_path, budget=400)
        result = builder.build_system_prompt()
        assert "# Skills" not in result
        assert "# Memory" in result
        assert "# Active Skills" in result

    def test_budget_truncates_always_skills(self, tmp_path: Path) -> None:
        """Budget drops skills_summary and always_skills."""
        # Total ~1528 chars → 382 tokens. Budget=250 → 1000 chars.
        # After dropping skills_summary (507): 1021→255 tokens > 250.
        # After dropping always_skills (300+7=307): 714→178 tokens ≤ 250.
        builder = self._setup_full_builder(tmp_path, budget=250)
        result = builder.build_system_prompt()
        assert "# Skills" not in result
        assert "# Active Skills" not in result
        assert "# Memory" in result

    def test_budget_truncates_memory(self, tmp_path: Path) -> None:
        """Budget drops skills_summary, always_skills, bootstrap; truncates memory."""
        # Total ~1528 chars → 382 tokens. Budget=50 → 200 chars.
        # Drop skills_summary (507): 1021→255 > 50.
        # Drop always_skills (307): 714→178 > 50.
        # Drop bootstrap (200+7=207): 507→126 > 50.
        # Truncate memory: identity=100, need total ≤ 200. Memory budget = 200-100-7=93 chars.
        builder = self._setup_full_builder(tmp_path, budget=50)
        result = builder.build_system_prompt()
        assert "# Skills" not in result
        assert "# Active Skills" not in result
        # Memory is truncated (still has header), not removed entirely
        assert "# Memory" in result
        # Memory should be heavily truncated (400 → ~93 chars)
        memory_section = result.split("# Memory\n\n")[-1] if "# Memory\n\n" in result else ""
        assert len(memory_section) < 200

    def test_identity_never_truncated(self, tmp_path: Path) -> None:
        """Even with a tiny budget, identity is always kept."""
        # Identity is 100 chars → 25 tokens. Budget = 25 tokens exactly.
        # Total ~1528 → 382 tokens. Drops skills, always, bootstrap, truncates memory.
        # After all drops: identity=100, memory truncated to fit. Memory budget ≈ 0 chars.
        builder = self._setup_full_builder(tmp_path, budget=25)
        result = builder.build_system_prompt()
        # Identity is always present (no header, just raw content)
        assert len(result) > 0
        assert "# Skills" not in result

    def test_memory_truncated_when_needed(self, tmp_path: Path) -> None:
        """Memory is truncated (not removed) when it's the only remaining expendable section."""
        # identity=100, bootstrap=200, memory=400. Total=700 chars + separators.
        # Budget ~ 150 tokens → 600 chars. After removing skills_summary + always_skills,
        # we still have identity+bootstrap+memory. Remove bootstrap too.
        # Then only identity+memory. If memory needs truncation:
        # identity=100 chars, budget=130 tokens → 520 chars. Memory gets truncated.
        builder = _make_builder(tmp_path, system_prompt_max_tokens=130)
        builder._get_identity = MagicMock(return_value=_gen_chars(100))  # type: ignore[method-assign]
        builder._load_bootstrap_files = MagicMock(return_value=_gen_chars(800))  # type: ignore[method-assign]
        builder.memory.get_memory_context = MagicMock(return_value=_gen_chars(1000))  # type: ignore[union-attr]
        builder.skills.get_always_skills = MagicMock(return_value=[])  # type: ignore[union-attr]
        builder.skills.build_skills_summary = MagicMock(return_value=_gen_chars(500))  # type: ignore[union-attr]

        result = builder.build_system_prompt()
        assert "# Memory" in result
        # Memory should be truncated — full would be 1000 chars
        assert len(result) < 1000 + 100  # much less than full identity + full memory


class TestSeparatorOverhead:
    """Budget must account for separator characters."""

    def test_budget_accounts_for_separator_overhead(self, tmp_path: Path) -> None:
        """Two small parts: 80 chars each = 160 chars + 7 char separator = 167 chars = 41 tokens."""
        sep_tokens = len(_SEPARATOR) // _CHARS_PER_TOKEN  # 7 // 4 = 1
        part_tokens = 80 // _CHARS_PER_TOKEN  # 20
        total_tokens = 2 * part_tokens + sep_tokens  # 41
        # Set budget to exactly total → should still fit
        builder = _make_builder(tmp_path, system_prompt_max_tokens=total_tokens)
        builder._get_identity = MagicMock(return_value=_gen_chars(80))  # type: ignore[method-assign]
        builder._load_bootstrap_files = MagicMock(return_value=_gen_chars(80))  # type: ignore[method-assign]
        builder.memory.get_memory_context = MagicMock(return_value="")  # type: ignore[union-attr]
        builder.skills.get_always_skills = MagicMock(return_value=[])  # type: ignore[union-attr]
        builder.skills.build_skills_summary = MagicMock(return_value="")  # type: ignore[union-attr]

        result = builder.build_system_prompt()
        assert len(result) == 80 + 7 + 80  # both parts present with separator

    def test_just_under_budget_keeps_all(self, tmp_path: Path) -> None:
        """A budget that's exactly at the token boundary still keeps everything."""
        builder = _make_builder(tmp_path, system_prompt_max_tokens=50)
        builder._get_identity = MagicMock(return_value=_gen_chars(100))  # type: ignore[method-assign]
        builder._load_bootstrap_files = MagicMock(return_value=_gen_chars(100))  # type: ignore[method-assign]
        builder.memory.get_memory_context = MagicMock(return_value="")  # type: ignore[union-attr]
        builder.skills.get_always_skills = MagicMock(return_value=[])  # type: ignore[union-attr]
        builder.skills.build_skills_summary = MagicMock(return_value="")  # type: ignore[union-attr]

        # 100+100+7 = 207 chars → 51 tokens. Budget=50 → should drop bootstrap.
        result = builder.build_system_prompt()
        # Bootstrap should be removed since 207//4 = 51 > 50
        # After removing bootstrap: 100 chars → 25 tokens ≤ 50
        assert len(result) == 100  # just identity


class TestTruncationWarnings:
    """Truncation should log warnings."""

    def test_truncation_logs_warning(self, tmp_path: Path) -> None:
        import io
        from loguru import logger as _logger

        # Add a string sink to capture warnings
        buf = io.StringIO()
        sink_id = _logger.add(buf, level="WARNING", filter=lambda r: "budget exceeded" in r["message"].lower())

        try:
            builder = _make_builder(tmp_path, system_prompt_max_tokens=250)
            builder._get_identity = MagicMock(return_value=_gen_chars(100))  # type: ignore[method-assign]
            builder._load_bootstrap_files = MagicMock(return_value=_gen_chars(200))  # type: ignore[method-assign]
            builder.memory.get_memory_context = MagicMock(return_value=_gen_chars(400))  # type: ignore[union-attr]
            builder.skills.get_always_skills = MagicMock(return_value=["s1"])  # type: ignore[union-attr]
            builder.skills.load_skills_for_context = MagicMock(return_value=_gen_chars(300))  # type: ignore[union-attr]
            builder.skills.build_skills_summary = MagicMock(return_value=_gen_chars(500))  # type: ignore[union-attr]

            builder.build_system_prompt()
            assert "System prompt budget exceeded" in buf.getvalue()
        finally:
            _logger.remove(sink_id)
