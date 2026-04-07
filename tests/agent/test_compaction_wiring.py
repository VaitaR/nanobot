"""Tests for context compaction wiring in AgentLoop."""

from __future__ import annotations

from unittest.mock import MagicMock


def _make_history(n: int, msg_len: int = 200) -> list[dict]:
    """Create n fake history messages."""
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}: " + "word " * msg_len,
        }
        for i in range(n)
    ]


class TestMaybeCompactHistory:
    """Test _maybe_compact_history on AgentLoop."""

    def _make_loop(
        self,
        *,
        enabled: bool = True,
        token_budget: int = 2000,
        safety_margin: int = 200,
        keep_recent: int = 5,
        threshold: float = 0.75,
    ) -> MagicMock:
        """Create a mock AgentLoop with compaction wired up."""
        from nanobot_workspace.memory.compaction import CompactionPolicy, compact_messages

        loop = MagicMock()
        loop._compaction_enabled = enabled
        loop._compact_fn = compact_messages if enabled else None
        loop._compaction_policy = (
            CompactionPolicy(
                token_budget=token_budget,
                safety_margin=safety_margin,
                keep_recent=keep_recent,
                compaction_threshold=threshold,
            )
            if enabled
            else None
        )
        # Bind the real method
        from nanobot.agent.loop import AgentLoop

        loop._maybe_compact_history = lambda history: AgentLoop._maybe_compact_history(
            loop, history
        )
        return loop

    def test_empty_history_unchanged(self) -> None:
        loop = self._make_loop()
        assert loop._maybe_compact_history([]) == []

    def test_under_threshold_unchanged(self) -> None:
        loop = self._make_loop(token_budget=100_000)
        history = _make_history(5, msg_len=5)  # tiny messages
        result = loop._maybe_compact_history(history)
        assert result == history

    def test_compaction_triggers_when_over_threshold(self) -> None:
        loop = self._make_loop(token_budget=2000, keep_recent=3)
        # Create enough messages to exceed budget
        history = _make_history(30, msg_len=50)
        result = loop._maybe_compact_history(history)
        # Should be compacted — result shorter than history
        assert len(result) < len(history)
        # Last 3 should be preserved
        assert result[-3:] == history[-3:]

    def test_compaction_summary_is_system_message(self) -> None:
        loop = self._make_loop(token_budget=2000, keep_recent=3)
        history = _make_history(20, msg_len=100)
        result = loop._maybe_compact_history(history)
        assert result[0]["role"] == "system"
        assert "Compacted" in result[0]["content"]

    def test_disabled_returns_history_unchanged(self) -> None:
        loop = self._make_loop(enabled=False, token_budget=100)
        history = _make_history(100, msg_len=100)
        result = loop._maybe_compact_history(history)
        assert result is history

    def test_graceful_degradation_on_error(self) -> None:
        loop = self._make_loop(token_budget=100)
        loop._compact_fn = MagicMock(side_effect=RuntimeError("boom"))
        history = _make_history(20, msg_len=50)
        # Should return history unchanged on error
        result = loop._maybe_compact_history(history)
        assert result == history

    def test_compaction_fn_none_returns_history(self) -> None:
        loop = self._make_loop()
        loop._compact_fn = None
        history = _make_history(50, msg_len=100)
        result = loop._maybe_compact_history(history)
        assert result == history

    def test_multimodal_content_handled(self) -> None:
        loop = self._make_loop(token_budget=2000, keep_recent=2)
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "Look at this image"},
                ],
            },
            {"role": "assistant", "content": "I see an image"},
        ] * 20  # repeat enough to exceed budget
        result = loop._maybe_compact_history(history)
        # Should not crash
        assert isinstance(result, list)
        assert len(result) > 0


class TestCompactionConfig:
    """Test CompactionConfig schema."""

    def test_default_values(self) -> None:
        from nanobot.config.schema import CompactionConfig

        cfg = CompactionConfig()
        assert cfg.enabled is True
        assert cfg.token_budget == 0  # auto
        assert cfg.safety_margin == 8_000
        assert cfg.keep_recent == 10
        assert cfg.compaction_threshold == 0.75

    def test_custom_values(self) -> None:
        from nanobot.config.schema import CompactionConfig

        cfg = CompactionConfig(
            enabled=False,
            token_budget=128_000,
            keep_recent=20,
        )
        assert cfg.enabled is False
        assert cfg.token_budget == 128_000
        assert cfg.keep_recent == 20

    def test_config_in_agent_defaults(self) -> None:
        from nanobot.config.schema import AgentDefaults, CompactionConfig

        defaults = AgentDefaults()
        assert isinstance(defaults.compaction, CompactionConfig)
        assert defaults.compaction.enabled is True

    def test_config_from_dict(self) -> None:
        from nanobot.config.schema import CompactionConfig

        cfg = CompactionConfig.model_validate({"enabled": False, "tokenBudget": 50000})
        assert cfg.enabled is False
        assert cfg.token_budget == 50000  # snake_case alias
