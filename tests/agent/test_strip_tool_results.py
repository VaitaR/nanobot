"""Tests for strip_old_tool_results — cheap context-token savings."""

from nanobot.agent.memory import (
    _TOOL_RESULT_STRIP_PLACEHOLDER,
    strip_old_tool_results,
)


def _tool_msg(call_id: str, name: str, content: str) -> dict:
    """Create a tool-role message with metadata."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def _user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------

class TestNoOp:
    """strip_old_tool_results should do nothing when under threshold."""

    def test_empty_messages(self) -> None:
        msgs: list[dict] = []
        assert strip_old_tool_results(msgs) == 0
        assert msgs == []

    def test_below_threshold(self) -> None:
        msgs = [_tool_msg("c1", "x", "data")]
        assert strip_old_tool_results(msgs, keep_recent=50) == 0
        assert msgs[0]["content"] == "data"

    def test_exactly_at_threshold(self) -> None:
        """len == keep_recent → no stripping."""
        msgs = [_tool_msg(f"c{i}", "x", f"data{i}") for i in range(50)]
        assert strip_old_tool_results(msgs, keep_recent=50) == 0
        assert all(m["content"].startswith("data") for m in msgs)

    def test_no_tool_messages(self) -> None:
        """Only user/assistant messages → nothing to strip."""
        msgs = [_user_msg("hi"), _assistant_msg("hello")] * 30
        assert strip_old_tool_results(msgs, keep_recent=50) == 0

    def test_tool_messages_but_all_within_recent_window(self) -> None:
        msgs = [_tool_msg(f"c{i}", "x", f"data{i}") for i in range(50)]
        assert strip_old_tool_results(msgs, keep_recent=50) == 0


# ---------------------------------------------------------------------------
# Stripping behavior
# ---------------------------------------------------------------------------

class TestStripping:
    """Verify that only old tool-role messages get stripped."""

    def test_one_over_threshold(self) -> None:
        msgs = [_tool_msg("c0", "x", "old data"), _user_msg("recent")]
        stripped = strip_old_tool_results(msgs, keep_recent=1)
        assert stripped == 1
        assert msgs[0]["content"] == _TOOL_RESULT_STRIP_PLACEHOLDER
        assert msgs[1]["content"] == "recent"

    def test_preserves_metadata(self) -> None:
        """tool_call_id and name must survive stripping."""
        msgs = [
            {"role": "tool", "tool_call_id": "abc_123", "name": "read_file",
             "content": "very long file content here", "timestamp": "2025-01-01"},
            _user_msg("next"),
        ]
        strip_old_tool_results(msgs, keep_recent=1)
        assert msgs[0]["tool_call_id"] == "abc_123"
        assert msgs[0]["name"] == "read_file"
        assert msgs[0]["timestamp"] == "2025-01-01"
        assert msgs[0]["content"] == _TOOL_RESULT_STRIP_PLACEHOLDER

    def test_user_and_assistant_untouched(self) -> None:
        long_content = "x" * 5000
        msgs = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
            _tool_msg("c1", "x", "tool data"),
            _user_msg("recent"),
        ]
        strip_old_tool_results(msgs, keep_recent=1)
        assert msgs[0]["content"] == long_content
        assert msgs[1]["content"] == long_content
        assert msgs[2]["content"] == _TOOL_RESULT_STRIP_PLACEHOLDER
        assert msgs[3]["content"] == "recent"

    def test_recent_tool_messages_preserved(self) -> None:
        """The last N messages' tool results must remain intact."""
        old = [_tool_msg(f"old_{i}", "x", f"old_data_{i}") for i in range(10)]
        recent = [_tool_msg(f"new_{i}", "x", f"new_data_{i}") for i in range(5)]
        msgs = old + recent  # 15 total
        stripped = strip_old_tool_results(msgs, keep_recent=5)
        assert stripped == 10
        for m in old:
            assert m["content"] == _TOOL_RESULT_STRIP_PLACEHOLDER
        for m in recent:
            assert m["content"].startswith("new_data_")

    def test_custom_keep_recent(self) -> None:
        msgs = [_tool_msg(f"c{i}", "x", f"d{i}") for i in range(20)]
        stripped = strip_old_tool_results(msgs, keep_recent=15)
        assert stripped == 5
        for i in range(5):
            assert msgs[i]["content"] == _TOOL_RESULT_STRIP_PLACEHOLDER
        for i in range(5, 20):
            assert msgs[i]["content"] == f"d{i}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and idempotency."""

    def test_idempotent(self) -> None:
        """Running strip twice on the same list doesn't double-count."""
        msgs = [_tool_msg("c0", "x", "data"), _user_msg("recent")]
        assert strip_old_tool_results(msgs, keep_recent=1) == 1
        assert strip_old_tool_results(msgs, keep_recent=1) == 0

    def test_tool_content_none(self) -> None:
        """Tool message with None content should be skipped (not crash)."""
        msgs = [
            {"role": "tool", "tool_call_id": "c1", "name": "x", "content": None},
            _user_msg("recent"),
        ]
        stripped = strip_old_tool_results(msgs, keep_recent=1)
        assert stripped == 0
        assert msgs[0]["content"] is None

    def test_mixed_roles_realistic(self) -> None:
        """Simulate a realistic conversation with interleaved roles."""
        msgs: list[dict] = []
        for i in range(8):
            msgs.append(_user_msg(f"question {i}"))
            msgs.append(_assistant_msg(f"thinking {i}"))
            msgs.append(_tool_msg(f"c{i}", "read_file", f"file content {i}" * 100))
        # 24 messages total.  Keep last 10, so first 14 are candidates.
        stripped = strip_old_tool_results(msgs, keep_recent=10)
        # Old range is [0:14]. Tool messages at indices 2, 5, 8, 11 → 4 stripped.
        assert stripped == 4
        # User/assistant in old range untouched
        assert msgs[0]["content"] == "question 0"
        assert msgs[1]["content"] == "thinking 0"
        # Old tool messages stripped
        assert msgs[2]["content"] == _TOOL_RESULT_STRIP_PLACEHOLDER
        # Recent messages untouched (indices 14+)
        assert msgs[14]["content"] == "file content 4" * 100
        assert msgs[23]["content"] == "file content 7" * 100

    def test_custom_placeholder(self) -> None:
        msgs = [_tool_msg("c0", "x", "data"), _user_msg("recent")]
        strip_old_tool_results(msgs, keep_recent=1, placeholder="[stripped]")
        assert msgs[0]["content"] == "[stripped]"

    def test_returns_count(self) -> None:
        msgs = [_tool_msg(f"c{i}", "x", f"d{i}") for i in range(100)]
        count = strip_old_tool_results(msgs, keep_recent=50)
        assert count == 50

    def test_default_keep_recent(self) -> None:
        """Default keep_recent should be 50."""
        msgs = [_tool_msg(f"c{i}", "x", f"d{i}") for i in range(51)]
        count = strip_old_tool_results(msgs)  # uses default keep_recent=50
        assert count == 1
        assert msgs[0]["content"] == _TOOL_RESULT_STRIP_PLACEHOLDER
        assert msgs[50]["content"] == "d50"
