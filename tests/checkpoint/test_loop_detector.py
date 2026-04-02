"""Tests for the LoopDetector."""

from __future__ import annotations

from nanobot.checkpoint.loop_detector import LoopDetected, LoopDetector


class TestExactRepeat:
    """Detect when the same (name, detail) pair repeats across the window."""

    def test_exact_repeat(self) -> None:
        det = LoopDetector(window=8)
        for i in range(8):
            det.observe("read_file", "path=/tmp/a.txt", i)
        result = det.detect()
        assert isinstance(result, LoopDetected)
        assert result.pattern == "exact_repeat"
        assert "read_file" in result.evidence

    def test_exact_repeat_below_minimum(self) -> None:
        """Fewer than 4 identical entries should not trigger."""
        det = LoopDetector(window=8)
        for i in range(3):
            det.observe("read_file", "path=/tmp/a.txt", i)
        assert det.detect() is None


class TestCycle2:
    """Detect alternating A-B-A-B pattern."""

    def test_cycle_2(self) -> None:
        det = LoopDetector(window=8)
        pairs = [
            ("read_file", "path=/tmp/a.txt"),
            ("edit_file", "path=/tmp/a.txt"),
        ]
        for i in range(8):
            name, detail = pairs[i % 2]
            det.observe(name, detail, i)
        result = det.detect()
        assert isinstance(result, LoopDetected)
        assert result.pattern == "cycle_2"


class TestCycle3:
    """Detect repeating A-B-C-A-B-C pattern."""

    def test_cycle_3(self) -> None:
        det = LoopDetector(window=9)
        triples = [
            ("read_file", "path=/tmp/a.txt"),
            ("edit_file", "path=/tmp/a.txt"),
            ("exec", "run tests"),
        ]
        for i in range(9):
            name, detail = triples[i % 3]
            det.observe(name, detail, i)
        result = det.detect()
        assert isinstance(result, LoopDetected)
        assert result.pattern == "cycle_3"


class TestCycle4:
    """Detect repeating A-B-C-D-A-B-C-D pattern."""

    def test_cycle_4(self) -> None:
        det = LoopDetector(window=8)
        quads = [
            ("read_file", "path=/tmp/a.txt"),
            ("edit_file", "path=/tmp/a.txt"),
            ("exec", "run tests"),
            ("read_file", "path=/tmp/b.txt"),
        ]
        for i in range(8):
            name, detail = quads[i % 4]
            det.observe(name, detail, i)
        result = det.detect()
        assert isinstance(result, LoopDetected)
        assert result.pattern == "cycle_4"


class TestNoFalsePositive:
    """Varied tool calls should not trigger detection."""

    def test_no_false_positive_varied_tools(self) -> None:
        det = LoopDetector(window=8)
        varied = [
            ("read_file", "path=/tmp/a.txt"),
            ("edit_file", "path=/tmp/a.txt"),
            ("exec", "run tests"),
            ("read_file", "path=/tmp/b.txt"),
            ("write_file", "path=/tmp/c.py"),
            ("exec", "ruff check"),
            ("edit_file", "path=/tmp/d.py"),
            ("exec", "pytest"),
        ]
        for i, (name, detail) in enumerate(varied):
            det.observe(name, detail, i)
        assert det.detect() is None

    def test_no_false_positive_below_window(self) -> None:
        """Fewer entries than period * 2 should never trigger cycles."""
        det = LoopDetector(window=8)
        det.observe("read_file", "a", 0)
        det.observe("edit_file", "b", 1)
        det.observe("read_file", "a", 2)
        det.observe("edit_file", "b", 3)
        # 4 entries: enough for 2-cycle (period*2=4)
        result = det.detect()
        assert isinstance(result, LoopDetected)
        assert result.pattern == "cycle_2"


class TestReset:
    """reset() clears history."""

    def test_reset(self) -> None:
        det = LoopDetector(window=8)
        for i in range(8):
            det.observe("read_file", "same", i)
        assert det.detect() is not None
        det.reset()
        assert det.detect() is None
