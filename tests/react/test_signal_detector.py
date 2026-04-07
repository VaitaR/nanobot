"""Tests for SignalDetector — failure / correction pattern detection."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nanobot.react.signal_detector import (
    DetectedSignal,
    SignalDetector,
    create_signal_detector,
)


# ---------------------------------------------------------------------------
# Unit tests: pattern detection
# ---------------------------------------------------------------------------


class TestDetectCorrection:
    """Detect user corrections."""

    def test_wrong(self):
        detector = SignalDetector()
        signals = detector.detect("That's wrong, try again")
        types = [s.signal_type for s in signals]
        assert "correction" in types

    def test_incorrect(self):
        detector = SignalDetector()
        signals = detector.detect("Your answer is incorrect")
        types = [s.signal_type for s in signals]
        assert "correction" in types

    def test_not_what_i(self):
        detector = SignalDetector()
        signals = detector.detect("That's not what I asked for")
        types = [s.signal_type for s in signals]
        assert "correction" in types

    def test_i_meant(self):
        detector = SignalDetector()
        signals = detector.detect("No, I meant the other file")
        types = [s.signal_type for s in signals]
        assert "correction" in types


class TestDetectFailureReport:
    """Detect user error/failure reports."""

    def test_error(self):
        detector = SignalDetector()
        signals = detector.detect("I got an error when running the script")
        types = [s.signal_type for s in signals]
        assert "failure_report" in types

    def test_broken(self):
        detector = SignalDetector()
        signals = detector.detect("The build is broken")
        types = [s.signal_type for s in signals]
        assert "failure_report" in types

    def test_traceback(self):
        detector = SignalDetector()
        signals = detector.detect("Here's the traceback from the crash")
        types = [s.signal_type for s in signals]
        assert "failure_report" in types

    def test_doesnt_work(self):
        detector = SignalDetector()
        signals = detector.detect("This doesn't work at all")
        types = [s.signal_type for s in signals]
        assert "failure_report" in types


class TestDetectNegativeFeedback:
    """Detect negative feedback signals."""

    def test_bad(self):
        detector = SignalDetector()
        signals = detector.detect("That's a bad approach")
        types = [s.signal_type for s in signals]
        assert "negative_feedback" in types

    def test_wrong_answer(self):
        detector = SignalDetector()
        signals = detector.detect("That's the wrong answer")
        types = [s.signal_type for s in signals]
        assert "negative_feedback" in types

    def test_stop_doing(self):
        detector = SignalDetector()
        signals = detector.detect("Stop doing that")
        types = [s.signal_type for s in signals]
        assert "negative_feedback" in types


class TestNoSignal:
    """Normal messages should not produce signals."""

    def test_normal_request(self):
        detector = SignalDetector()
        signals = detector.detect("Please read the file at /tmp/config.yaml")
        assert len(signals) == 0

    def test_greeting(self):
        detector = SignalDetector()
        signals = detector.detect("Hello, how are you?")
        assert len(signals) == 0

    def test_empty_message(self):
        detector = SignalDetector()
        signals = detector.detect("")
        assert len(signals) == 0

    def test_none_message(self):
        detector = SignalDetector()
        signals = detector.detect(None)
        assert len(signals) == 0


class TestMultipleSignals:
    """A single message can trigger multiple signal types."""

    def test_correction_plus_failure(self):
        detector = SignalDetector()
        signals = detector.detect("That's wrong — I got an error running it")
        types = [s.signal_type for s in signals]
        assert "correction" in types
        assert "failure_report" in types


class TestDetectedSignalFields:
    """Verify structure of DetectedSignal."""

    def test_fields_populated(self):
        detector = SignalDetector()
        signals = detector.detect("That's wrong")
        # "That's wrong" matches both "correction" and "negative_feedback" patterns
        assert len(signals) >= 1
        sig = signals[0]
        assert isinstance(sig, DetectedSignal)
        assert sig.signal_type == "correction"
        assert sig.matched_text  # non-empty
        assert sig.message_preview  # non-empty

    def test_message_preview_truncated(self):
        detector = SignalDetector()
        long_msg = "wrong " * 100
        signals = detector.detect(long_msg)
        assert len(signals) >= 1
        assert len(signals[0].message_preview) <= 120


class TestEnableDisable:
    """Detector can be enabled/disabled."""

    def test_disabled_returns_empty(self):
        detector = SignalDetector()
        detector.disable()
        signals = detector.detect("That's completely wrong")
        assert len(signals) == 0

    def test_reenable_works(self):
        detector = SignalDetector()
        detector.disable()
        detector.enable()
        signals = detector.detect("That's wrong")
        assert len(signals) >= 1


class TestDetectAndFeed:
    """detect_and_feed calls evolution callback."""

    @pytest.mark.asyncio
    async def test_callback_called_on_signal(self):
        callback = AsyncMock()
        detector = SignalDetector(evolution_callback=callback)
        await detector.detect_and_feed("That's wrong", metadata={"session": "test"})
        # "That's wrong" matches both "correction" and "negative_feedback"
        callback.assert_awaited()
        assert callback.await_count >= 1
        # First callback should be for "correction" signal
        call_args = callback.call_args_list[0][0]
        assert call_args[0] == "correction"  # signal_type
        assert call_args[1] == "That's wrong"  # message
        assert call_args[2]["session"] == "test"  # metadata

    @pytest.mark.asyncio
    async def test_no_callback_still_detects(self):
        detector = SignalDetector(evolution_callback=None)
        signals = await detector.detect_and_feed("That's wrong")
        assert len(signals) >= 1

    @pytest.mark.asyncio
    async def test_no_signal_no_callback(self):
        callback = AsyncMock()
        detector = SignalDetector(evolution_callback=callback)
        await detector.detect_and_feed("Please read the file")
        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_raise(self):
        callback = AsyncMock(side_effect=RuntimeError("evolution pipeline down"))
        detector = SignalDetector(evolution_callback=callback)
        signals = await detector.detect_and_feed("That's wrong")
        # Should not raise — callback error is caught
        assert len(signals) >= 1


class TestCreateSignalDetector:
    """Factory function with graceful degradation."""

    def test_returns_detector(self):
        detector = create_signal_detector()
        assert detector is not None
        assert isinstance(detector, SignalDetector)

    def test_returns_detector_with_callback(self):
        async def _cb(sig, msg, meta):
            pass

        detector = create_signal_detector(evolution_callback=_cb)
        assert detector is not None

    def test_returns_none_on_exception(self):
        import nanobot.react.signal_detector as mod
        orig = mod.SignalDetector
        mod.SignalDetector = None  # type: ignore[assignment]
        try:
            result = create_signal_detector()
            assert result is None
        finally:
            mod.SignalDetector = orig
