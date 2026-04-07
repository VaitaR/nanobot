"""SignalDetector: detect failure / correction patterns in user messages.

Scans inbound user text for signals such as explicit corrections, error
reports, or negative feedback.  Results are fed to the *evolution pipeline*
(a callback) when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from loguru import logger

# Patterns that indicate the user is correcting the agent or reporting a failure.
_SIGNAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("correction", re.compile(
        r"\b(wrong|incorrect|mistake|that('?s| is) not|no,?\s*(i said|that|I meant)|"
        r"I didn'?t (say|mean|ask)|not what I|try again|redo|undo)\b",
        re.IGNORECASE,
    )),
    ("failure_report", re.compile(
        r"\b(error|failed|failure|broken|crashed|doesn'?t work|not working|"
        r"bug|issue|problem|exception|traceback|stack.?trace)\b",
        re.IGNORECASE,
    )),
    ("negative_feedback", re.compile(
        r"\b(bad|terrible|awful|useless|unhelpful|wrong answer|incorrect|"
        r"that('?s| is) wrong|stop doing|don'?t do)\b",
        re.IGNORECASE,
    )),
]

# Evolution callback type: receives (signal_type, message_text, metadata).
EvolutionCallback = Callable[[str, str, dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class DetectedSignal:
    """A signal extracted from a user message."""

    signal_type: str  # "correction" | "failure_report" | "negative_feedback"
    matched_text: str
    message_preview: str  # first 120 chars of the original message


@dataclass
class SignalDetector:
    """Scan user messages for failure / correction signals."""

    patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=lambda: list(_SIGNAL_PATTERNS))
    evolution_callback: EvolutionCallback | None = None
    _enabled: bool = field(default=True, init=False)

    # -- public API --------------------------------------------------------

    def detect(self, message: str) -> list[DetectedSignal]:
        """Return all signals found in *message*."""
        if not self._enabled or not message:
            return []
        signals: list[DetectedSignal] = []
        for sig_type, pattern in self.patterns:
            match = pattern.search(message)
            if match:
                signals.append(DetectedSignal(
                    signal_type=sig_type,
                    matched_text=match.group(),
                    message_preview=message[:120],
                ))
        return signals

    async def detect_and_feed(self, message: str, metadata: dict[str, Any] | None = None) -> list[DetectedSignal]:
        """Detect signals and push to evolution pipeline if available."""
        signals = self.detect(message)
        if signals and self.evolution_callback is not None:
            meta = metadata or {}
            for signal in signals:
                try:
                    await self.evolution_callback(signal.signal_type, message, meta)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.debug("Evolution callback error for '{}': {}", signal.signal_type, exc)
        return signals

    def disable(self) -> None:
        """Silently disable detection."""
        self._enabled = False

    def enable(self) -> None:
        """Re-enable detection."""
        self._enabled = True


def create_signal_detector(
    evolution_callback: EvolutionCallback | None = None,
) -> SignalDetector | None:
    """Factory with graceful degradation.

    Returns ``None`` if initialisation fails so callers can safely skip
    signal detection.
    """
    try:
        detector = SignalDetector(evolution_callback=evolution_callback)
        logger.debug("SignalDetector initialised")
        return detector
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("SignalDetector failed to initialise, skipping: {}", exc)
        return None
