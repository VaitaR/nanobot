"""Detect repetitive tool-call patterns indicative of agent loops."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LoopDetected:
    """Evidence that a loop pattern was found."""

    pattern: str  # "exact_repeat", "cycle_2", "cycle_3", "cycle_4"
    evidence: str  # human-readable description


class LoopDetector:
    """Incremental loop detector using a sliding window of (name, detail) tuples.

    Observes tool calls one at a time and can be queried for loop detection
    over the most recent *window* entries.  Supports exact repeats and
    cycles of period 2, 3, and 4.
    """

    def __init__(self, window: int = 8) -> None:
        self._window = window
        self._history: deque[tuple[str, str]] = deque(maxlen=window)

    # -- public API ----------------------------------------------------------

    def observe(self, tool_name: str, detail: str, iteration: int) -> None:
        """Record a single tool call."""
        self._history.append((tool_name, detail))

    def detect(self) -> LoopDetected | None:
        """Check the current window for loop patterns.

        Returns the *first* matching pattern found (exact repeat > cycles
        of decreasing period).
        """
        result = self._check_exact_repeat()
        if result is not None:
            return result
        for period in (2, 3, 4):
            result = self._check_cycle(period)
            if result is not None:
                return result
        return None

    def reset(self) -> None:
        """Clear all observed history."""
        self._history.clear()

    # -- internal helpers ----------------------------------------------------

    def _check_exact_repeat(self) -> LoopDetected | None:
        """All entries in the window are identical."""
        if len(self._history) < 4:
            return None
        first = self._history[0]
        if all(entry == first for entry in self._history):
            name, detail = first
            trunc = detail[:60] + ("..." if len(detail) > 60 else "")
            return LoopDetected(
                pattern="exact_repeat",
                evidence=f"{name}({trunc}) repeated {len(self._history)} times",
            )
        return None

    def _check_cycle(self, period: int) -> LoopDetected | None:
        """Check for a repeating cycle of the given *period*."""
        sigs = list(self._history)
        if len(sigs) < period * 2:
            return None
        template = sigs[:period]
        for i in range(period, len(sigs)):
            if sigs[i] != template[i % period]:
                return None
        names = "->".join(s[0] for s in template)
        return LoopDetected(
            pattern=f"cycle_{period}",
            evidence=f"{period}-cycle: {names}",
        )
