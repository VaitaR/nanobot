"""Review policy that decides whether to auto-continue or escalate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nanobot.checkpoint.loop_detector import LoopDetected, LoopDetector
from nanobot.checkpoint.snapshot import CheckpointSnapshot


class CheckpointAction(StrEnum):
    """Actions the review policy can recommend."""

    AUTO_CONTINUE = "auto_continue"
    ESCALATE = "escalate"


@dataclass(slots=True)
class ReviewDecision:
    """Result of evaluating a checkpoint snapshot."""

    action: CheckpointAction
    reason: str
    confidence: float  # 0.0–1.0


class ReviewPolicy:
    """Evaluate a :class:`CheckpointSnapshot` and decide on an action.

    Phase 1 behaviour only: **PROGRESS** → ``AUTO_CONTINUE``, **LOOP** /
    **STUCK** → ``ESCALATE``.  Escalation never grants additional budget.
    """

    def __init__(
        self,
        checkpoint_threshold_pct: float = 0.80,
        read_only_hint: bool = False,
        escalation_cooldown: int = 5,
        review_timeout: int = 120,
        loop_window: int = 8,
        max_checkpoints: int = 3,
    ) -> None:
        self.checkpoint_threshold_pct = checkpoint_threshold_pct
        self.read_only_hint = read_only_hint
        self.escalation_cooldown = escalation_cooldown
        self.review_timeout = review_timeout
        self.loop_window = loop_window
        self.max_checkpoints = max_checkpoints

    def compute_threshold(self, max_iterations: int) -> int:
        """Return the iteration number at which the first checkpoint fires."""
        return max(1, int(max_iterations * self.checkpoint_threshold_pct))

    # -- public API ----------------------------------------------------------

    def evaluate(self, snapshot: CheckpointSnapshot) -> ReviewDecision:
        """Run all checks and return a decision."""
        # Priority 1: loop detection (always blocks auto-continue)
        loop = self._check_loop(snapshot)
        if loop is not None:
            return ReviewDecision(
                action=CheckpointAction.ESCALATE,
                reason=f"Loop detected: {loop.evidence}",
                confidence=0.95,
            )

        # Priority 2: stuck detection
        stuck = self._check_stuck(snapshot)
        if stuck:
            reason_parts = ", ".join(stuck)
            return ReviewDecision(
                action=CheckpointAction.ESCALATE,
                reason=f"Agent appears stuck: {reason_parts}",
                confidence=0.8,
            )

        # Default: healthy progress
        return ReviewDecision(
            action=CheckpointAction.AUTO_CONTINUE,
            reason="Agent making progress",
            confidence=0.9,
        )

    # -- internal helpers ----------------------------------------------------

    def _check_loop(self, snapshot: CheckpointSnapshot) -> LoopDetected | None:
        """Run the loop detector over the snapshot's tool calls."""
        if not snapshot.tool_calls:
            return None
        detector = LoopDetector(window=8)
        for tc in snapshot.tool_calls:
            detector.observe(tc.tool_name, tc.detail, tc.iteration)
        return detector.detect()

    def _check_stuck(self, snapshot: CheckpointSnapshot) -> list[str]:
        """Return list of stuck signals that fired (empty = not stuck)."""
        signals: list[str] = []

        # Signal A: low tool-call diversity
        recent = snapshot.tool_calls[-8:] if len(snapshot.tool_calls) >= 8 else snapshot.tool_calls
        if len(recent) >= 4:
            unique_pairs = {(tc.tool_name, tc.detail) for tc in recent}
            diversity_ratio = len(unique_pairs) / len(recent)
            if diversity_ratio < 0.3:
                signals.append("low tool-call diversity")

        # Signal B: low LLM output diversity (Jaccard similarity)
        outputs = snapshot.last_llm_outputs
        if len(outputs) >= 3:
            token_sets = [self._tokenize(o) for o in outputs]
            mean_overlap = self._mean_jaccard(token_sets)
            if mean_overlap > 0.85:
                signals.append("repetitive LLM outputs")

        # Signal C: no file changes (skip when read_only_hint)
        if not self.read_only_hint and len(snapshot.tool_calls) >= 6:
            recent_calls = snapshot.tool_calls[-6:]
            if not any(tc.file_path is not None for tc in recent_calls):
                signals.append("no files touched in recent iterations")

        return signals

    # -- tokenization helpers ------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace + punctuation tokenizer."""
        words = text.lower().split()
        return {w.strip(".,;:!?()[]{}\"'") for w in words if len(w.strip(".,;:!?()[]{}\"'")) > 0}

    @staticmethod
    def _mean_jaccard(sets: list[set[str]]) -> float:
        """Mean pairwise Jaccard similarity over consecutive pairs."""
        if len(sets) < 2:
            return 0.0
        scores: list[float] = []
        for i in range(len(sets) - 1):
            a, b = sets[i], sets[i + 1]
            if not a and not b:
                continue
            intersection = len(a & b)
            union = len(a | b)
            scores.append(intersection / union if union else 0.0)
        return sum(scores) / len(scores) if scores else 0.0
