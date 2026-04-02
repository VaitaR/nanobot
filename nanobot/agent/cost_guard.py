"""Token budget enforcement and rate limiting for agent runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import CostPolicy


@dataclass(slots=True)
class CostCheckResult:
    """Result of a pre-call budget/rate-limit check."""

    allowed: bool
    reason: str | None = None


@dataclass(slots=True)
class CostGuard:
    """Enforces token budget and rate limits within agent runs.

    All ``0`` / ``0.0`` defaults mean *no limit* — zero-config backward compatible.
    The guard is lightweight: no disk I/O, no async.
    """

    max_tokens_per_turn: int
    max_tokens_per_session: int
    max_cost_usd: float
    rate_limit_rpm: int

    _session_tokens: int = 0
    _session_cost_usd: float = 0.0
    _last_call_tokens: int = 0
    _call_timestamps: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: CostPolicy) -> CostGuard:
        """Build a guard from a :class:`CostPolicy` config object."""
        return cls(
            max_tokens_per_turn=config.max_tokens_per_turn,
            max_tokens_per_session=config.max_tokens_per_session,
            max_cost_usd=config.max_cost_usd,
            rate_limit_rpm=config.rate_limit_rpm,
        )

    @classmethod
    def disabled(cls) -> CostGuard:
        """Return a guard that never blocks (all limits = 0)."""
        return cls(
            max_tokens_per_turn=0,
            max_tokens_per_session=0,
            max_cost_usd=0.0,
            rate_limit_rpm=0,
        )

    # ------------------------------------------------------------------
    # Pre-call check
    # ------------------------------------------------------------------

    def check_before_call(self, estimated_prompt_tokens: int = 0) -> CostCheckResult:
        """Check whether the next LLM call is allowed.

        Call **before** each ``provider.chat_*`` call.  If the guard
        determines the budget is exhausted a
        :class:`CostCheckResult(allowed=False)` is returned with a
        human-readable reason.
        """
        # --- Per-turn token limit ---
        if self.max_tokens_per_turn > 0 and self._session_tokens > 0:
            estimated_total = estimated_prompt_tokens + self._last_call_tokens
            if estimated_total > self.max_tokens_per_turn:
                return CostCheckResult(
                    allowed=False,
                    reason=(
                        f"Per-turn token budget exceeded: estimated "
                        f"{estimated_total} tokens (limit {self.max_tokens_per_turn})."
                    ),
                )

        # --- Per-session token limit ---
        if self.max_tokens_per_session > 0 and self._session_tokens >= self.max_tokens_per_session:
            return CostCheckResult(
                allowed=False,
                reason=(
                    f"Session token budget exceeded: {self._session_tokens} tokens "
                    f"used (limit {self.max_tokens_per_session})."
                ),
            )

        # --- Per-session cost limit ---
        if self.max_cost_usd > 0 and self._session_cost_usd >= self.max_cost_usd:
            return CostCheckResult(
                allowed=False,
                reason=(
                    f"Session cost budget exceeded: ${self._session_cost_usd:.4f} "
                    f"spent (limit ${self.max_cost_usd:.4f})."
                ),
            )

        # --- RPM rate limit (sliding window) ---
        if self.rate_limit_rpm > 0:
            now = time.monotonic()
            window_start = now - 60.0
            self._call_timestamps = [
                t for t in self._call_timestamps if t >= window_start
            ]
            if len(self._call_timestamps) >= self.rate_limit_rpm:
                return CostCheckResult(
                    allowed=False,
                    reason=(
                        f"Rate limit reached: {len(self._call_timestamps)} calls "
                        f"in the last 60 seconds (limit {self.rate_limit_rpm} rpm)."
                    ),
                )

        return CostCheckResult(allowed=True)

    # ------------------------------------------------------------------
    # Post-call recording
    # ------------------------------------------------------------------

    def record_usage(
        self,
        usage: dict[str, int],
        *,
        model: str | None = None,
    ) -> None:
        """Record token usage after each LLM call.

        Parameters
        ----------
        usage:
            Dict with ``prompt_tokens`` and ``completion_tokens`` keys.
        model:
            Optional model identifier used for cost estimation.
        """
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total = prompt_tokens + completion_tokens
        self._session_tokens += total
        self._last_call_tokens = total
        self._call_timestamps.append(time.monotonic())

        if model and self.max_cost_usd > 0:
            cost = _estimate_cost(model, prompt_tokens, completion_tokens)
            self._session_cost_usd += cost
            logger.debug(
                "Cost guard: +{} tokens ({} in / {} out), ${:.6f} this call, "
                "${:.4f} session total (limit ${:.4f})",
                total,
                prompt_tokens,
                completion_tokens,
                cost,
                self._session_cost_usd,
                self.max_cost_usd,
            )
        else:
            logger.debug(
                "Cost guard: +{} tokens ({} in / {} out), {} session total",
                total,
                prompt_tokens,
                completion_tokens,
                self._session_tokens,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_tokens(self) -> int:
        """Cumulative tokens recorded across all turns in this session."""
        return self._session_tokens

    @property
    def session_cost_usd(self) -> float:
        """Estimated cumulative cost in USD for this session."""
        return self._session_cost_usd


# ======================================================================
# Cost estimation helper
# ======================================================================

# (input per million, output per million) — simplified per-model pricing.
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-3-5-20241022": (0.8, 4.0),
    "claude-haiku-3-5": (0.8, 4.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "o1": (15.0, 60.0),
    "o1-mini": (1.1, 4.4),
    "o3-mini": (1.1, 4.4),
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for a single call.

    Uses the model's input + output pricing.  Returns 0.0 if the
    model is not in the price table.
    """
    key = model.lower().split("/")[-1] if "/" in model else model.lower()
    prices = _MODEL_PRICES.get(key)
    if prices is None:
        return 0.0
    input_rate, output_rate = prices
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
