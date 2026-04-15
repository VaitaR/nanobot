"""Circuit breaker for LLM provider calls.

Tracks provider health and short-circuits requests when a provider
is consistently failing, preventing thundering-herd retries.
"""

import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Rejecting calls (provider unhealthy)
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and calls are rejected."""

    def __init__(self, failures: int, recovery_at: float):
        self.failures = failures
        self.recovery_at = recovery_at
        remaining = max(0, recovery_at - time.time())
        super().__init__(
            f"Circuit breaker OPEN after {failures} failures. "
            f"Retry in {remaining:.0f}s."
        )


class CircuitBreaker:
    """Tracks provider health and short-circuits when unhealthy.

    Args:
        failure_threshold: Consecutive failures before opening circuit.
        recovery_timeout: Seconds before transitioning from OPEN to HALF_OPEN.
        success_threshold: Successes in HALF_OPEN needed to close circuit.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._total_trips = 0

    @property
    def state(self) -> CircuitState:
        """Current state, with automatic OPEN → HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    def pre_call(self) -> None:
        """Check before making an LLM request.

        Raises:
            CircuitOpenError: If the circuit is open.
        """
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError(
                self._failure_count,
                self._last_failure_time + self.recovery_timeout,
            )

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                logger.info("Circuit breaker closing (recovered)")
                self._state = CircuitState.CLOSED
        else:
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                self._total_trips += 1
                logger.warning(
                    "Circuit breaker OPEN (failures=%d, trip #%d)",
                    self._failure_count,
                    self._total_trips,
                )
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
