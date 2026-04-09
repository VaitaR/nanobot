"""Tests for unified error classifier."""

from __future__ import annotations

import httpx

from nanobot.providers.error_classifier import (
    ErrorType,
    classify_provider_error,
    get_backoff_seconds,
    is_retryable,
    should_circuit_break,
)


class _FakeError(Exception):
    """Minimal exception with a status_code attribute."""

    def __init__(self, status_code: int, body: str = ""):
        super().__init__(body)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# classify_provider_error — status_code based
# ---------------------------------------------------------------------------


class TestClassifyByStatusCode:
    def test_429_is_transient(self):
        err = _FakeError(429)
        assert classify_provider_error(err) is ErrorType.TRANSIENT

    def test_401_is_fatal(self):
        err = _FakeError(401)
        assert classify_provider_error(err) is ErrorType.FATAL

    def test_403_is_fatal(self):
        err = _FakeError(403)
        assert classify_provider_error(err) is ErrorType.FATAL

    def test_500_is_transient(self):
        err = _FakeError(500)
        assert classify_provider_error(err) is ErrorType.TRANSIENT

    def test_502_is_transient(self):
        err = _FakeError(502)
        assert classify_provider_error(err) is ErrorType.TRANSIENT

    def test_503_is_transient(self):
        err = _FakeError(503)
        assert classify_provider_error(err) is ErrorType.TRANSIENT

    def test_504_is_transient(self):
        err = _FakeError(504)
        assert classify_provider_error(err) is ErrorType.TRANSIENT

    def test_422_is_unknown(self):
        err = _FakeError(422)
        assert classify_provider_error(err) is ErrorType.UNKNOWN

    def test_none_input(self):
        assert classify_provider_error(None) is ErrorType.UNKNOWN

    def test_exception_without_status_code(self):
        err = RuntimeError("something went wrong")
        # No status_code, content comes from str(err)
        assert classify_provider_error(err) is ErrorType.UNKNOWN


# ---------------------------------------------------------------------------
# classify_provider_error — content-string based
# ---------------------------------------------------------------------------


class TestClassifyByContent:
    def test_rate_limit_string(self):
        assert classify_provider_error(None, "rate limit exceeded") is ErrorType.TRANSIENT

    def test_timeout_string(self):
        assert classify_provider_error(None, "request timed out") is ErrorType.TRANSIENT

    def test_overloaded_string(self):
        assert classify_provider_error(None, "server is overloaded") is ErrorType.TRANSIENT

    def test_server_error_string(self):
        assert classify_provider_error(None, "internal server error") is ErrorType.TRANSIENT

    def test_temporarily_unavailable_string(self):
        assert (
            classify_provider_error(None, "service temporarily unavailable") is ErrorType.TRANSIENT
        )

    def test_authentication_string(self):
        assert classify_provider_error(None, "authentication failed") is ErrorType.FATAL

    def test_invalid_api_key_string(self):
        assert classify_provider_error(None, "invalid api_key provided") is ErrorType.FATAL

    def test_account_deactivated_string(self):
        assert classify_provider_error(None, "account deactivated") is ErrorType.FATAL

    def test_quota_string_is_transient(self):
        assert classify_provider_error(None, "quota exceeded for today") is ErrorType.TRANSIENT

    def test_1308_string_is_transient(self):
        assert classify_provider_error(None, "error code 1308") is ErrorType.TRANSIENT

    def test_usage_limit_string_is_transient(self):
        assert classify_provider_error(None, "usage limit reached") is ErrorType.TRANSIENT

    def test_garbage_string(self):
        assert classify_provider_error(None, "xyzzy") is ErrorType.UNKNOWN

    def test_empty_string(self):
        assert classify_provider_error(None, "") is ErrorType.UNKNOWN

    def test_none_content_no_error(self):
        assert classify_provider_error(None, None) is ErrorType.UNKNOWN

    def test_error_with_both_status_and_content(self):
        """status_code takes precedence over content."""
        # status 401 is FATAL even if content mentions rate limit
        err = _FakeError(401, "rate limit something")
        assert classify_provider_error(err, "rate limit something") is ErrorType.FATAL


# ---------------------------------------------------------------------------
# is_retryable
# ---------------------------------------------------------------------------


class TestIsRetryable:
    def test_transient_is_retryable(self):
        assert is_retryable(None, "rate limit exceeded") is True

    def test_fatal_is_not_retryable(self):
        assert is_retryable(None, "authentication failed") is False

    def test_unknown_is_not_retryable(self):
        """UNKNOWN errors are not retried (fail-closed), matching original behaviour."""
        assert is_retryable(None, "some weird error") is False


# ---------------------------------------------------------------------------
# get_backoff_seconds
# ---------------------------------------------------------------------------


class TestGetBackoffSeconds:
    def test_attempt_0_in_range(self):
        for _ in range(50):
            val = get_backoff_seconds(0)
            assert 1.0 <= val <= 2.0

    def test_attempt_1_in_range(self):
        for _ in range(50):
            val = get_backoff_seconds(1)
            assert 2.0 <= val <= 4.0

    def test_attempt_2_in_range(self):
        for _ in range(50):
            val = get_backoff_seconds(2)
            assert 4.0 <= val <= 8.0

    def test_capped_at_max_delay(self):
        for _ in range(50):
            val = get_backoff_seconds(10, max_delay=4.0)
            assert 2.0 <= val <= 4.0


# ---------------------------------------------------------------------------
# should_circuit_break
# ---------------------------------------------------------------------------


class TestShouldCircuitBreak:
    def test_always_false(self):
        assert should_circuit_break(None) is False
        assert should_circuit_break(RuntimeError("boom")) is False
        assert should_circuit_break(None, "authentication failed") is False


# ---------------------------------------------------------------------------
# Integration: httpx-style HTTPStatusError
# ---------------------------------------------------------------------------


class TestHttpxIntegration:
    def test_httpx_status_error(self):
        """httpx.HTTPStatusError has status_code."""
        request = httpx.Request("POST", "https://api.example.com/chat")
        response = httpx.Response(429, request=request)
        err = httpx.HTTPStatusError("Rate limited", request=request, response=response)
        assert classify_provider_error(err) is ErrorType.TRANSIENT
        assert is_retryable(err) is True

    def test_httpx_401_error(self):
        request = httpx.Request("POST", "https://api.example.com/chat")
        response = httpx.Response(401, request=request)
        err = httpx.HTTPStatusError("Unauthorized", request=request, response=response)
        assert classify_provider_error(err) is ErrorType.FATAL
        assert is_retryable(err) is False
