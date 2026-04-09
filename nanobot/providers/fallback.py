"""FallbackProvider: wraps a primary + backup LLM provider.

When the primary returns any error the request is transparently retried
against the backup provider with a pre-configured fallback model.
Quota-specific errors are logged with extra diagnostics.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.base import _QUOTA_ERROR_MARKERS, LLMProvider, LLMResponse

_QUOTA_MARKERS: tuple[str, ...] = _QUOTA_ERROR_MARKERS


def _is_quota_error(content: str | None) -> bool:
    """Check if error content indicates a quota / rate-limit issue."""
    if not content:
        return False
    lc = content.lower()
    return any(m in lc for m in _QUOTA_MARKERS)


class FallbackProvider(LLMProvider):
    """Try *primary*; on ANY provider error fall back to *backup*."""

    def __init__(
        self,
        primary: LLMProvider,
        backup: LLMProvider,
        fallback_model: str,
    ):
        super().__init__()
        self.primary = primary
        self.backup = backup
        self.fallback_model = fallback_model
        self.generation = primary.generation
        self.used_fallback: bool = False

    @property
    def active_provider_info(self) -> str | None:
        """Return the fallback model name if backup was used this call, else None."""
        return self.fallback_model if self.used_fallback else None

    def get_default_model(self) -> str:
        return self.primary.get_default_model()

    def _should_fallback(self, response: LLMResponse) -> bool:
        """Primary errored — always fall back for availability."""
        return response.finish_reason == "error"

    def _log_fallback(self, content: str | None, stream: bool = False) -> None:
        suffix = " (stream)" if stream else ""
        if _is_quota_error(content):
            logger.warning(
                "Primary provider quota limit hit, falling back to {}{}",
                self.fallback_model,
                suffix,
            )
        else:
            logger.warning(
                "Primary provider error, falling back to {}{}: {}",
                self.fallback_model,
                suffix,
                (content or "unknown error")[:200],
            )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.used_fallback = False
        response = await self.primary.chat(messages, tools=tools, model=model, **kwargs)
        if self._should_fallback(response):
            self._log_fallback(response.content)
            self.used_fallback = True
            return await self.backup.chat(
                messages, tools=tools, model=self.fallback_model, **kwargs
            )
        return response

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.used_fallback = False
        response = await self.primary.chat_stream(
            messages,
            tools=tools,
            model=model,
            on_content_delta=on_content_delta,
            **kwargs,
        )
        if self._should_fallback(response):
            self._log_fallback(response.content, stream=True)
            self.used_fallback = True
            return await self.backup.chat_stream(
                messages,
                tools=tools,
                model=self.fallback_model,
                on_content_delta=on_content_delta,
                **kwargs,
            )
        return response
