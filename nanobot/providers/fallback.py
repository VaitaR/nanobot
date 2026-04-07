"""FallbackProvider: wraps a primary + backup LLM provider.

When the primary returns a quota / hard-limit error the request is
transparently retried against the backup provider with a pre-configured
fallback model.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse


# Phrases that indicate a hard quota/limit (not a transient network error).
# These are NOT retried by the base retry loop, so we intercept them here.
_QUOTA_MARKERS: tuple[str, ...] = (
    "quota",
    "usage limit",
    "1308",         # GLM-specific quota error code
    "limit reached",
    "insufficient_quota",
    "exceeded your",
)


def _is_quota_error(content: str | None) -> bool:
    if not content:
        return False
    lc = content.lower()
    return any(m in lc for m in _QUOTA_MARKERS)


class FallbackProvider(LLMProvider):
    """Try *primary*; on quota error fall back to *backup* with *fallback_model*."""

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
        # Inherit generation settings from primary
        self.generation = primary.generation

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def get_default_model(self) -> str:
        return self.primary.get_default_model()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        response = await self.primary.chat(messages, tools=tools, model=model, **kwargs)
        if response.finish_reason == "error" and _is_quota_error(response.content):
            logger.warning(
                "Primary provider quota limit hit, falling back to {}",
                self.fallback_model,
            )
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
        response = await self.primary.chat_stream(
            messages, tools=tools, model=model,
            on_content_delta=on_content_delta, **kwargs,
        )
        if response.finish_reason == "error" and _is_quota_error(response.content):
            logger.warning(
                "Primary provider quota limit hit, falling back to {} (stream)",
                self.fallback_model,
            )
            return await self.backup.chat_stream(
                messages, tools=tools, model=self.fallback_model,
                on_content_delta=on_content_delta, **kwargs,
            )
        return response
