"""Executor resolver — maps executor aliases to provider + model.

Supports two execution modes:
  * **API-based** (glm-*, openrouter): resolved to LLMProvider + model, used via AgentRunner.
  * **CLI-based** (claude-native, claude-zai, codex): routed through ACPX subprocess.

Usage::

    from nanobot.agent.executor import resolve_executor, ExecutorInfo

    info = resolve_executor("glm-5.1", config)
    # info.provider    → LLMProvider instance (for api-based)
    # info.model       → model string
    # info.mode        → "api" | "cli"
    # info.acpx_agent  → "codex" | "claude" | None (None for api-based)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import Config

# ---------------------------------------------------------------------------
# Data class returned by the resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutorInfo:
    """Resolved executor descriptor."""

    model: str
    mode: str  # "api" | "cli"
    provider: Any | None = None  # LLMProvider for api mode, None for cli
    alias: str = ""  # original alias
    acpx_agent: str | None = None  # "codex" | "claude" | None (api-based = None)

    @property
    def is_api(self) -> bool:
        return self.mode == "api"

    @property
    def is_cli(self) -> bool:
        return self.acpx_agent is not None


# ---------------------------------------------------------------------------
# Alias registry — maps alias → (config_provider_key, default_model, mode, acpx_agent)
# ---------------------------------------------------------------------------

_EXECUTOR_REGISTRY: dict[str, tuple[str, str, str, str | None]] = {
    # alias: (config provider key, default model, mode, acpx_agent)
    "glm-turbo": ("custom", "glm-5-turbo", "api", None),
    "glm-5.1": ("custom", "glm-5.1", "api", None),
    "openrouter": ("openrouter", "google/gemma-4-26b-a4b-it:free", "api", None),
    # CLI-based — routed through ACPX subprocess
    "claude-native": ("", "claude-sonnet-4-20250514", "cli", "claude"),
    "claude-zai": ("", "claude-sonnet-4-20250514", "cli", "claude"),
    "codex-5.3": ("", "gpt-5.3-codex", "cli", "codex"),
    "codex-5.4": ("", "gpt-5.4", "cli", "codex"),
    "codex-5.4-mini": ("", "gpt-5.4-mini", "cli", "codex"),
}


def get_known_executors() -> list[str]:
    """Return list of all registered executor aliases."""
    return sorted(_EXECUTOR_REGISTRY.keys())


def resolve_executor(alias: str, config: "Config | None" = None) -> ExecutorInfo:
    """Resolve an executor alias to an ExecutorInfo.

    For **API-based** executors this creates an LLMProvider instance from the
    config.  For **CLI-based** executors it returns a descriptor without a
    provider (the caller is responsible for shelling out).

    Raises:
        ValueError: If *alias* is not recognised.
    """
    entry = _EXECUTOR_REGISTRY.get(alias)
    if entry is None:
        available = ", ".join(get_known_executors())
        raise ValueError(f"Unknown executor '{alias}'. Known executors: {available}")

    provider_key, default_model, mode, acpx_agent = entry

    if mode == "cli":
        logger.info("Executor '{}' is CLI-based (acpx_agent={})", alias, acpx_agent)
        return ExecutorInfo(
            model=default_model,
            mode="cli",
            alias=alias,
            acpx_agent=acpx_agent,
        )

    # --- API-based: build LLMProvider from config ---
    provider = _build_provider_for_alias(alias, provider_key, default_model, config)
    return ExecutorInfo(model=default_model, mode="api", provider=provider, alias=alias)


def _build_provider_for_alias(
    alias: str,
    provider_key: str,
    default_model: str,
    config: "Config | None" = None,
) -> Any:
    """Create an LLMProvider for an API-based executor alias."""
    from nanobot.providers.openai_compat_provider import OpenAICompatProvider
    from nanobot.providers.registry import find_by_name

    api_key: str | None = None
    api_base: str | None = None
    spec = find_by_name(provider_key) if provider_key else None

    if config is not None:
        p_cfg = getattr(config.providers, provider_key, None) if provider_key else None
        if p_cfg is not None:
            api_key = p_cfg.api_key or None
            api_base = p_cfg.api_base or None

    if not api_key:
        logger.warning(
            "Executor '{}': no API key configured for provider '{}'. "
            "The subagent will likely fail at runtime.",
            alias,
            provider_key,
        )

    return OpenAICompatProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=default_model,
        spec=spec,
    )
