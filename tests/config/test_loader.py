"""Tests for config loader env var resolution."""

import json

from nanobot.config.loader import _resolve_env_vars, load_config, save_config


# ---------------------------------------------------------------------------
# _resolve_env_vars unit tests
# ---------------------------------------------------------------------------


def test_env_var_reference_resolves_to_value(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "sk-secret-123")
    assert _resolve_env_vars("$MY_API_KEY") == "sk-secret-123"


def test_missing_env_var_returns_empty_string(monkeypatch):
    monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
    assert _resolve_env_vars("$NONEXISTENT_VAR") == ""


def test_regular_strings_pass_through():
    assert _resolve_env_vars("plain-string") == "plain-string"
    assert _resolve_env_vars("some_prefix_$VAR") == "some_prefix_$VAR"
    assert _resolve_env_vars("") == ""


def test_non_matching_dollar_patterns_pass_through():
    # $$escaped is not a valid var name after $, so no match
    assert _resolve_env_vars("$$escaped") == "$$escaped"
    # $123 starts with digit, not valid
    assert _resolve_env_vars("$123") == "$123"
    # Trailing chars after var name
    assert _resolve_env_vars("$VAR_NAME suffix") == "$VAR_NAME suffix"
    # ${VAR} syntax not supported
    assert _resolve_env_vars("${VAR_NAME}") == "${VAR_NAME}"


def test_nested_dicts_resolved_recursively(monkeypatch):
    monkeypatch.setenv("NESTED_KEY", "resolved-value")
    data = {
        "level1": {
            "level2": {
                "key": "$NESTED_KEY",
                "plain": "unchanged",
            }
        }
    }
    result = _resolve_env_vars(data)
    assert result["level1"]["level2"]["key"] == "resolved-value"
    assert result["level1"]["level2"]["plain"] == "unchanged"


def test_lists_resolved_recursively(monkeypatch):
    monkeypatch.setenv("ITEM_KEY", "item-value")
    monkeypatch.delenv("MISSING_VAR", raising=False)
    data = {
        "items": ["$ITEM_KEY", "plain", {"nested": "$MISSING_VAR"}],
    }
    result = _resolve_env_vars(data)
    assert result["items"][0] == "item-value"
    assert result["items"][1] == "plain"
    assert result["items"][2]["nested"] == ""


def test_non_string_values_preserved():
    data = {"int": 42, "float": 3.14, "bool": True, "null": None}
    result = _resolve_env_vars(data)
    assert result == data


# ---------------------------------------------------------------------------
# Integration: load_config with env vars
# ---------------------------------------------------------------------------


def test_load_config_resolves_env_var_in_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-456")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"providers": {"anthropic": {"apiKey": "$ANTHROPIC_API_KEY"}}}),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.providers.anthropic.api_key == "sk-ant-test-456"


def test_load_config_with_missing_env_var_gives_empty_string(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"providers": {"openai": {"apiKey": "$OPENAI_API_KEY"}}}),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.providers.openai.api_key == ""


def test_save_config_does_not_resolve_env_vars(tmp_path):
    """save_config should save resolved values as-is (no re-expansion)."""
    from nanobot.config.schema import Config

    config_path = tmp_path / "config.json"
    config = Config()
    config.providers.anthropic.api_key = "sk-direct-value"

    save_config(config, config_path)
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["providers"]["anthropic"]["apiKey"] == "sk-direct-value"


def test_load_config_backward_compatible(tmp_path):
    """Plain string values (no $) still work as before."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"providers": {"anthropic": {"apiKey": "sk-plain-key"}}}),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.providers.anthropic.api_key == "sk-plain-key"
