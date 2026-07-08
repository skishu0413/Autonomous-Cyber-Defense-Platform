"""Property-based tests for configuration loading and serialization (Req 13)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from hypothesis import given, settings

from acdp.config.loader import ConfigLoader
from acdp.models import PlatformConfig
from tests.strategies import partial_platform_config_dicts, platform_configs


# Feature: autonomous-cyber-defense-platform, Property 16: For any valid
# configuration, loading it, serializing the active configuration, and loading
# again SHALL produce a configuration equivalent to the first load.
# Validates: Requirements 13.1, 13.3, 13.4
@settings(max_examples=200)
@given(config=platform_configs())
def test_config_load_serialize_round_trip(config) -> None:
    loader = ConfigLoader()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # First load: write the generated config's YAML to a file and load it back.
        first_path = tmp_dir / "config_first.yaml"
        first_path.write_text(loader.dump(config), encoding="utf-8")
        first_load = loader.load(first_path)

        # Serialize the active configuration and load it again.
        second_path = tmp_dir / "config_second.yaml"
        second_path.write_text(loader.dump(first_load), encoding="utf-8")
        second_load = loader.load(second_path)

    # The two loads must be equivalent (round-trip property).
    assert second_load == first_load


# The documented default for each optional setting, sourced from the field
# defaults declared on the PlatformConfig model itself (the single source of
# truth for "documented defaults").
_DOCUMENTED_DEFAULTS = {
    name: field.get_default(call_default_factory=True)
    for name, field in PlatformConfig.model_fields.items()
}


# Feature: autonomous-cyber-defense-platform, Property 17: Configuration applies
# documented defaults
# For any valid configuration that omits optional values, the loaded
# configuration SHALL contain the documented default value for each omitted
# setting.
# Validates: Requirements 13.5
@settings(max_examples=200)
@given(partial=partial_platform_config_dicts())
def test_config_applies_documented_defaults(partial) -> None:
    loader = ConfigLoader()

    omitted = set(_DOCUMENTED_DEFAULTS) - set(partial)
    # The partial strategy always omits at least one setting.
    assert omitted

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "partial_config.yaml"
        config_path.write_text(
            yaml.safe_dump(partial, sort_keys=True), encoding="utf-8"
        )
        loaded = loader.load(config_path)

    # Every omitted setting takes its documented default value.
    for name in omitted:
        assert getattr(loaded, name) == _DOCUMENTED_DEFAULTS[name], (
            f"omitted setting {name!r} did not fall back to its documented default"
        )


# --- Unit tests: configuration validation errors (Req 13.2) -----------------
#
# These are example-based (not property-based) tests asserting that missing or
# invalid required configuration values cause the loader to raise a ConfigError
# whose message names the specific offending field, so an operator can see
# exactly which value to fix (Req 13.2).

import pytest

from acdp.config.loader import _describe_validation_error
from acdp.exceptions import ConfigError
from pydantic import BaseModel, ValidationError


def _write_config(tmp_path: Path, text: str) -> Path:
    """Write raw YAML ``text`` to a temp file and return its path."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_blank_required_value_raises_config_error_naming_field(tmp_path) -> None:
    """A required value left blank (null) raises ConfigError naming the field.

    Every PlatformConfig field is a required-typed setting; supplying it with a
    null value (an operator leaving the value empty) is an invalid/missing value
    that must be rejected with the field named (Req 13.2).
    """
    loader = ConfigLoader()
    config_path = _write_config(tmp_path, "reasoning_model:\n")  # YAML null value

    with pytest.raises(ConfigError) as exc_info:
        loader.load(config_path)

    assert "reasoning_model" in str(exc_info.value)


def test_missing_required_value_message_names_field() -> None:
    """The loader's error description names a truly-missing required field.

    PlatformConfig ships a documented default for every setting, so an omitted
    key falls back to its default (Req 13.5) rather than erroring. The loader's
    error-description logic that implements the "missing required value" clause
    of Req 13.2 is exercised directly here against a model with a required
    field, confirming the resulting message names the specific field.
    """

    class _RequiredOnly(BaseModel):
        needed: str

    try:
        _RequiredOnly.model_validate({})
    except ValidationError as exc:
        message = _describe_validation_error(exc)
    else:  # pragma: no cover - validation must fail for an empty mapping
        pytest.fail("expected a validation error for a missing required field")

    assert "missing required value" in message
    assert "needed" in message


def test_invalid_int_value_raises_config_error_naming_field(tmp_path) -> None:
    """A wrong-typed integer setting raises ConfigError naming the field (Req 13.2)."""
    loader = ConfigLoader()
    config_path = _write_config(tmp_path, "top_k: not-a-number\n")

    with pytest.raises(ConfigError) as exc_info:
        loader.load(config_path)

    assert "top_k" in str(exc_info.value)


def test_invalid_float_value_raises_config_error_naming_field(tmp_path) -> None:
    """A wrong-typed similarity_threshold raises ConfigError naming the field (Req 13.2)."""
    loader = ConfigLoader()
    config_path = _write_config(tmp_path, "similarity_threshold: not-a-float\n")

    with pytest.raises(ConfigError) as exc_info:
        loader.load(config_path)

    assert "similarity_threshold" in str(exc_info.value)


def test_invalid_enum_value_raises_config_error_naming_field(tmp_path) -> None:
    """An out-of-range severity enum raises ConfigError naming the field (Req 13.2)."""
    loader = ConfigLoader()
    config_path = _write_config(tmp_path, "severity_threshold: extreme\n")

    with pytest.raises(ConfigError) as exc_info:
        loader.load(config_path)

    assert "severity_threshold" in str(exc_info.value)


def test_invalid_literal_value_raises_config_error_naming_field(tmp_path) -> None:
    """An invalid guardrail_default_action literal raises ConfigError naming the field (Req 13.2)."""
    loader = ConfigLoader()
    config_path = _write_config(tmp_path, "guardrail_default_action: maybe\n")

    with pytest.raises(ConfigError) as exc_info:
        loader.load(config_path)

    assert "guardrail_default_action" in str(exc_info.value)
