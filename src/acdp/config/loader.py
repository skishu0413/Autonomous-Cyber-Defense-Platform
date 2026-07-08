"""Configuration loading, validation, and serialization (Layer 1 — Foundation).

The :class:`ConfigLoader` reads platform configuration from a YAML file, applies
documented defaults for omitted settings (via the Pydantic field defaults on
:class:`~acdp.models.PlatformConfig`), validates the result, and raises a
:class:`~acdp.exceptions.ConfigError` naming the specific missing/invalid value
on failure (Req 13.1, 13.2, 13.5).

It also serializes an active configuration back into YAML (Req 13.3) such that
``load(dump(config))`` yields an equivalent configuration (Req 13.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from acdp.exceptions import ConfigError
from acdp.models import PlatformConfig

__all__ = ["ConfigLoader"]


class ConfigLoader:
    """Loads, validates, and serializes :class:`PlatformConfig` instances."""

    def load(self, path: Path | str) -> PlatformConfig:
        """Load YAML from ``path``, apply defaults, and validate.

        Args:
            path: Filesystem path to the YAML configuration file.

        Returns:
            A validated :class:`PlatformConfig`. Settings omitted from the file
            take their documented default values (Req 13.5).

        Raises:
            ConfigError: If the file is missing, unreadable, contains malformed
                YAML, is not a mapping, or fails validation. The error message
                names the specific offending value (Req 13.2).
        """
        path = Path(path)

        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(f"Configuration file not found: {str(path)!r}") from exc
        except OSError as exc:
            raise ConfigError(
                f"Could not read configuration file {str(path)!r}: {exc}"
            ) from exc

        try:
            data: Any = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Malformed YAML in configuration file {str(path)!r}: {exc}"
            ) from exc

        # An empty file is valid: every setting falls back to its default.
        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise ConfigError(
                "Configuration root must be a mapping of settings, got "
                f"{type(data).__name__}"
            )

        return self._validate(data)

    def dump(self, config: PlatformConfig) -> str:
        """Serialize an active configuration back into YAML (Req 13.3).

        The output round-trips: ``load(dump(config))`` is equivalent to
        ``config`` (Req 13.4). Enums and literals are serialized to their JSON
        scalar form so the YAML is human-editable and free of Python tags.
        """
        data = config.model_dump(mode="json")
        # connectors is a proper ConnectorConfig field — model_dump serializes
        # it automatically via Pydantic; no manual overrides needed.
        return yaml.safe_dump(data, sort_keys=True, default_flow_style=False)

    @staticmethod
    def _validate(data: dict[str, Any]) -> PlatformConfig:
        """Validate a raw mapping, translating validation failures to ConfigError."""
        from acdp.connectors.config import ConnectorConfig

        # Extract and pre-validate the connectors section before passing the full
        # dict to PlatformConfig.model_validate so we can emit a descriptive
        # ConfigError that names the offending sub-field (Req 13.2 / 1.3).
        connectors_data = data.pop("connectors", None)

        try:
            config = PlatformConfig.model_validate(data)
        except Exception as exc:  # pydantic.ValidationError and subclasses
            raise ConfigError(_describe_validation_error(exc)) from exc

        # When the connectors: section is absent, the field default_factory
        # already produced an all-defaults ConnectorConfig; nothing to do.
        if connectors_data is None:
            return config

        if not isinstance(connectors_data, dict):
            raise ConfigError(
                "Invalid configuration - invalid value for 'connectors': "
                "expected a mapping"
            )
        try:
            config.connectors = ConnectorConfig.model_validate(connectors_data)
        except Exception as exc:
            raise ConfigError(
                "Invalid configuration in connectors section - "
                + _describe_connectors_error(exc)
            ) from exc

        return config


def _describe_validation_error(exc: Exception) -> str:
    """Build a ConfigError message naming the specific invalid/missing field.

    Uses the structured error list from a Pydantic ``ValidationError`` when
    available so operators see exactly which configuration value is at fault
    (Req 13.2).
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return f"Invalid configuration: {exc}"

    details: list[str] = []
    for err in errors():
        loc = err.get("loc", ())
        field = ".".join(str(part) for part in loc) if loc else "<root>"
        msg = err.get("msg", "invalid value")
        etype = err.get("type", "")
        if etype == "missing":
            details.append(f"missing required value {field!r}")
        else:
            details.append(f"invalid value for {field!r}: {msg}")

    if not details:
        return f"Invalid configuration: {exc}"
    return "Invalid configuration - " + "; ".join(details)


def _describe_connectors_error(exc: Exception) -> str:
    """Build a descriptive error message for ConnectorConfig validation failures."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return f"invalid value: {exc}"

    details: list[str] = []
    for err in errors():
        loc = err.get("loc", ())
        field = ".".join(str(part) for part in loc) if loc else "<root>"
        msg = err.get("msg", "invalid value")
        etype = err.get("type", "")
        if etype == "missing":
            details.append(f"missing required value 'connectors.{field}'")
        else:
            details.append(f"invalid value for 'connectors.{field}': {msg}")

    if not details:
        return f"invalid value: {exc}"
    return "; ".join(details)
