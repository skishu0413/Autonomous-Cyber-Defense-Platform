"""Shared exception types for the Autonomous Cyber Defense Platform.

Errors are treated as first-class, auditable outcomes rather than silent
failures. Each exception names the offending resource where possible so the
failure can be surfaced clearly to the operator.
"""

from __future__ import annotations

__all__ = [
    "ACDPError",
    "ConfigError",
    "AuditWriteError",
    "ModelUnavailableError",
    "LLMTimeoutError",
    "IngestionError",
]


class ACDPError(Exception):
    """Base class for all platform-specific errors."""


class ConfigError(ACDPError):
    """Raised when required configuration is missing or invalid (Req 13.2)."""


class AuditWriteError(ACDPError):
    """Raised when an append to the Audit Log fails (Req 12.4)."""


class ModelUnavailableError(ACDPError):
    """Raised when a requested model is unavailable (Req 4.3, 4.4).

    Never resolved by silently substituting a different model.
    """

    def __init__(self, model_name: str, message: str | None = None) -> None:
        self.model_name = model_name
        super().__init__(message or f"Model unavailable: {model_name!r}")


class LLMTimeoutError(ACDPError):
    """Raised when a reasoning request exceeds the configured timeout (Req 4.5)."""


class IngestionError(ACDPError):
    """Raised when a knowledge source is unreadable or malformed (Req 2.4)."""

    def __init__(self, source_id: str, message: str | None = None) -> None:
        self.source_id = source_id
        super().__init__(message or f"Failed to ingest source: {source_id!r}")
