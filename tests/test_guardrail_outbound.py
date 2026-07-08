"""Tests for the Guardrail Agent's outbound response scrubbing (Req 6.1-6.3).

Covers Property 12 (outbound scrubbing removes every detected PII value and
secret value, replacing each with a masking or redaction token) plus unit tests
for each detected category.

Scrubbing is a pure function of the response text, so these tests inject an
in-memory audit log and assert on the returned :class:`OutboundResult`.
"""

from __future__ import annotations

from hypothesis import given, settings

from acdp.agents.guardrail_agent import (
    DEFAULT_MASKING_TOKENS,
    GuardrailAgent,
    MaskingCategory,
)
from acdp.models import AuditAction, AuditRecord, PlatformConfig
from tests.strategies import text_with_pii_and_secrets


class _RecordingAuditLog:
    """An in-memory audit log that records every appended record."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> AuditRecord:
        stored = record.model_copy(update={"seq": len(self.records) + 1})
        self.records.append(stored)
        return stored

    def read_all(self) -> list[AuditRecord]:
        return list(self.records)


class _UnusedBlocklist:
    """A Blocklist stub; outbound scrubbing never consults it."""

    def is_available(self) -> bool:  # pragma: no cover - never called
        return True

    def is_empty(self) -> bool:  # pragma: no cover - never called
        return True

    def match(self, prompt: str):  # pragma: no cover - never called
        return []


def _agent(audit: _RecordingAuditLog) -> GuardrailAgent:
    return GuardrailAgent(
        config=PlatformConfig(),
        audit_log=audit,
        blocklist=_UnusedBlocklist(),
        retriever=None,
    )


# All masking/redaction tokens the scrubber can emit. Used to distinguish a
# genuine leaked value from an incidental token substring.
_ALL_TOKENS = tuple(DEFAULT_MASKING_TOKENS.values())


# Feature: autonomous-cyber-defense-platform, Property 12: Outbound scrubbing
# removes every detected sensitive value — For any outbound response, no
# detected PII value and no detected secret value SHALL remain in the returned
# output; each detected item SHALL be replaced by a masking or redaction token.
# Validates: Requirements 6.1, 6.2, 6.3
@settings(max_examples=200)
@given(scenario=text_with_pii_and_secrets())
def test_outbound_scrubbing_removes_every_detected_sensitive_value(
    scenario: dict[str, object],
) -> None:
    text: str = scenario["text"]  # type: ignore[assignment]
    sensitive_values: list[str] = scenario["sensitive_values"]  # type: ignore[assignment]

    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound(text)

    # Req 6.1-6.3: every injected PII and secret value is detected, so scrubbing
    # occurs and the response differs from the raw text.
    assert result.masked is True

    # Req 6.2, 6.3: no detected sensitive value remains anywhere in the returned
    # output — each has been replaced by a masking/redaction token.
    for value in sensitive_values:
        assert value not in result.response

    # Each detected item is replaced by a masking or redaction token, so the
    # scrubbed output contains at least one such token (Req 6.2, 6.3).
    assert any(token in result.response for token in _ALL_TOKENS)


# --- Unit tests ------------------------------------------------------------


def test_email_is_masked() -> None:
    """An email address is replaced by the email masking token (Req 6.2)."""
    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound("contact alice@example.com for info")

    assert "alice@example.com" not in result.response
    assert DEFAULT_MASKING_TOKENS[MaskingCategory.EMAIL] in result.response
    assert result.masked is True


def test_ssn_is_masked() -> None:
    """A US SSN is replaced by the SSN masking token (Req 6.2)."""
    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound("ssn 123-45-6789 on file")

    assert "123-45-6789" not in result.response
    assert DEFAULT_MASKING_TOKENS[MaskingCategory.SSN] in result.response


def test_financial_account_is_masked() -> None:
    """A financial account number is replaced by its masking token (Req 6.2)."""
    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound("account 123456789 balance")

    assert "123456789" not in result.response
    assert DEFAULT_MASKING_TOKENS[MaskingCategory.FINANCIAL_ACCOUNT] in result.response


def test_api_key_is_redacted() -> None:
    """A secret API key is replaced by the redaction token (Req 6.3)."""
    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound(
        "use sk-live_ABCDEFGHIJKLMNOP0123 as the key"
    )

    assert "sk-live_ABCDEFGHIJKLMNOP0123" not in result.response
    assert DEFAULT_MASKING_TOKENS[MaskingCategory.API_KEY] in result.response


def test_mixed_pii_and_secret_all_removed_and_audited() -> None:
    """PII and a secret in one response are all removed; masking is audited (Req 6.1-6.4)."""
    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound(
        "email bob@corp.io ssn 987-65-4321 key AKIA1234567890ABCDEF"
    )

    for raw in ("bob@corp.io", "987-65-4321", "AKIA1234567890ABCDEF"):
        assert raw not in result.response

    assert result.masked is True
    # Exactly one masking audit record; detail carries counts + categories only.
    assert len(audit.records) == 1
    record = audit.records[0]
    assert record.action == AuditAction.MASKING_APPLIED
    assert record.detail["total_masked"] == result.total_masked
