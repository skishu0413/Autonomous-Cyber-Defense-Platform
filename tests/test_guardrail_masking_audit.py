"""Tests for the Guardrail Agent's masking audit safety (Req 6.4).

Covers Property 14 (when masking or redaction occurs, the audit record SHALL
contain the count and category of masked items and SHALL NOT contain any raw
masked or redacted value) plus unit tests for the audit record structure.

The core concern is that the :class:`~acdp.models.AuditRecord` written by
:meth:`~acdp.agents.guardrail.GuardrailAgent.scrub_outbound` leaks *nothing*
about the sensitive values it removed — only counts and category labels.
"""

from __future__ import annotations

import json

from hypothesis import given, settings

from acdp.agents.guardrail_agent import GuardrailAgent
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


# Feature: autonomous-cyber-defense-platform, Property 14: Masking audit records
# never leak raw values — For any outbound response in which masking or redaction
# occurs, the audit record SHALL contain the count and category of masked items
# and SHALL NOT contain any raw masked or redacted value.
# Validates: Requirements 6.4
@settings(max_examples=100)
@given(scenario=text_with_pii_and_secrets())
def test_masking_audit_record_never_leaks_raw_values(
    scenario: dict[str, object],
) -> None:
    text: str = scenario["text"]  # type: ignore[assignment]
    sensitive_values: list[str] = scenario["sensitive_values"]  # type: ignore[assignment]

    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound(text)

    # Masking is guaranteed to occur (text_with_pii_and_secrets always injects
    # at least one PII value and one secret value that the detectors will match).
    assert result.masked is True, "Expected masking to occur for this input"

    # Exactly one MASKING_APPLIED audit record must be present (Req 6.4, 12.1).
    masking_records = [
        r for r in audit.records if r.action == AuditAction.MASKING_APPLIED
    ]
    assert len(masking_records) == 1, (
        f"Expected exactly one MASKING_APPLIED record; got {len(masking_records)}"
    )

    record = masking_records[0]
    detail = record.detail

    # Req 6.4: the audit record MUST contain a count of masked items.
    assert "total_masked" in detail, (
        "Audit record detail must contain 'total_masked'"
    )
    assert isinstance(detail["total_masked"], int), (
        "'total_masked' must be an integer"
    )
    assert detail["total_masked"] > 0, (
        "'total_masked' must be > 0 when masking occurred"
    )

    # Req 6.4: the audit record MUST contain the categories of masked items.
    assert "categories" in detail, (
        "Audit record detail must contain 'categories'"
    )
    assert isinstance(detail["categories"], dict), (
        "'categories' must be a dict"
    )
    assert len(detail["categories"]) > 0, (
        "'categories' must be non-empty when masking occurred"
    )

    # Req 6.4: the audit record MUST NOT contain any raw sensitive value.
    # Serialise the entire detail dict to JSON (the form it would be stored/
    # transmitted in) and assert that none of the injected raw values appear in
    # it, even as a substring.
    detail_json = json.dumps(detail)
    for raw_value in sensitive_values:
        assert raw_value not in detail_json, (
            f"Audit record detail leaks raw sensitive value {raw_value!r}: "
            f"found in serialized detail {detail_json!r}"
        )

    # Also check all individual record fields directly (not just the detail
    # dict) — the raw value must not appear in the outcome, target, or actor_id
    # fields either.
    record_json = json.dumps(record.model_dump(mode="json"))
    for raw_value in sensitive_values:
        assert raw_value not in record_json, (
            f"Audit record leaks raw sensitive value {raw_value!r} outside "
            f"the detail dict: found in full record {record_json!r}"
        )


# --- Unit tests ------------------------------------------------------------


def test_masking_audit_record_contains_total_count() -> None:
    """The MASKING_APPLIED record carries the total number of masked items (Req 6.4)."""
    audit = _RecordingAuditLog()
    # Two distinct PII items (email + SSN) — total_masked must equal 2.
    _agent(audit).scrub_outbound(
        "contact alice@example.com ssn 123-45-6789 for info"
    )

    assert len(audit.records) == 1
    record = audit.records[0]
    assert record.action == AuditAction.MASKING_APPLIED
    assert record.detail["total_masked"] == 2


def test_masking_audit_record_contains_category_labels() -> None:
    """The MASKING_APPLIED record carries category labels, not raw values (Req 6.4)."""
    audit = _RecordingAuditLog()
    _agent(audit).scrub_outbound(
        "email bob@corp.io ssn 987-65-4321 key AKIA1234567890ABCDEF"
    )

    assert len(audit.records) == 1
    record = audit.records[0]
    detail = record.detail

    # All three categories are recorded by label only.
    assert "email" in detail["categories"]
    assert "ssn" in detail["categories"]
    assert "api_key" in detail["categories"]

    # No raw values appear anywhere in the detail.
    for raw in ("bob@corp.io", "987-65-4321", "AKIA1234567890ABCDEF"):
        assert raw not in json.dumps(detail)


def test_masking_audit_record_does_not_contain_raw_email() -> None:
    """Raw email address is never stored in the audit record (Req 6.4)."""
    audit = _RecordingAuditLog()
    raw_email = "secret.user@private.org"
    _agent(audit).scrub_outbound(f"user is {raw_email}")

    assert len(audit.records) == 1
    detail_json = json.dumps(audit.records[0].model_dump(mode="json"))
    assert raw_email not in detail_json


def test_masking_audit_record_does_not_contain_raw_api_key() -> None:
    """Raw API key is never stored in the audit record (Req 6.4)."""
    audit = _RecordingAuditLog()
    raw_key = "sk-test_ABCDEFGHIJKLMNOPabcdefgh"
    _agent(audit).scrub_outbound(f"use {raw_key} for auth")

    assert len(audit.records) == 1
    detail_json = json.dumps(audit.records[0].model_dump(mode="json"))
    assert raw_key not in detail_json


def test_masking_audit_record_does_not_contain_raw_ssn() -> None:
    """Raw SSN is never stored in the audit record (Req 6.4)."""
    audit = _RecordingAuditLog()
    raw_ssn = "012-34-5678"
    _agent(audit).scrub_outbound(f"ssn on file: {raw_ssn}")

    assert len(audit.records) == 1
    detail_json = json.dumps(audit.records[0].model_dump(mode="json"))
    assert raw_ssn not in detail_json
