"""Tests for the Guardrail Agent's clean-response passthrough (Req 6.5).

Covers Property 13 (an outbound response containing no PII or secret patterns is
returned byte-for-byte unchanged, with no masking/redaction applied and no
``MASKING_APPLIED`` audit record written) plus unit tests for representative
clean responses.

Scrubbing is a pure function of the response text, so these tests inject an
in-memory audit log and assert on the returned :class:`OutboundResult` and the
(un)written audit records.
"""

from __future__ import annotations

from hypothesis import given, settings

from acdp.agents.guardrail_agent import GuardrailAgent
from acdp.models import AuditAction, AuditRecord, PlatformConfig
from tests.strategies import clean_text


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


# Feature: autonomous-cyber-defense-platform, Property 13: Clean responses pass
# through unchanged — For any outbound response containing no PII or secret
# patterns, the returned output SHALL be identical to the input and no masking
# or redaction operation SHALL be applied.
# Validates: Requirements 6.5
@settings(max_examples=200)
@given(text=clean_text())
def test_clean_response_passes_through_unchanged(text: str) -> None:
    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound(text)

    # Req 6.5: the returned output is identical to the input.
    assert result.response == text

    # Req 6.5: no masking or redaction operation was applied.
    assert result.masked is False
    assert result.summaries == []
    assert result.total_masked == 0

    # Req 6.5: no MASKING_APPLIED audit record was emitted (no audit record at
    # all, since a clean response performs no auditable masking operation).
    assert audit.records == []
    assert all(
        record.action != AuditAction.MASKING_APPLIED for record in audit.records
    )


# --- Unit tests ------------------------------------------------------------


def test_empty_response_passes_through_unchanged() -> None:
    """An empty response is returned unchanged with no masking (Req 6.5)."""
    audit = _RecordingAuditLog()
    result = _agent(audit).scrub_outbound("")

    assert result.response == ""
    assert result.masked is False
    assert audit.records == []


def test_plain_prose_passes_through_unchanged() -> None:
    """Ordinary prose with no PII/secrets is returned verbatim (Req 6.5)."""
    audit = _RecordingAuditLog()
    text = "The scan completed successfully with no findings to report."
    result = _agent(audit).scrub_outbound(text)

    assert result.response == text
    assert result.masked is False
    assert result.summaries == []
    assert audit.records == []
