"""Unit tests for malformed telemetry parse error handling (Task 14.3).

Validates Requirement 7.2:
  IF a Telemetry record is malformed or in an unsupported format,
  THEN THE Blue_Team_Agent SHALL reject the record and record a parse error
  in the Audit_Log identifying the record.

Each test verifies:
  1. A ValueError (TelemetryParseError) is raised, rejecting the record.
  2. Exactly one PARSE_ERROR audit record is appended to the audit log.
  3. The audit record identifies (names) the malformed record via its ``target``
     field or ``detail`` dict.
"""

from __future__ import annotations

import pytest

from acdp.agents.blue_team_agent import BlueTeamAgent, TelemetryParseError
from acdp.authorization import AuthorizationService
from acdp.models import (
    AuditAction,
    AuditRecord,
    PlatformConfig,
    Severity,
    TargetScope,
)
from acdp.knowledge_base.retrieve import Retriever
from datetime import datetime, timedelta, timezone
import uuid


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class InMemoryAuditLog:
    """Minimal in-memory audit log for test isolation."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []
        self._seq = 0

    def append(self, record: AuditRecord) -> AuditRecord:
        self._seq += 1
        stored = record.model_copy(update={"seq": self._seq})
        self._records.append(stored)
        return stored

    def read_all(self) -> list[AuditRecord]:
        return list(self._records)


class StubRetriever:
    """No-op retriever for tests that don't exercise RAG."""

    def retrieve(self, query: str, top_k: int | None = None, category: str | None = None) -> list:
        return []


def _make_scope(assets: list[str], active: bool = True) -> TargetScope:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=1) if active else now - timedelta(hours=1)
    return TargetScope(
        scope_id=str(uuid.uuid4()),
        assets=assets,
        created_at=now,
        expires_at=expires_at,
    )


def _make_agent(audit_log: InMemoryAuditLog | None = None) -> tuple[BlueTeamAgent, InMemoryAuditLog]:
    """Return a (BlueTeamAgent, audit_log) pair wired together for testing."""
    if audit_log is None:
        audit_log = InMemoryAuditLog()
    config = PlatformConfig(severity_threshold=Severity.HIGH, containment_requires_approval=True)
    retriever = StubRetriever()
    scope = _make_scope(["host-a"])
    authz = AuthorizationService(scopes=[scope], audit=audit_log)
    agent = BlueTeamAgent(
        audit_log=audit_log,
        config=config,
        retriever=retriever,
        authz=authz,
    )
    return agent, audit_log


def _parse_errors(audit_log: InMemoryAuditLog) -> list[AuditRecord]:
    """Return only PARSE_ERROR records from the audit log."""
    return [r for r in audit_log.read_all() if r.action == AuditAction.PARSE_ERROR]


# ---------------------------------------------------------------------------
# 1. Malformed JSON records
# ---------------------------------------------------------------------------


class TestMalformedJSON:
    """Records that start with '{' but are not valid JSON."""

    def test_truncated_json_raises_parse_error(self) -> None:
        """A truncated JSON object is rejected with TelemetryParseError."""
        agent, audit_log = _make_agent()
        bad = "{not valid json"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)

    def test_truncated_json_appends_exactly_one_parse_error_audit_record(self) -> None:
        """Exactly one PARSE_ERROR audit record is appended for a truncated JSON record."""
        agent, audit_log = _make_agent()
        bad = "{not valid json"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1

    def test_truncated_json_audit_record_names_the_record(self) -> None:
        """The PARSE_ERROR audit record's target identifies the malformed record."""
        agent, audit_log = _make_agent()
        bad = "{not valid json"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        error = _parse_errors(audit_log)[0]
        # The target should contain the beginning of the raw record
        assert error.target is not None
        assert bad[:20] in (error.target or "")

    def test_json_array_root_raises_parse_error(self) -> None:
        """A JSON array root (not an object) is rejected."""
        agent, audit_log = _make_agent()
        bad = '[{"event": "test"}]'
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)

    def test_json_array_root_appends_exactly_one_parse_error_audit_record(self) -> None:
        """Exactly one PARSE_ERROR audit record is appended for a JSON array root."""
        agent, audit_log = _make_agent()
        bad = '[{"event": "test"}]'
        # Note: '[' does NOT start with '{', so it falls through to syslog parse,
        # which also fails. Either way a PARSE_ERROR record must be appended.
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1

    def test_empty_json_object_parses_gracefully(self) -> None:
        """An empty JSON object '{}' should not be treated as malformed — it parses
        with defaults (no timestamp → now, no severity → INFO)."""
        agent, audit_log = _make_agent()
        # {} is valid JSON, parse should succeed (no error raised)
        event = agent.parse("{}")
        assert event.source_format == "json"
        assert _parse_errors(audit_log) == []

    def test_json_with_invalid_unicode_escape_raises_parse_error(self) -> None:
        """A JSON string containing an invalid escape sequence is rejected."""
        agent, audit_log = _make_agent()
        bad = '{"host": "\\uXXXX"}'
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1

    def test_json_audit_record_outcome_is_rejected(self) -> None:
        """The PARSE_ERROR audit record carries outcome='rejected'."""
        agent, audit_log = _make_agent()
        bad = "{bad: json}"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        error = _parse_errors(audit_log)[0]
        assert error.outcome == "rejected"


# ---------------------------------------------------------------------------
# 2. Malformed CEF records
# ---------------------------------------------------------------------------


class TestMalformedCEF:
    """Records that start with 'CEF:' but have an invalid header."""

    def test_cef_incomplete_header_raises_parse_error(self) -> None:
        """CEF with fewer than 8 pipe-delimited fields is rejected."""
        agent, audit_log = _make_agent()
        bad = "CEF:NOTVALID"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)

    def test_cef_incomplete_header_appends_exactly_one_parse_error_audit_record(self) -> None:
        """Exactly one PARSE_ERROR audit record is appended for invalid CEF header."""
        agent, audit_log = _make_agent()
        bad = "CEF:NOTVALID"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1

    def test_cef_incomplete_header_audit_record_names_the_record(self) -> None:
        """The PARSE_ERROR target identifies the malformed CEF record."""
        agent, audit_log = _make_agent()
        bad = "CEF:NOTVALID"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        error = _parse_errors(audit_log)[0]
        assert error.target is not None
        assert bad[:20] in (error.target or "")

    def test_cef_missing_severity_field_raises_parse_error(self) -> None:
        """CEF with missing mandatory severity field is rejected."""
        agent, audit_log = _make_agent()
        bad = "CEF:0|Vendor|Product|1.0|100|Login"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1

    def test_cef_audit_record_outcome_is_rejected(self) -> None:
        """The PARSE_ERROR audit record for a bad CEF record carries outcome='rejected'."""
        agent, audit_log = _make_agent()
        bad = "CEF:missing|fields"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        error = _parse_errors(audit_log)[0]
        assert error.outcome == "rejected"

    def test_cef_audit_record_detail_contains_raw_preview(self) -> None:
        """The PARSE_ERROR audit detail dict contains a raw_preview of the record."""
        agent, audit_log = _make_agent()
        bad = "CEF:0|only|two"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        error = _parse_errors(audit_log)[0]
        assert "raw_preview" in error.detail
        assert bad[:50] in error.detail["raw_preview"]


# ---------------------------------------------------------------------------
# 3. Unsupported / unrecognized formats (syslog path)
# ---------------------------------------------------------------------------


class TestUnsupportedFormat:
    """Records that don't match any supported format."""

    def test_random_garbage_raises_parse_error(self) -> None:
        """Completely unrecognized text is rejected."""
        agent, audit_log = _make_agent()
        bad = "this is just garbage text !!!@@@###"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)

    def test_random_garbage_appends_exactly_one_parse_error_audit_record(self) -> None:
        """Exactly one PARSE_ERROR audit record is appended for unrecognized text."""
        agent, audit_log = _make_agent()
        bad = "this is just garbage text !!!@@@###"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1

    def test_random_garbage_audit_record_names_the_record(self) -> None:
        """The PARSE_ERROR target identifies the unrecognized record."""
        agent, audit_log = _make_agent()
        bad = "UNKNOWN_FORMAT: some data here"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        error = _parse_errors(audit_log)[0]
        assert error.target is not None
        assert bad[:20] in (error.target or "")

    def test_binary_like_content_raises_parse_error(self) -> None:
        """Content resembling binary data (null bytes, control chars) is rejected."""
        agent, audit_log = _make_agent()
        bad = "data\x00\x01\x02binary"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1

    def test_empty_string_raises_parse_error(self) -> None:
        """An empty string (after stripping) is unrecognized and rejected."""
        agent, audit_log = _make_agent()
        bad = "   "
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1

    def test_unsupported_format_audit_record_outcome_is_rejected(self) -> None:
        """The PARSE_ERROR audit record for unsupported format carries outcome='rejected'."""
        agent, audit_log = _make_agent()
        bad = "TOTALLY_UNSUPPORTED random text"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad)
        error = _parse_errors(audit_log)[0]
        assert error.outcome == "rejected"


# ---------------------------------------------------------------------------
# 4. General invariants across all malformed inputs
# ---------------------------------------------------------------------------


class TestParseErrorInvariants:
    """Cross-cutting invariants that must hold for all malformed inputs."""

    MALFORMED_RECORDS = [
        "{not valid json",              # truncated JSON
        "CEF:NOTVALID",                 # bad CEF header
        "this is not any known format", # unrecognized
        "   ",                          # empty / whitespace-only
        "CEF:0|only|two",               # incomplete CEF
        "[1, 2, 3]",                    # JSON array, not object
    ]

    @pytest.mark.parametrize("bad_raw", MALFORMED_RECORDS)
    def test_malformed_record_raises_telemetry_parse_error(self, bad_raw: str) -> None:
        """Every malformed record raises TelemetryParseError (a ValueError subclass)."""
        agent, _ = _make_agent()
        with pytest.raises(TelemetryParseError) as exc_info:
            agent.parse(bad_raw)
        # TelemetryParseError must be a ValueError subclass (Req 7.2)
        assert isinstance(exc_info.value, ValueError)

    @pytest.mark.parametrize("bad_raw", MALFORMED_RECORDS)
    def test_malformed_record_appends_exactly_one_parse_error_record(self, bad_raw: str) -> None:
        """Exactly one PARSE_ERROR audit record is appended per malformed parse attempt."""
        agent, audit_log = _make_agent()
        with pytest.raises(TelemetryParseError):
            agent.parse(bad_raw)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1, (
            f"Expected exactly 1 PARSE_ERROR record for {bad_raw!r}, got {len(errors)}"
        )

    @pytest.mark.parametrize("bad_raw", MALFORMED_RECORDS)
    def test_parse_error_audit_record_names_the_record(self, bad_raw: str) -> None:
        """The PARSE_ERROR audit record's target contains the beginning of the raw record."""
        agent, audit_log = _make_agent()
        with pytest.raises(TelemetryParseError):
            agent.parse(bad_raw.strip())
        errors = _parse_errors(audit_log)
        assert len(errors) == 1
        error = errors[0]
        stripped = bad_raw.strip()
        if stripped:
            # Target should name (contain a prefix of) the raw record
            assert error.target is not None, "PARSE_ERROR audit record must have a non-None target"
            assert stripped[:20] in (error.target or ""), (
                f"Expected {stripped[:20]!r} in target {error.target!r}"
            )

    @pytest.mark.parametrize("bad_raw", MALFORMED_RECORDS)
    def test_parse_error_audit_record_has_required_fields(self, bad_raw: str) -> None:
        """The PARSE_ERROR audit record has action=PARSE_ERROR and outcome='rejected'."""
        agent, audit_log = _make_agent()
        with pytest.raises(TelemetryParseError):
            agent.parse(bad_raw)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1
        error = errors[0]
        assert error.action == AuditAction.PARSE_ERROR
        assert error.outcome == "rejected"

    @pytest.mark.parametrize("bad_raw", MALFORMED_RECORDS)
    def test_parse_error_audit_record_detail_has_reason(self, bad_raw: str) -> None:
        """The PARSE_ERROR audit record detail dict contains a 'reason' key."""
        agent, audit_log = _make_agent()
        with pytest.raises(TelemetryParseError):
            agent.parse(bad_raw)
        errors = _parse_errors(audit_log)
        assert len(errors) == 1
        error = errors[0]
        assert "reason" in error.detail, (
            f"PARSE_ERROR detail missing 'reason' key: {error.detail!r}"
        )

    def test_multiple_malformed_records_each_appends_own_parse_error(self) -> None:
        """Each successive malformed parse appends its own independent PARSE_ERROR record."""
        agent, audit_log = _make_agent()
        bad_records = ["{bad1", "{bad2", "not syslog 123"]
        for bad in bad_records:
            with pytest.raises(TelemetryParseError):
                agent.parse(bad)
        errors = _parse_errors(audit_log)
        assert len(errors) == len(bad_records), (
            f"Expected {len(bad_records)} PARSE_ERROR records, got {len(errors)}"
        )

    def test_exception_raw_attribute_matches_input(self) -> None:
        """TelemetryParseError.raw attribute contains the original malformed input."""
        agent, _ = _make_agent()
        bad = "{malformed json input here"
        with pytest.raises(TelemetryParseError) as exc_info:
            agent.parse(bad)
        # The exception should carry the raw input for downstream handling
        assert exc_info.value.raw == bad

    def test_successful_parse_appends_no_parse_error_records(self) -> None:
        """A valid telemetry record appends zero PARSE_ERROR audit records."""
        import json as _json
        agent, audit_log = _make_agent()
        valid = _json.dumps({
            "event_id": "ev-valid",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "host": "web-01",
            "severity": "info",
        })
        agent.parse(valid)
        assert _parse_errors(audit_log) == []
