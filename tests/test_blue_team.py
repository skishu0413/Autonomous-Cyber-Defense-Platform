"""Tests for the Blue Team Agent (Tasks 14.1 and 14.4).

Covers:
- parse/serialize round-trip for JSON, syslog, CEF formats
- malformed record rejection with PARSE_ERROR audit record
- anomaly detection finding classification
- containment threshold gating
- approval-hold behavior
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.agents.blue_team_agent import (
    BlueTeamAgent,
    ContainmentRequest,
    TelemetryParseError,
)
from acdp.audit import AuditLog
from acdp.authorization import AuthorizationService, ScopeRegistry
from acdp.models import (
    AuditAction,
    AuditRecord,
    AuthorizationDecision,
    EmbeddedChunk,
    Finding,
    NormalizedEvent,
    PlatformConfig,
    ScoredChunk,
    Severity,
    SourceCategory,
    TargetScope,
)
from acdp.knowledge_base.retrieve import Retriever
from tests.strategies import normalized_events, telemetry_records


# ---------------------------------------------------------------------------
# Test helpers / stubs
# ---------------------------------------------------------------------------


class InMemoryAuditLog:
    """Simple in-memory audit log for tests."""

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
    """Retriever stub returning configurable scored chunks."""

    def __init__(self, chunks: list[ScoredChunk] | None = None) -> None:
        self._chunks = chunks or []

    def retrieve(
        self, query: str, top_k: int | None = None, category: str | None = None
    ) -> list[ScoredChunk]:
        if category is None:
            return list(self._chunks)
        return [c for c in self._chunks if c.chunk.category.value == category]


def _make_scored_chunk(source_id: str, category: SourceCategory, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk=EmbeddedChunk(
            chunk_id=str(uuid.uuid4()),
            source_id=source_id,
            category=category,
            ingested_at=datetime.now(timezone.utc),
            text="playbook content",
            vector=[0.1, 0.2],
        ),
        score=score,
    )


def _make_scope(assets: list[str], active: bool = True) -> TargetScope:
    now = datetime.now(timezone.utc)
    if active:
        expires_at = now + timedelta(hours=1)
    else:
        expires_at = now - timedelta(hours=1)
    return TargetScope(
        scope_id=str(uuid.uuid4()),
        assets=assets,
        created_at=now,
        expires_at=expires_at,
    )


def _make_agent(
    audit_log: InMemoryAuditLog | None = None,
    config: PlatformConfig | None = None,
    retriever: StubRetriever | None = None,
    authz: AuthorizationService | None = None,
) -> BlueTeamAgent:
    if audit_log is None:
        audit_log = InMemoryAuditLog()
    if config is None:
        config = PlatformConfig(severity_threshold=Severity.HIGH, containment_requires_approval=True)
    if retriever is None:
        retriever = StubRetriever()
    if authz is None:
        scope = _make_scope(["host-a", "host-b"])
        authz = AuthorizationService(scopes=[scope], audit=audit_log)
    return BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)


# ---------------------------------------------------------------------------
# Unit tests — parse / serialize
# ---------------------------------------------------------------------------


class TestParseJSON:
    def test_minimal_json(self) -> None:
        agent = _make_agent()
        raw = json.dumps({
            "event_id": "ev-001",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "host": "web-01",
            "actor": "alice",
            "action": "login",
            "severity": "medium",
        })
        event = agent.parse(raw)
        assert event.event_id == "ev-001"
        assert event.source_format == "json"
        assert event.host == "web-01"
        assert event.actor == "alice"
        assert event.action == "login"
        assert event.severity == Severity.MEDIUM

    def test_json_with_extra_attributes(self) -> None:
        agent = _make_agent()
        raw = json.dumps({
            "event_id": "ev-002",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "port": "443",
            "protocol": "https",
        })
        event = agent.parse(raw)
        assert event.attributes.get("port") == "443"
        assert event.attributes.get("protocol") == "https"

    def test_json_attributes_are_sorted(self) -> None:
        agent = _make_agent()
        raw = json.dumps({
            "event_id": "ev-003",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "z_field": "z",
            "a_field": "a",
            "m_field": "m",
        })
        event = agent.parse(raw)
        keys = list(event.attributes.keys())
        assert keys == sorted(keys)

    def test_invalid_json_raises_and_audits(self) -> None:
        audit_log = InMemoryAuditLog()
        agent = _make_agent(audit_log=audit_log)
        with pytest.raises(TelemetryParseError):
            agent.parse("{not valid json")
        records = audit_log.read_all()
        assert any(r.action == AuditAction.PARSE_ERROR for r in records)

    def test_json_array_root_raises_and_audits(self) -> None:
        audit_log = InMemoryAuditLog()
        agent = _make_agent(audit_log=audit_log)
        with pytest.raises(TelemetryParseError):
            agent.parse("[1, 2, 3]")
        records = audit_log.read_all()
        assert any(r.action == AuditAction.PARSE_ERROR for r in records)

    def test_unknown_severity_defaults_to_info(self) -> None:
        agent = _make_agent()
        raw = json.dumps({
            "event_id": "ev-004",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "severity": "bogus",
        })
        event = agent.parse(raw)
        assert event.severity == Severity.INFO


class TestParseSyslog:
    def test_rfc3164_minimal(self) -> None:
        agent = _make_agent()
        raw = "Jan 15 10:00:00 web-01 sshd[1234]: Failed password for alice from 1.2.3.4"
        event = agent.parse(raw)
        assert event.source_format == "syslog"
        assert event.host == "web-01"
        assert event.actor == "sshd"
        assert "Failed password" in event.attributes.get("msg", "")

    def test_rfc5424_minimal(self) -> None:
        agent = _make_agent()
        raw = "<34>1 2024-01-15T10:00:00+00:00 web-01 sshd 1234 msgid1 - Failed password"
        event = agent.parse(raw)
        assert event.source_format == "syslog"
        assert event.host == "web-01"

    def test_malformed_syslog_raises_and_audits(self) -> None:
        audit_log = InMemoryAuditLog()
        agent = _make_agent(audit_log=audit_log)
        with pytest.raises(TelemetryParseError):
            agent.parse("this is just garbage text with no syslog format at all !!!@@@")
        records = audit_log.read_all()
        assert any(r.action == AuditAction.PARSE_ERROR for r in records)


class TestParseCEF:
    def test_basic_cef(self) -> None:
        agent = _make_agent()
        raw = "CEF:0|ArcSight|Logger|1.0|100|Login Failure|5|src=1.2.3.4 suser=alice dhost=web-01"
        event = agent.parse(raw)
        assert event.source_format == "cef"
        assert event.host == "web-01"
        assert event.actor == "alice"
        assert event.severity == Severity.MEDIUM

    def test_cef_high_severity(self) -> None:
        agent = _make_agent()
        raw = "CEF:0|Vendor|Product|1.0|200|Port Scan|8|src=10.0.0.1 dhost=target-host"
        event = agent.parse(raw)
        assert event.severity == Severity.HIGH

    def test_cef_invalid_header_raises_and_audits(self) -> None:
        audit_log = InMemoryAuditLog()
        agent = _make_agent(audit_log=audit_log)
        with pytest.raises(TelemetryParseError):
            agent.parse("CEF:NOTVALID")
        records = audit_log.read_all()
        assert any(r.action == AuditAction.PARSE_ERROR for r in records)


class TestParseAuditRecord:
    def test_parse_error_audit_names_the_record(self) -> None:
        """Req 7.2: malformed record is named in the audit record."""
        audit_log = InMemoryAuditLog()
        agent = _make_agent(audit_log=audit_log)
        bad_raw = "{bad json"
        with pytest.raises(TelemetryParseError):
            agent.parse(bad_raw)
        records = audit_log.read_all()
        parse_errors = [r for r in records if r.action == AuditAction.PARSE_ERROR]
        assert len(parse_errors) == 1
        # The target field names the record
        assert parse_errors[0].target is not None
        assert bad_raw[:20] in (parse_errors[0].target or "")


# ---------------------------------------------------------------------------
# Unit tests — serialize
# ---------------------------------------------------------------------------


class TestSerialize:
    def test_json_serialize_roundtrip_fields(self) -> None:
        agent = _make_agent()
        raw = json.dumps({
            "event_id": "ev-round",
            "timestamp": "2024-06-01T12:00:00+00:00",
            "host": "db-01",
            "actor": "bob",
            "action": "query",
            "severity": "low",
            "table": "users",
        })
        event = agent.parse(raw)
        serialized = agent.serialize(event)
        event2 = agent.parse(serialized)
        assert event2.event_id == event.event_id
        assert event2.host == event.host
        assert event2.actor == event.actor
        assert event2.action == event.action
        assert event2.severity == event.severity
        assert event2.attributes == event.attributes

    def test_syslog_serialize_produces_parseable_output(self) -> None:
        agent = _make_agent()
        raw = "Jan 15 10:00:00 web-01 sshd[1234]: Test message"
        event = agent.parse(raw)
        serialized = agent.serialize(event)
        # Serialized syslog should be parseable again
        event2 = agent.parse(serialized)
        assert event2.source_format == "syslog"
        assert event2.host == event.host

    def test_cef_serialize_contains_cef_prefix(self) -> None:
        agent = _make_agent()
        raw = "CEF:0|Vendor|Product|1.0|100|Test Event|3|suser=charlie dhost=host-x"
        event = agent.parse(raw)
        serialized = agent.serialize(event)
        assert serialized.startswith("CEF:")


# ---------------------------------------------------------------------------
# Unit tests — analyze (anomaly detection)
# ---------------------------------------------------------------------------


class TestAnalyze:
    def _make_event(self, **kwargs: Any) -> NormalizedEvent:
        defaults = {
            "event_id": str(uuid.uuid4()),
            "source_format": "json",
            "timestamp": datetime.now(timezone.utc),
            "severity": Severity.INFO,
        }
        defaults.update(kwargs)
        return NormalizedEvent(**defaults)

    def test_high_severity_event_yields_finding(self) -> None:
        agent = _make_agent()
        event = self._make_event(severity=Severity.HIGH, host="web-01", action="access")
        findings = agent.analyze(event)
        assert len(findings) >= 1
        assert all(isinstance(f, Finding) for f in findings)

    def test_critical_event_yields_critical_finding(self) -> None:
        agent = _make_agent()
        event = self._make_event(severity=Severity.CRITICAL, host="db-01")
        findings = agent.analyze(event)
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical) >= 1

    def test_port_scan_in_msg_yields_high_finding(self) -> None:
        agent = _make_agent()
        event = self._make_event(
            attributes={"msg": "port scan detected from 1.2.3.4"},
            host="firewall-01",
        )
        findings = agent.analyze(event)
        high_or_critical = [f for f in findings if f.severity in (Severity.HIGH, Severity.CRITICAL)]
        assert len(high_or_critical) >= 1

    def test_privilege_escalation_yields_critical_finding(self) -> None:
        agent = _make_agent()
        event = self._make_event(
            action="sudo",
            attributes={"msg": "sudo privilege escalation attempt"},
            host="server-01",
            actor="mallory",
        )
        findings = agent.analyze(event)
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical) >= 1

    def test_repeated_failed_logins_yields_finding(self) -> None:
        """After 5+ failed logins for same actor, a finding is raised."""
        agent = _make_agent()
        for _ in range(6):
            event = self._make_event(
                action="login_failure",
                actor="brute",
                host="auth-01",
                attributes={"msg": "failed password invalid attempt denied"},
            )
            agent.analyze(event)
        # Analyze one more to get the finding
        event = self._make_event(
            action="login_failure",
            actor="brute",
            host="auth-01",
            attributes={"msg": "failed password invalid attempt denied"},
        )
        findings = agent.analyze(event)
        assert len(findings) >= 1

    def test_suspicious_actor_root(self) -> None:
        agent = _make_agent()
        event = self._make_event(actor="root", host="server-01", action="exec")
        findings = agent.analyze(event)
        assert len(findings) >= 1

    def test_low_severity_event_no_findings_from_severity_rule(self) -> None:
        agent = _make_agent()
        event = self._make_event(severity=Severity.LOW, action="read", host="normal-host")
        # Low severity event should not trigger the high-severity rule
        findings = agent.analyze(event)
        severity_rule_findings = [
            f for f in findings if "High-severity" in f.title
        ]
        assert len(severity_rule_findings) == 0

    def test_findings_have_severity_in_defined_set(self) -> None:
        """Req 8.1 / Property 18: all findings have a valid Severity."""
        agent = _make_agent()
        event = self._make_event(severity=Severity.CRITICAL, host="target")
        findings = agent.analyze(event)
        valid_severities = set(Severity)
        for f in findings:
            assert f.severity in valid_severities

    def test_finding_references_originating_event(self) -> None:
        agent = _make_agent()
        event = self._make_event(severity=Severity.HIGH, host="target-01")
        findings = agent.analyze(event)
        for f in findings:
            assert f.originating_event_id == event.event_id


# ---------------------------------------------------------------------------
# Unit tests — request_containment
# ---------------------------------------------------------------------------


class TestRequestContainment:
    def _make_finding(
        self,
        severity: Severity,
        asset: str | None = "host-a",
    ) -> Finding:
        return Finding(
            finding_id=str(uuid.uuid4()),
            originating_event_id=str(uuid.uuid4()),
            agent_id="blue_team",
            severity=severity,
            title="Test finding",
            detail="Test detail",
            asset=asset,
            created_at=datetime.now(timezone.utc),
        )

    def test_below_threshold_returns_denied(self) -> None:
        """Req 8.2: finding below threshold => denied, no containment."""
        config = PlatformConfig(severity_threshold=Severity.HIGH, containment_requires_approval=False)
        audit_log = InMemoryAuditLog()
        retriever = StubRetriever([_make_scored_chunk("playbook-1", SourceCategory.PLAYBOOK)])
        scope = _make_scope(["host-a"])
        authz = AuthorizationService(scopes=[scope], audit=audit_log)
        agent = BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)

        # MEDIUM does not exceed HIGH threshold
        finding = self._make_finding(Severity.MEDIUM)
        result = agent.request_containment(finding, scope)
        assert result.status == "denied"
        assert "threshold" in (result.denied_reason or "").lower()

    def test_at_threshold_returns_denied(self) -> None:
        """Req 8.2: finding at threshold (not strictly above) => denied."""
        config = PlatformConfig(severity_threshold=Severity.HIGH, containment_requires_approval=False)
        audit_log = InMemoryAuditLog()
        retriever = StubRetriever([_make_scored_chunk("playbook-1", SourceCategory.PLAYBOOK)])
        scope = _make_scope(["host-a"])
        authz = AuthorizationService(scopes=[scope], audit=audit_log)
        agent = BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)

        finding = self._make_finding(Severity.HIGH)
        result = agent.request_containment(finding, scope)
        assert result.status == "denied"

    def test_above_threshold_with_approval_required(self) -> None:
        """Req 8.5: CRITICAL > HIGH threshold + approval required => pending_approval."""
        config = PlatformConfig(severity_threshold=Severity.HIGH, containment_requires_approval=True)
        audit_log = InMemoryAuditLog()
        retriever = StubRetriever([_make_scored_chunk("playbook-1", SourceCategory.PLAYBOOK)])
        scope = _make_scope(["host-a"])
        authz = AuthorizationService(scopes=[scope], audit=audit_log)
        agent = BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)

        finding = self._make_finding(Severity.CRITICAL)
        result = agent.request_containment(finding, scope)
        assert result.status == "pending_approval"

    def test_above_threshold_without_approval(self) -> None:
        """CRITICAL > HIGH threshold + no approval required => approved."""
        config = PlatformConfig(severity_threshold=Severity.HIGH, containment_requires_approval=False)
        audit_log = InMemoryAuditLog()
        retriever = StubRetriever([_make_scored_chunk("playbook-1", SourceCategory.PLAYBOOK)])
        scope = _make_scope(["host-a"])
        authz = AuthorizationService(scopes=[scope], audit=audit_log)
        agent = BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)

        finding = self._make_finding(Severity.CRITICAL)
        result = agent.request_containment(finding, scope)
        assert result.status == "approved"

    def test_authz_denied_out_of_scope(self) -> None:
        """Req 8.3: out-of-scope asset => authz deny."""
        config = PlatformConfig(severity_threshold=Severity.LOW, containment_requires_approval=False)
        audit_log = InMemoryAuditLog()
        retriever = StubRetriever([_make_scored_chunk("playbook-1", SourceCategory.PLAYBOOK)])
        # Scope does NOT include host-z
        scope = _make_scope(["host-a", "host-b"])
        authz = AuthorizationService(scopes=[scope], audit=audit_log)
        agent = BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)

        # Finding targets "host-z" which is not in scope
        finding = self._make_finding(Severity.CRITICAL, asset="host-z")
        result = agent.request_containment(finding, scope)
        assert result.status == "denied"

    def test_playbook_refs_attached(self) -> None:
        """Req 8.4: playbook context is attached to the containment request."""
        config = PlatformConfig(severity_threshold=Severity.HIGH, containment_requires_approval=False)
        audit_log = InMemoryAuditLog()
        retriever = StubRetriever([
            _make_scored_chunk("playbook-alpha", SourceCategory.PLAYBOOK),
            _make_scored_chunk("playbook-beta", SourceCategory.PLAYBOOK),
        ])
        scope = _make_scope(["host-a"])
        authz = AuthorizationService(scopes=[scope], audit=audit_log)
        agent = BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)

        finding = self._make_finding(Severity.CRITICAL, asset="host-a")
        result = agent.request_containment(finding, scope)
        assert len(result.playbook_refs) >= 1
        assert "playbook-alpha" in result.playbook_refs or "playbook-beta" in result.playbook_refs

    def test_expired_scope_denied(self) -> None:
        """Expired scope => authz deny."""
        config = PlatformConfig(severity_threshold=Severity.LOW, containment_requires_approval=False)
        audit_log = InMemoryAuditLog()
        retriever = StubRetriever([_make_scored_chunk("playbook-1", SourceCategory.PLAYBOOK)])
        expired_scope = _make_scope(["host-a"], active=False)
        authz = AuthorizationService(scopes=[expired_scope], audit=audit_log)
        agent = BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)

        finding = self._make_finding(Severity.CRITICAL, asset="host-a")
        result = agent.request_containment(finding, expired_scope)
        assert result.status == "denied"


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: autonomous-cyber-defense-platform, Property 15: Telemetry parse/serialize round-trip
@given(record_and_format=telemetry_records())
@settings(max_examples=100)
def test_telemetry_round_trip_property(
    record_and_format: tuple[str, str],
) -> None:
    """Property 15: parse -> serialize -> parse produces an equivalent NormalizedEvent.

    For any telemetry record in a supported format (JSON, syslog, CEF):
      1. Parse it into a NormalizedEvent (event1).
      2. Serialize event1 back to its canonical representation (raw2).
      3. Parse raw2 into event2.
      4. Assert event2 is equivalent to event1 on all fields that survive the
         round-trip: source_format, host, actor, action, severity, and attributes.

    **Validates: Requirements 7.1, 7.3, 7.4**
    """
    raw, expected_format = record_and_format
    agent = _make_agent()

    event1 = agent.parse(raw)
    assert event1.source_format == expected_format

    raw2 = agent.serialize(event1)
    event2 = agent.parse(raw2)

    # Core identity fields must survive the round-trip
    assert event2.source_format == event1.source_format
    assert event2.host == event1.host
    assert event2.severity == event1.severity

    # Format-specific equivalences
    if expected_format == "json":
        # JSON round-trip preserves all fields exactly
        assert event2.event_id == event1.event_id
        assert event2.actor == event1.actor
        assert event2.action == event1.action
        assert event2.attributes == event1.attributes
    elif expected_format == "syslog":
        # Syslog round-trip preserves host, actor (tag), and msg attribute
        assert event2.actor == event1.actor
        assert event2.attributes.get("msg") == event1.attributes.get("msg")
    elif expected_format == "cef":
        # CEF round-trip preserves action (name field) and severity
        assert event2.action == event1.action


# Feature: autonomous-cyber-defense-platform, Property 15 (NormalizedEvent variant): serialize -> parse round-trip
@given(event=normalized_events())
@settings(max_examples=100)
def test_normalized_event_round_trip_property(event: NormalizedEvent) -> None:
    """Property 15 (NormalizedEvent variant): serialize -> parse produces an equivalent event.

    For any NormalizedEvent in JSON format, serializing it and parsing the result
    must produce an equivalent event — all fields are preserved exactly.

    **Validates: Requirements 7.3, 7.4**
    """
    agent = _make_agent()

    raw = agent.serialize(event)
    event2 = agent.parse(raw)

    assert event2.event_id == event.event_id
    assert event2.source_format == event.source_format
    assert event2.host == event.host
    assert event2.actor == event.actor
    assert event2.action == event.action
    assert event2.severity == event.severity
    assert event2.attributes == event.attributes


# Feature: autonomous-cyber-defense-platform, Property 18: Anomaly matches yield severity-classified findings
@given(
    severity=st.sampled_from([Severity.HIGH, Severity.CRITICAL]),
    host=st.one_of(
        st.none(),
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
            min_size=1, max_size=15
        ),
    ),
)
@settings(max_examples=100)
def test_severity_classified_findings_property(
    severity: Severity,
    host: str | None,
) -> None:
    """Property 18: Anomaly matches yield severity-classified findings.

    For any normalized event carrying HIGH or CRITICAL severity (guaranteed to
    trigger the high-severity escalation rule), BlueTeamAgent.analyze() must
    return at least one Finding whose severity is a valid member of the Severity
    enum, whose originating_event_id links back to the source event, and whose
    agent_id identifies the Blue Team agent.

    **Validates: Requirements 8.1**
    """
    agent = _make_agent()
    event = NormalizedEvent(
        event_id=str(uuid.uuid4()),
        source_format="json",
        timestamp=datetime.now(timezone.utc),
        severity=severity,
        host=host,
    )
    findings = agent.analyze(event)
    valid_severities = set(Severity)
    # High/Critical events must trigger at least one finding
    assert len(findings) >= 1, (
        f"Expected >=1 finding for severity={severity.value}, got 0"
    )
    for f in findings:
        assert f.severity in valid_severities, (
            f"Finding severity {f.severity!r} is not a member of the Severity enum"
        )
        assert f.originating_event_id == event.event_id
        assert f.agent_id == "blue_team"


# Feature: autonomous-cyber-defense-platform, Property 18: Anomaly matches yield severity-classified findings
@given(event=normalized_events())
@settings(max_examples=200)
def test_severity_classified_findings_all_analyzed_events_property(
    event: NormalizedEvent,
) -> None:
    """Property 18 (broad): for ANY normalized event, every finding produced by
    BlueTeamAgent.analyze() has a severity that is a valid member of the Severity
    enum, is linked to the originating event, and is attributed to the Blue Team agent.

    This property holds universally — regardless of whether any rules fire. When
    no anomaly rule matches, the finding list is empty, which trivially satisfies
    the property. When one or more rules match, every finding must carry a
    valid Severity classification.

    **Validates: Requirements 8.1**
    """
    agent = _make_agent()
    findings = agent.analyze(event)
    valid_severities = set(Severity)
    for f in findings:
        assert f.severity in valid_severities, (
            f"Finding severity {f.severity!r} is not a valid Severity enum member"
        )
        assert isinstance(f.finding_id, str) and f.finding_id, (
            "Finding must have a non-empty finding_id"
        )
        assert f.originating_event_id == event.event_id, (
            f"Finding originating_event_id {f.originating_event_id!r} "
            f"does not match event.event_id {event.event_id!r}"
        )
        assert f.agent_id == "blue_team", (
            f"Expected agent_id='blue_team', got {f.agent_id!r}"
        )


# Feature: autonomous-cyber-defense-platform, Property 18: Anomaly matches yield severity-classified findings
@given(
    actor=st.sampled_from(["root", "admin", "anonymous", "guest", "system"]),
    host=st.one_of(
        st.none(),
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
            min_size=1, max_size=15,
        ),
    ),
    action=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789_ ",
        min_size=1, max_size=20,
    ),
)
@settings(max_examples=100)
def test_severity_classified_findings_suspicious_actor_property(
    actor: str,
    host: str | None,
    action: str,
) -> None:
    """Property 18 (suspicious actor): events from known-suspicious actors trigger
    findings with valid Severity classification.

    **Validates: Requirements 8.1**
    """
    agent = _make_agent()
    event = NormalizedEvent(
        event_id=str(uuid.uuid4()),
        source_format="json",
        timestamp=datetime.now(timezone.utc),
        severity=Severity.INFO,
        host=host,
        actor=actor,
        action=action,
    )
    findings = agent.analyze(event)
    valid_severities = set(Severity)
    # Suspicious actors always trigger the suspicious-actor rule
    assert len(findings) >= 1, (
        f"Expected >=1 finding for suspicious actor={actor!r}, got 0"
    )
    for f in findings:
        assert f.severity in valid_severities, (
            f"Finding severity {f.severity!r} is not a member of the Severity enum"
        )
        assert f.originating_event_id == event.event_id
        assert f.agent_id == "blue_team"


# Feature: autonomous-cyber-defense-platform, Property 20: Containment requiring approval is held, not executed
@given(
    finding_severity=st.sampled_from([Severity.CRITICAL]),  # always above HIGH threshold
    requires_approval=st.booleans(),
)
@settings(max_examples=100)
def test_approval_held_property(
    finding_severity: Severity,
    requires_approval: bool,
) -> None:
    """Property 20: When containment requires approval, status is pending_approval.

    **Validates: Requirements 8.5**
    """
    config = PlatformConfig(
        severity_threshold=Severity.HIGH,
        containment_requires_approval=requires_approval
    )
    audit_log = InMemoryAuditLog()
    retriever = StubRetriever([_make_scored_chunk("pb-1", SourceCategory.PLAYBOOK)])
    scope = _make_scope(["host-a"])
    authz = AuthorizationService(scopes=[scope], audit=audit_log)
    agent = BlueTeamAgent(audit_log=audit_log, config=config, retriever=retriever, authz=authz)

    finding = Finding(
        finding_id=str(uuid.uuid4()),
        originating_event_id=str(uuid.uuid4()),
        agent_id="blue_team",
        severity=finding_severity,
        title="Test",
        detail="detail",
        asset="host-a",
        created_at=datetime.now(timezone.utc),
    )
    result = agent.request_containment(finding, scope)

    if requires_approval:
        assert result.status == "pending_approval", (
            f"Expected pending_approval, got {result.status!r}"
        )
    else:
        assert result.status == "approved", (
            f"Expected approved, got {result.status!r}"
        )
