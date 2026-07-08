"""Property 19: Findings above threshold request containment with playbook context.

# Feature: autonomous-cyber-defense-platform, Property 19: Findings above threshold request containment with playbook context

This module implements the property-based test for Requirement 8.2 and 8.4:

- For any finding with severity STRICTLY ABOVE the configured threshold, the
  BlueTeamAgent SHALL request containment (status != "denied") AND any such
  request SHALL carry non-empty playbook_refs retrieved from the RAG Core.
- For any finding with severity AT OR BELOW the configured threshold, the
  BlueTeamAgent SHALL NOT request containment (status == "denied").

**Validates: Requirements 8.2, 8.4**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.agents.blue_team_agent import BlueTeamAgent, ContainmentRequest
from acdp.authorization import AuthorizationService
from acdp.models import (
    ActionRequest,
    AuditAction,
    AuditRecord,
    AuthorizationDecision,
    EmbeddedChunk,
    Finding,
    PlatformConfig,
    ScoredChunk,
    Severity,
    SourceCategory,
    TargetScope,
)

# ---------------------------------------------------------------------------
# Severity ordering — mirrors BlueTeamAgent's internal _SEVERITY_ORDER
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

_ALL_SEVERITIES = list(Severity)


def _severity_exceeds(sev: Severity, threshold: Severity) -> bool:
    """Return True iff sev is strictly above threshold."""
    return _SEVERITY_ORDER[sev] > _SEVERITY_ORDER[threshold]


# ---------------------------------------------------------------------------
# Test helpers / fakes
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


def _make_playbook_chunk(source_id: str) -> ScoredChunk:
    """Build a ScoredChunk tagged as PLAYBOOK for injection into the fake retriever."""
    return ScoredChunk(
        chunk=EmbeddedChunk(
            chunk_id=str(uuid.uuid4()),
            source_id=source_id,
            category=SourceCategory.PLAYBOOK,
            ingested_at=datetime.now(timezone.utc),
            text="Isolate host and revoke sessions per incident response playbook.",
            vector=[0.1, 0.2, 0.3],
        ),
        score=0.95,
    )


class AlwaysGrantAuthService:
    """Fake Authorization Service that always grants, for containment tests.

    Using a fake here rather than a real AuthorizationService so that the
    test result depends only on the threshold/severity relationship (Property 19)
    and is not accidentally affected by scope expiry or asset membership checks.
    """

    def authorize(self, request: ActionRequest, at: datetime) -> AuthorizationDecision:
        return AuthorizationDecision(
            grant=True,
            reason="always-grant (test fake)",
            scope_id="fake-scope",
        )


class PlaybookRetriever:
    """Fake Retriever that returns a configurable list of non-empty ScoredChunks.

    Always returns chunks tagged as PLAYBOOK for any query or category filter,
    ensuring that when containment is triggered, the playbook context is
    non-empty (Req 8.4). The returned chunk list is configurable so we can
    inject a specific set of source_ids for assertion.
    """

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        # Return all injected chunks regardless of category filter so the
        # agent always gets non-empty playbook context when it calls us.
        return list(self._chunks)


def _make_active_scope(asset: str) -> TargetScope:
    """Build a TargetScope that includes the given asset and is currently active."""
    now = datetime.now(timezone.utc)
    return TargetScope(
        scope_id=str(uuid.uuid4()),
        assets=[asset],
        created_at=now,
        expires_at=now + timedelta(hours=2),
    )


def _make_finding(severity: Severity, asset: str = "host-a") -> Finding:
    return Finding(
        finding_id=str(uuid.uuid4()),
        originating_event_id=str(uuid.uuid4()),
        agent_id="blue_team",
        severity=severity,
        title="Test finding for containment",
        detail="Detected anomalous behaviour on the monitored asset.",
        asset=asset,
        created_at=datetime.now(timezone.utc),
    )


def _make_agent(
    threshold: Severity,
    playbook_chunks: list[ScoredChunk],
    requires_approval: bool = False,
) -> tuple[BlueTeamAgent, InMemoryAuditLog, TargetScope]:
    """Build a BlueTeamAgent wired with fakes suitable for Property 19 tests."""
    audit_log = InMemoryAuditLog()
    config = PlatformConfig(
        severity_threshold=threshold,
        containment_requires_approval=requires_approval,
    )
    retriever = PlaybookRetriever(playbook_chunks)
    authz = AlwaysGrantAuthService()
    scope = _make_active_scope("host-a")
    agent = BlueTeamAgent(
        audit_log=audit_log,
        config=config,
        retriever=retriever,
        authz=authz,  # type: ignore[arg-type]
    )
    return agent, audit_log, scope


# ---------------------------------------------------------------------------
# Strategies for Property 19
# ---------------------------------------------------------------------------


@st.composite
def threshold_and_severity_above(
    draw: st.DrawFn,
) -> tuple[Severity, Severity]:
    """Draw a (threshold, finding_severity) pair where finding_severity > threshold.

    Ensures at least one severity level exists above the threshold, so
    CRITICAL (the maximum) cannot be drawn as a threshold.
    """
    # Severities that can serve as a threshold (must have at least one level above)
    valid_thresholds = [s for s in _ALL_SEVERITIES if _SEVERITY_ORDER[s] < _SEVERITY_ORDER[Severity.CRITICAL]]
    threshold = draw(st.sampled_from(valid_thresholds))
    # Finding severity must be strictly above the threshold
    above = [s for s in _ALL_SEVERITIES if _SEVERITY_ORDER[s] > _SEVERITY_ORDER[threshold]]
    finding_severity = draw(st.sampled_from(above))
    return threshold, finding_severity


@st.composite
def threshold_and_severity_at_or_below(
    draw: st.DrawFn,
) -> tuple[Severity, Severity]:
    """Draw a (threshold, finding_severity) pair where finding_severity <= threshold.

    Ensures at least one severity level exists at or below the threshold, so
    INFO (the minimum) cannot be drawn as a threshold unless finding_severity == INFO.
    """
    threshold = draw(st.sampled_from(_ALL_SEVERITIES))
    # Finding severity must be at or below the threshold
    at_or_below = [s for s in _ALL_SEVERITIES if _SEVERITY_ORDER[s] <= _SEVERITY_ORDER[threshold]]
    finding_severity = draw(st.sampled_from(at_or_below))
    return threshold, finding_severity


# ---------------------------------------------------------------------------
# Property 19 — above threshold: containment requested with non-empty playbook context
# ---------------------------------------------------------------------------


# Feature: autonomous-cyber-defense-platform, Property 19: Findings above threshold request containment with playbook context
@given(
    threshold_and_severity=threshold_and_severity_above(),
    requires_approval=st.booleans(),
)
@settings(max_examples=100)
def test_property19_above_threshold_containment_requested_with_playbook(
    threshold_and_severity: tuple[Severity, Severity],
    requires_approval: bool,
) -> None:
    """Property 19 (above-threshold branch): for any finding whose severity strictly
    exceeds the configured threshold, the Blue Team Agent SHALL request a containment
    action (status is NOT 'denied') AND the containment request SHALL carry non-empty
    playbook context retrieved from the RAG Core.

    The test injects:
    - An AlwaysGrantAuthService so the result depends solely on the threshold check.
    - A PlaybookRetriever returning two non-empty PLAYBOOK-tagged ScoredChunks so
      the "non-empty playbook context" constraint (Req 8.4) is always satisfied when
      the agent proceeds to retrieval.

    **Validates: Requirements 8.2, 8.4**
    """
    threshold, finding_severity = threshold_and_severity
    assert _severity_exceeds(finding_severity, threshold), (
        f"Precondition violated: {finding_severity.value!r} must exceed {threshold.value!r}"
    )

    playbook_chunks = [
        _make_playbook_chunk("playbook-src-1"),
        _make_playbook_chunk("playbook-src-2"),
    ]
    agent, _audit_log, scope = _make_agent(
        threshold=threshold,
        playbook_chunks=playbook_chunks,
        requires_approval=requires_approval,
    )
    finding = _make_finding(finding_severity, asset="host-a")

    result = agent.request_containment(finding, scope)

    # Req 8.2: severity exceeds threshold => containment MUST be requested
    assert result.status != "denied", (
        f"Expected containment to be requested (status != 'denied') for "
        f"severity={finding_severity.value!r} exceeding threshold={threshold.value!r}, "
        f"but got status={result.status!r} with reason={result.denied_reason!r}"
    )
    assert result.status in ("approved", "pending_approval"), (
        f"Unexpected containment status {result.status!r}; "
        f"expected 'approved' or 'pending_approval'"
    )

    # Req 8.4: the containment request MUST carry non-empty playbook context
    assert len(result.playbook_refs) > 0, (
        f"Expected non-empty playbook_refs for severity={finding_severity.value!r} "
        f"above threshold={threshold.value!r}, but got an empty list"
    )

    # The playbook_refs should reference the injected source identifiers
    injected_source_ids = {"playbook-src-1", "playbook-src-2"}
    assert any(ref in injected_source_ids for ref in result.playbook_refs), (
        f"playbook_refs {result.playbook_refs!r} do not include any of the injected "
        f"playbook source ids {injected_source_ids}"
    )


# ---------------------------------------------------------------------------
# Property 19 — at or below threshold: containment NOT requested
# ---------------------------------------------------------------------------


# Feature: autonomous-cyber-defense-platform, Property 19: Findings above threshold request containment with playbook context
@given(
    threshold_and_severity=threshold_and_severity_at_or_below(),
)
@settings(max_examples=100)
def test_property19_at_or_below_threshold_containment_not_requested(
    threshold_and_severity: tuple[Severity, Severity],
) -> None:
    """Property 19 (at-or-below-threshold branch): for any finding whose severity is
    AT OR BELOW the configured threshold, the Blue Team Agent SHALL NOT request a
    containment action — the returned ContainmentRequest MUST have status='denied'.

    **Validates: Requirements 8.2**
    """
    threshold, finding_severity = threshold_and_severity
    assert not _severity_exceeds(finding_severity, threshold), (
        f"Precondition violated: {finding_severity.value!r} must NOT exceed {threshold.value!r}"
    )

    playbook_chunks = [
        _make_playbook_chunk("playbook-src-1"),
    ]
    agent, _audit_log, scope = _make_agent(
        threshold=threshold,
        playbook_chunks=playbook_chunks,
        requires_approval=False,
    )
    finding = _make_finding(finding_severity, asset="host-a")

    result = agent.request_containment(finding, scope)

    # Req 8.2: severity at/below threshold => containment MUST NOT be requested
    assert result.status == "denied", (
        f"Expected status='denied' for severity={finding_severity.value!r} "
        f"at or below threshold={threshold.value!r}, but got status={result.status!r}"
    )
    assert result.denied_reason is not None, (
        "Expected a non-None denied_reason when containment is blocked by threshold"
    )
    assert "threshold" in (result.denied_reason or "").lower(), (
        f"denied_reason {result.denied_reason!r} does not mention 'threshold'"
    )


# ---------------------------------------------------------------------------
# Property 19 — combined: if-and-only-if biconditional across all severities
# ---------------------------------------------------------------------------


# Feature: autonomous-cyber-defense-platform, Property 19: Findings above threshold request containment with playbook context
@given(
    threshold=st.sampled_from(_ALL_SEVERITIES),
    finding_severity=st.sampled_from(_ALL_SEVERITIES),
)
@settings(max_examples=200)
def test_property19_biconditional_threshold_containment(
    threshold: Severity,
    finding_severity: Severity,
) -> None:
    """Property 19 (biconditional): containment is requested IF AND ONLY IF the
    finding severity strictly exceeds the configured threshold.

    This single test exercises both branches (above and at-or-below) in one
    universal quantification over the full cross-product of (threshold, severity),
    covering every possible pair. For each pair the expected outcome (containment
    requested or not) is computed from the severity ordering and compared to the
    actual ContainmentRequest.status.

    **Validates: Requirements 8.2, 8.4**
    """
    playbook_chunks = [
        _make_playbook_chunk("playbook-biconditional-1"),
        _make_playbook_chunk("playbook-biconditional-2"),
    ]
    agent, _audit_log, scope = _make_agent(
        threshold=threshold,
        playbook_chunks=playbook_chunks,
        requires_approval=False,  # use auto-approve so "requested" == "approved"
    )
    finding = _make_finding(finding_severity, asset="host-a")

    result = agent.request_containment(finding, scope)

    exceeds = _severity_exceeds(finding_severity, threshold)

    if exceeds:
        # Req 8.2: above threshold → containment requested
        assert result.status != "denied", (
            f"severity={finding_severity.value!r} exceeds threshold={threshold.value!r} "
            f"but containment was denied (status={result.status!r}, "
            f"reason={result.denied_reason!r})"
        )
        # Req 8.4: above threshold → playbook context must be non-empty
        assert len(result.playbook_refs) > 0, (
            f"severity={finding_severity.value!r} exceeds threshold={threshold.value!r} "
            f"but playbook_refs is empty — Req 8.4 violated"
        )
    else:
        # Req 8.2: at/below threshold → containment NOT requested
        assert result.status == "denied", (
            f"severity={finding_severity.value!r} does NOT exceed "
            f"threshold={threshold.value!r} but containment was not denied "
            f"(status={result.status!r})"
        )
