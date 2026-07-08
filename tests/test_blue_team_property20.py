"""Property 20: Containment requiring approval is held, not executed.

# Feature: autonomous-cyber-defense-platform, Property 20: Containment requiring approval is held, not executed

This module implements the property-based test for Requirement 8.5:

  "WHERE the Operator has configured containment to require approval, THE
  Blue_Team_Agent SHALL hold the Containment_Action pending Operator approval
  before execution."

Property 20 full statement:
  "For any finding that triggers containment while approval is required by
  configuration, the containment action SHALL remain in a pending state and
  SHALL NOT be executed until operator approval is granted."

Three complementary property tests are implemented:

1. **approval_required=True, severity > threshold** — the ContainmentRequest
   MUST have status="pending_approval" (held, not executed).
2. **approval_required=False, severity > threshold** — the ContainmentRequest
   MUST have status="approved" (auto-executed, for contrast).
3. **Universal biconditional** — for every (threshold, severity,
   requires_approval) triple, the status is "pending_approval" iff and only iff
   severity exceeds the threshold AND approval is required.

**Validates: Requirements 8.5**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from acdp.agents.blue_team_agent import BlueTeamAgent, ContainmentRequest
from acdp.models import (
    ActionRequest,
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
    """Fake Authorization Service that always grants.

    Using a fake here rather than the real AuthorizationService so the test
    result depends only on the approval-required flag (Property 20) and is not
    affected by scope expiry or asset-membership checks.
    """

    def authorize(self, request: ActionRequest, at: datetime) -> AuthorizationDecision:
        return AuthorizationDecision(
            grant=True,
            reason="always-grant (test fake)",
            scope_id="fake-scope",
        )


class PlaybookRetriever:
    """Fake Retriever that always returns the injected PLAYBOOK-tagged chunks.

    Always returns non-empty context regardless of the query or category
    filter, ensuring the RAG retrieval step (Req 8.4) does not block the
    containment path for the approval-hold property being tested here.
    """

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[ScoredChunk]:
        return list(self._chunks)


def _make_active_scope(asset: str = "host-a") -> TargetScope:
    """Return a TargetScope that includes the given asset and is currently active."""
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
        title="Test finding for approval-hold containment",
        detail="Detected anomalous behaviour requiring containment.",
        asset=asset,
        created_at=datetime.now(timezone.utc),
    )


def _make_agent(
    threshold: Severity,
    requires_approval: bool,
    playbook_chunks: list[ScoredChunk] | None = None,
) -> tuple[BlueTeamAgent, InMemoryAuditLog, TargetScope]:
    """Build a BlueTeamAgent wired with fakes for Property 20 tests."""
    if playbook_chunks is None:
        playbook_chunks = [
            _make_playbook_chunk("playbook-src-1"),
            _make_playbook_chunk("playbook-src-2"),
        ]
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
# Strategies for Property 20
# ---------------------------------------------------------------------------


@st.composite
def threshold_and_severity_above(
    draw: st.DrawFn,
) -> tuple[Severity, Severity]:
    """Draw a (threshold, finding_severity) pair where finding_severity > threshold.

    Ensures there is always at least one severity level above the threshold, so
    CRITICAL (the maximum) is never drawn as the threshold.
    """
    valid_thresholds = [s for s in _ALL_SEVERITIES if _SEVERITY_ORDER[s] < _SEVERITY_ORDER[Severity.CRITICAL]]
    threshold = draw(st.sampled_from(valid_thresholds))
    above = [s for s in _ALL_SEVERITIES if _SEVERITY_ORDER[s] > _SEVERITY_ORDER[threshold]]
    finding_severity = draw(st.sampled_from(above))
    return threshold, finding_severity


# ---------------------------------------------------------------------------
# Property 20 — approval_required=True, severity > threshold:
#   status MUST be "pending_approval" (held, not executed)
# ---------------------------------------------------------------------------


# Feature: autonomous-cyber-defense-platform, Property 20: Containment requiring approval is held, not executed
@given(
    threshold_and_severity=threshold_and_severity_above(),
)
@settings(max_examples=100)
def test_property20_approval_required_holds_containment_pending(
    threshold_and_severity: tuple[Severity, Severity],
) -> None:
    """Property 20 (approval_required=True branch):

    For any finding whose severity STRICTLY EXCEEDS the configured threshold,
    when containment_requires_approval=True, the Blue Team Agent SHALL return a
    ContainmentRequest with status="pending_approval".

    This means the containment action is HELD pending operator approval and has
    NOT been executed.

    **Validates: Requirements 8.5**
    """
    threshold, finding_severity = threshold_and_severity
    assert _severity_exceeds(finding_severity, threshold), (
        f"Precondition violated: {finding_severity.value!r} must exceed {threshold.value!r}"
    )

    agent, _audit_log, scope = _make_agent(
        threshold=threshold,
        requires_approval=True,  # approval IS required
    )
    finding = _make_finding(finding_severity, asset="host-a")

    result = agent.request_containment(finding, scope)

    # The containment must be triggered (not denied by threshold)
    assert result.status != "denied", (
        f"Expected containment to be triggered (not 'denied') when severity "
        f"{finding_severity.value!r} exceeds threshold {threshold.value!r}, "
        f"but got status={result.status!r} with reason={result.denied_reason!r}"
    )

    # Req 8.5: when approval is required, containment MUST be HELD (pending)
    assert result.status == "pending_approval", (
        f"Expected status='pending_approval' when containment_requires_approval=True "
        f"and severity={finding_severity.value!r} exceeds threshold={threshold.value!r}, "
        f"but got status={result.status!r}"
    )

    # Confirm the containment has NOT been auto-executed
    assert result.status != "approved", (
        f"Containment must NOT be auto-approved (i.e., executed) when approval "
        f"is required, but got status='approved' for "
        f"severity={finding_severity.value!r}, threshold={threshold.value!r}"
    )

    # The finding_id on the result must match the input finding
    assert result.finding_id == finding.finding_id, (
        f"ContainmentRequest.finding_id {result.finding_id!r} does not match "
        f"the input finding_id {finding.finding_id!r}"
    )


# ---------------------------------------------------------------------------
# Property 20 — approval_required=False, severity > threshold:
#   status MUST be "approved" (auto-executed, for contrast)
# ---------------------------------------------------------------------------


# Feature: autonomous-cyber-defense-platform, Property 20: Containment requiring approval is held, not executed
@given(
    threshold_and_severity=threshold_and_severity_above(),
)
@settings(max_examples=100)
def test_property20_no_approval_required_executes_containment(
    threshold_and_severity: tuple[Severity, Severity],
) -> None:
    """Property 20 (approval_required=False branch, contrast test):

    For any finding whose severity STRICTLY EXCEEDS the configured threshold,
    when containment_requires_approval=False, the Blue Team Agent SHALL return a
    ContainmentRequest with status="approved" — the action IS executed immediately
    (not held).

    This test is the logical complement of the approval-hold test and validates
    that the "pending_approval" outcome is specific to the approval-required
    configuration, not a general default.

    **Validates: Requirements 8.5**
    """
    threshold, finding_severity = threshold_and_severity
    assert _severity_exceeds(finding_severity, threshold), (
        f"Precondition violated: {finding_severity.value!r} must exceed {threshold.value!r}"
    )

    agent, _audit_log, scope = _make_agent(
        threshold=threshold,
        requires_approval=False,  # approval NOT required
    )
    finding = _make_finding(finding_severity, asset="host-a")

    result = agent.request_containment(finding, scope)

    # Req 8.5 contrast: no approval needed → containment IS auto-executed
    assert result.status == "approved", (
        f"Expected status='approved' when containment_requires_approval=False "
        f"and severity={finding_severity.value!r} exceeds threshold={threshold.value!r}, "
        f"but got status={result.status!r}"
    )

    assert result.status != "pending_approval", (
        f"Containment should NOT be held when approval is not required, "
        f"but got status='pending_approval'"
    )


# ---------------------------------------------------------------------------
# Property 20 — universal biconditional over all (threshold, severity,
#   requires_approval) combinations
# ---------------------------------------------------------------------------


# Feature: autonomous-cyber-defense-platform, Property 20: Containment requiring approval is held, not executed
@given(
    threshold=st.sampled_from(_ALL_SEVERITIES),
    finding_severity=st.sampled_from(_ALL_SEVERITIES),
    requires_approval=st.booleans(),
)
@settings(max_examples=200)
def test_property20_biconditional_approval_held(
    threshold: Severity,
    finding_severity: Severity,
    requires_approval: bool,
) -> None:
    """Property 20 (biconditional over all combinations):

    The containment action status is "pending_approval" IF AND ONLY IF:
      - finding.severity STRICTLY EXCEEDS the configured threshold, AND
      - containment_requires_approval is True.

    In every other case the action is either:
      - "denied" (severity at/below threshold), or
      - "approved" (exceeds threshold but approval not required).

    This property exercises the full cross-product of:
      (threshold ∈ Severity) × (severity ∈ Severity) × (requires_approval ∈ {True, False})

    **Validates: Requirements 8.5**
    """
    agent, _audit_log, scope = _make_agent(
        threshold=threshold,
        requires_approval=requires_approval,
    )
    finding = _make_finding(finding_severity, asset="host-a")

    result = agent.request_containment(finding, scope)

    exceeds = _severity_exceeds(finding_severity, threshold)

    if not exceeds:
        # Below/at threshold: MUST be denied regardless of approval setting
        assert result.status == "denied", (
            f"severity={finding_severity.value!r} does NOT exceed "
            f"threshold={threshold.value!r}: expected status='denied' "
            f"but got {result.status!r}"
        )

    elif requires_approval:
        # Above threshold AND approval required: MUST be pending (held)
        assert result.status == "pending_approval", (
            f"severity={finding_severity.value!r} exceeds threshold={threshold.value!r} "
            f"with requires_approval=True: expected status='pending_approval' "
            f"(held, not executed), but got {result.status!r}"
        )
        # Confirm it was NOT auto-executed
        assert result.status != "approved", (
            f"ContainmentRequest must NOT be auto-executed (status='approved') "
            f"when approval is required"
        )

    else:
        # Above threshold AND approval NOT required: MUST be approved (auto-executed)
        assert result.status == "approved", (
            f"severity={finding_severity.value!r} exceeds threshold={threshold.value!r} "
            f"with requires_approval=False: expected status='approved' "
            f"(auto-executed), but got {result.status!r}"
        )
        # Confirm it was NOT held
        assert result.status != "pending_approval", (
            f"ContainmentRequest must NOT be held when approval is not required"
        )

    # In all cases: finding_id must be preserved
    assert result.finding_id == finding.finding_id, (
        f"ContainmentRequest.finding_id {result.finding_id!r} does not match "
        f"the input finding_id {finding.finding_id!r}"
    )
