"""Shared Pydantic v2 data models for the Autonomous Cyber Defense Platform.

All models use strict types and deterministic serialization so that
parsing/serialization boundaries (telemetry, configuration) obey round-trip
laws that can be verified with property-based tests.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from acdp.connectors.config import ConnectorConfig

__all__ = [
    "SourceCategory",
    "KnowledgeSource",
    "EmbeddedChunk",
    "ScoredChunk",
    "Severity",
    "Finding",
    "TaskState",
    "Task",
    "TaskPlan",
    "TaskResult",
    "TargetScope",
    "SecurityEvent",
    "SecurityEventType",
    "ActionRequest",
    "AuthorizationDecision",
    "AuditAction",
    "AuditRecord",
    "NormalizedEvent",
    "PlatformConfig",
    "LLMRequest",
    "LLMResponse",
    "ConnectorConfig",
]


class SourceCategory(str, Enum):
    """Category of an ingested knowledge source."""

    OWASP_GENAI = "owasp_genai"
    MITRE_ATTACK = "mitre_attack"
    MITRE_ATLAS = "mitre_atlas"
    PLAYBOOK = "playbook"
    COMPLIANCE = "compliance"
    TOPOLOGY = "topology"
    OTHER = "other"


class KnowledgeSource(BaseModel):
    """A raw knowledge source submitted for ingestion into the RAG Core."""

    source_id: str
    category: SourceCategory
    content: str


class EmbeddedChunk(BaseModel):
    """A chunk of a knowledge source stored in the Vector Store."""

    chunk_id: str  # hash(source_id + index + text) -> idempotent
    source_id: str
    category: SourceCategory
    ingested_at: datetime
    text: str
    vector: list[float]


class ScoredChunk(BaseModel):
    """An embedded chunk annotated with a similarity score."""

    chunk: EmbeddedChunk
    score: float  # similarity, higher = closer


class Severity(str, Enum):
    """Severity classification for findings and normalized events."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Finding(BaseModel):
    """A structured record of a detected vulnerability, threat, or anomaly."""

    finding_id: str  # unique, links to originating event (Req 12.3)
    originating_event_id: str
    agent_id: str
    severity: Severity
    title: str
    detail: str
    asset: str | None = None
    context_refs: list[str] = Field(default_factory=list)  # source_ids of retrieved RAG context
    created_at: datetime


class SecurityEventType(str, Enum):
    """Classifies the kind of security event received by the Orchestrator.

    The event type drives agent routing in ``Orchestrator.handle_event``
    (Req 1.1):

    * ``TELEMETRY`` — normalized log/telemetry data → Blue Team Agent.
    * ``PROBE_REQUEST`` — adversary simulation request → Red Team Agent.
    * ``VULNERABILITY_FINDING`` — code/dependency vulnerability → DevSecOps Agent.
    * ``PROMPT`` — inbound user prompt that needs firewall screening → Guardrail Agent.
    * ``UNKNOWN`` — unclassified event; routed to Guardrail as a safe default.
    """

    TELEMETRY = "telemetry"
    PROBE_REQUEST = "probe_request"
    VULNERABILITY_FINDING = "vulnerability_finding"
    PROMPT = "prompt"
    UNKNOWN = "unknown"


class SecurityEvent(BaseModel):
    """A security event received by the Orchestrator (Req 1.1).

    The Orchestrator uses ``event_type`` to route the event to the responsible
    agent(s). ``target_scope_id`` names the operator-defined
    :class:`TargetScope` that the tasks produced for this event must carry.
    ``payload`` carries format-specific data the assigned agent will consume
    (e.g. a raw telemetry string, a prompt, or a repository reference).
    """

    event_id: str
    event_type: SecurityEventType
    target_scope_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class TaskState(str, Enum):
    """Lifecycle state of an orchestrated task."""

    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """A unit of work dispatched to an agent by the Orchestrator."""

    task_id: str
    event_id: str
    assigned_agent: str
    target_scope_id: str
    state: TaskState = TaskState.PENDING
    failure_reason: str | None = None


class TaskPlan(BaseModel):
    """A plan of tasks produced by the Orchestrator for a security event."""

    plan_id: str
    event_id: str
    tasks: list[Task]


class TaskResult(BaseModel):
    """The outcome of dispatching one :class:`Task`.

    ``task`` is the final (mutated) task after dispatch; ``findings`` are the
    :class:`Finding` records returned by the agent for this task (empty on
    failure or when the agent found nothing). ``failure_reason`` mirrors
    ``task.failure_reason`` for convenience.
    """

    task: Task
    findings: list[Finding] = Field(default_factory=list)
    failure_reason: str | None = None


class TargetScope(BaseModel):
    """An operator-defined, expiring set of assets that agents may act on."""

    scope_id: str
    assets: list[str]  # hosts, domains, repos, endpoints
    created_at: datetime
    expires_at: datetime  # required expiration (Req 11.3)
    revoked: bool = False

    def is_active(self, at: datetime) -> bool:
        """Return True iff the scope is neither revoked nor expired at ``at``."""
        return not self.revoked and at < self.expires_at


class ActionRequest(BaseModel):
    """A request by an agent to act on an external asset (Req 11.1, 11.5).

    Every request names the requesting ``agent_id`` and the ``asset`` it wants
    to act on so the Authorization Service can decide, and audit, on the basis
    of both. ``action`` is a free-form label for the kind of side effect
    (e.g. "probe", "containment", "pr") carried through into the audit trail.
    """

    agent_id: str  # the requesting agent
    asset: str  # the target asset (host, domain, repo, endpoint)
    action: str = "act"  # kind of side effect being requested


class AuthorizationDecision(BaseModel):
    """The outcome of an authorization check (Req 11.1, 11.2, 11.4).

    ``grant`` is ``True`` only when the asset was positively matched to a
    currently valid scope; every other case fails closed to ``grant=False``.
    ``scope_id`` names the matching scope on a grant (``None`` on a deny) and
    ``reason`` records why the request was denied for the audit trail.
    """

    grant: bool
    reason: str
    scope_id: str | None = None


class AuditAction(str, Enum):
    """The type of action recorded in an audit record."""

    FINDING_RECORDED = "finding_recorded"
    TASK_FAILED = "task_failed"
    SCOPE_DEFINED = "scope_defined"
    AUTHZ_GRANT = "authz_grant"
    AUTHZ_DENY = "authz_deny"
    GUARDRAIL_DECISION = "guardrail_decision"
    MASKING_APPLIED = "masking_applied"
    PARSE_ERROR = "parse_error"
    PROBE_REFUSED = "probe_refused"
    PR_OPENED = "pr_opened"
    REMEDIATION_DECLINED = "remediation_declined"


class AuditRecord(BaseModel):
    """An append-only audit log entry."""

    seq: int | None = None  # assigned on append, monotonic
    timestamp: datetime
    actor_id: str  # agent or "orchestrator"
    action: AuditAction
    outcome: str
    target: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)  # never contains raw masked values (Req 6.4)


class NormalizedEvent(BaseModel):
    """A telemetry record parsed into a canonical, normalized form."""

    event_id: str
    source_format: str  # e.g. "syslog", "cef", "json"
    timestamp: datetime
    host: str | None = None
    actor: str | None = None
    action: str | None = None
    severity: Severity = Severity.INFO
    attributes: dict[str, str] = Field(default_factory=dict)  # ordered, canonical keys for round-trip


class PlatformConfig(BaseModel):
    """Platform-wide configuration loaded from YAML and validated."""

    reasoning_model: str = "llama3"
    embedding_model: str = "nomic-embed-text"
    llm_timeout_seconds: float = 30.0
    similarity_threshold: float = 0.85  # Guardrail; < 1.0 configurable (Req 5.2)
    top_k: int = 5  # Req 3.1
    severity_threshold: Severity = Severity.HIGH
    containment_requires_approval: bool = True
    guardrail_default_action: Literal["allow", "block"] = "block"  # Req 5.5
    vector_store_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    audit_log_path: str = "./audit.log"
    # Production connectors configuration.  The type annotation uses a string
    # reference to avoid a circular import at module load time (connectors/
    # __init__.py transitively imports acdp.models).  The factory imports
    # ConnectorConfig lazily so the default is always a fully-populated,
    # all-disabled ConnectorConfig instance (Req 1.2, 1.4).
    connectors: "ConnectorConfig" = Field(
        default_factory=lambda: _make_connector_config()
    )


class LLMRequest(BaseModel):
    """A reasoning request routed by the LLM Gateway to a local model.

    ``model`` is optional: when unset, the Gateway routes to the configured
    reasoning model; when set, the Gateway routes to that named model without
    substituting a different one (Req 4.1, 4.4).
    """

    prompt: str
    model: str | None = None  # named override; None -> configured reasoning model
    system: str | None = None  # optional system prompt


class LLMResponse(BaseModel):
    """A reasoning response returned by the LLM Gateway."""

    model: str  # the model that actually produced the response
    content: str


def _make_connector_config() -> "ConnectorConfig":
    """Lazy factory for ConnectorConfig — avoids circular import at module load."""
    from acdp.connectors.config import ConnectorConfig as _CC
    return _CC()


# Rebuild PlatformConfig so the forward-referenced ``ConnectorConfig`` type is
# resolved.  This must happen after the helper is defined and after the
# connectors.config sub-module can safely be imported (it only depends on
# pydantic, so there is no circular dependency at this point).
def _rebuild_platform_config() -> None:
    from acdp.connectors.config import ConnectorConfig as _CC
    PlatformConfig.model_rebuild(_types_namespace={"ConnectorConfig": _CC})


_rebuild_platform_config()
