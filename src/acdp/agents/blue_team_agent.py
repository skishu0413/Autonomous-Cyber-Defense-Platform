"""Blue Team Agent — telemetry parsing, anomaly detection, and containment (Layer 4 — Agents).

The :class:`BlueTeamAgent` ingests raw telemetry in three supported formats
(``"json"``, ``"syslog"``, ``"cef"``), normalizes it into :class:`~acdp.models.NormalizedEvent`,
applies configurable anomaly rules to produce severity-classified
:class:`~acdp.models.Finding`s, and routes containment requests through the
:class:`~acdp.authz.AuthorizationService` after retrieving relevant playbook
context from the RAG Core (Req 7.1-7.4, 8.1-8.2, 8.4-8.5).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from acdp.audit import AuditLog, guarded_action
from acdp.authorization import AuthorizationService
from acdp.models import (
    ActionRequest,
    AuditAction,
    AuditRecord,
    Finding,
    NormalizedEvent,
    PlatformConfig,
    Severity,
    SourceCategory,
    TargetScope,
)
from acdp.knowledge_base.retrieve import Retriever

__all__ = [
    "BlueTeamAgent",
    "ContainmentRequest",
    "RawTelemetry",
    "TelemetryParseError",
]

# Type alias for raw telemetry: a plain text string in syslog, CEF, or JSON format.
RawTelemetry = str


class TelemetryParseError(ValueError):
    """Raised when a telemetry record is malformed or in an unsupported format (Req 7.2)."""

    def __init__(self, raw: str, reason: str) -> None:
        self.raw = raw
        self.reason = reason
        super().__init__(f"Telemetry parse error ({reason}): {raw!r}")


@dataclass
class ContainmentRequest:
    """The outcome of a :meth:`BlueTeamAgent.request_containment` call.

    ``status`` is either ``"approved"`` (auto-approved) or ``"pending_approval"``
    (held for operator review when ``config.containment_requires_approval`` is
    True). ``playbook_refs`` lists the RAG source identifiers attached as context.
    ``denied_reason`` is set when the authorization service denied the action.
    """

    finding_id: str
    asset: str
    status: str  # "approved" | "pending_approval" | "denied"
    playbook_refs: list[str] = field(default_factory=list)
    denied_reason: str | None = None


# ---------------------------------------------------------------------------
# Severity ordering used for threshold comparisons
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def _severity_exceeds(sev: Severity, threshold: Severity) -> bool:
    """Return True iff ``sev`` is strictly above ``threshold``."""
    return _SEVERITY_ORDER[sev] > _SEVERITY_ORDER[threshold]


# ---------------------------------------------------------------------------
# Telemetry parsers
# ---------------------------------------------------------------------------

# RFC 3164/5424 syslog header patterns. We accept both the legacy BSD format
# (MMM DD HH:MM:SS host tag[pid]: msg) and a simplified RFC 5424 variant.
_SYSLOG_RFC3164 = re.compile(
    r"^(?P<pri><\d+>)?"
    r"\s*(?P<month>[A-Za-z]+)\s+(?P<day>\d+)\s+(?P<time>\d{2}:\d{2}:\d{2})"
    r"\s+(?P<host>\S+)"
    r"\s+(?P<tag>[^:\s]+)"
    r"(?:\[(?P<pid>\d+)\])?"
    r":\s*(?P<msg>.*)$"
)

_SYSLOG_RFC5424 = re.compile(
    r"^<(?P<pri>\d+)>"
    r"(?P<version>\d)\s+"
    r"(?P<timestamp>\S+)\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<app>\S+)\s+"
    r"(?P<procid>\S+)\s+"
    r"(?P<msgid>\S+)\s+"
    r"(?P<structured>\S+)\s*"
    r"(?P<msg>.*)$"
)

# ArcSight CEF header: CEF:Version|DeviceVendor|DeviceProduct|DeviceVersion|
#                           SignatureID|Name|Severity|Extension
_CEF_HEADER = re.compile(
    r"^CEF:(?P<version>\d+)"
    r"\|(?P<vendor>[^|]*)"
    r"\|(?P<product>[^|]*)"
    r"\|(?P<dev_version>[^|]*)"
    r"\|(?P<sig_id>[^|]*)"
    r"\|(?P<name>[^|]*)"
    r"\|(?P<severity>[^|]*)"
    r"\|(?P<extension>.*)$"
)

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _cef_severity_to_enum(cef_sev: str) -> Severity:
    """Map a CEF numeric severity (0-10) or name to :class:`Severity`."""
    cef_sev = cef_sev.strip().lower()
    # Named mapping
    named = {"low": Severity.LOW, "medium": Severity.MEDIUM,
              "high": Severity.HIGH, "critical": Severity.CRITICAL,
              "info": Severity.INFO, "unknown": Severity.INFO}
    if cef_sev in named:
        return named[cef_sev]
    try:
        n = int(cef_sev)
        if n <= 3:
            return Severity.LOW
        if n <= 6:
            return Severity.MEDIUM
        if n <= 8:
            return Severity.HIGH
        return Severity.CRITICAL
    except ValueError:
        return Severity.INFO


def _parse_cef_extension(ext: str) -> dict[str, str]:
    """Parse a CEF extension string into a key=value dict."""
    result: dict[str, str] = {}
    # CEF extension uses key=value pairs; values may contain spaces, keys are
    # alphanumeric. We tokenize by splitting on key= boundaries.
    pattern = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")
    for match in pattern.finditer(ext):
        result[match.group(1)] = match.group(2).strip()
    return result


# ---------------------------------------------------------------------------
# Anomaly rules
# ---------------------------------------------------------------------------

# Each rule is a callable: (NormalizedEvent, context: dict) -> Finding | None
# context holds per-agent mutable state (e.g. event counters).
# Rules are applied in order; a rule returning None means no match.

_FAILED_LOGIN_THRESHOLD = 5  # repeated failed logins within a session


def _rule_failed_login(event: NormalizedEvent, ctx: dict[str, Any]) -> Finding | None:
    """Detect repeated failed login attempts (brute-force indicator)."""
    action = (event.action or "").lower()
    msg = event.attributes.get("msg", "").lower()
    if "fail" in action or "fail" in msg or "invalid" in msg or "denied" in msg:
        actor = event.actor or event.host or "unknown"
        key = f"failed_login:{actor}"
        ctx[key] = ctx.get(key, 0) + 1
        count = ctx[key]
        if count >= _FAILED_LOGIN_THRESHOLD:
            severity = Severity.HIGH if count >= 10 else Severity.MEDIUM
            return Finding(
                finding_id=str(uuid.uuid4()),
                originating_event_id=event.event_id,
                agent_id="blue_team",
                severity=severity,
                title="Repeated failed login attempts detected",
                detail=(
                    f"Actor {actor!r} has {count} failed login attempt(s). "
                    "Possible brute-force or credential-stuffing attack."
                ),
                asset=event.host,
                created_at=datetime.now(timezone.utc),
            )
    return None


def _rule_high_severity_event(event: NormalizedEvent, ctx: dict[str, Any]) -> Finding | None:
    """Escalate events that already carry HIGH or CRITICAL severity."""
    if event.severity in (Severity.HIGH, Severity.CRITICAL):
        return Finding(
            finding_id=str(uuid.uuid4()),
            originating_event_id=event.event_id,
            agent_id="blue_team",
            severity=event.severity,
            title=f"High-severity event from {event.host or 'unknown host'}",
            detail=(
                f"Received event with severity {event.severity.value!r}. "
                f"Action: {event.action or 'N/A'}. Actor: {event.actor or 'N/A'}."
            ),
            asset=event.host,
            created_at=datetime.now(timezone.utc),
        )
    return None


def _rule_suspicious_actor(event: NormalizedEvent, ctx: dict[str, Any]) -> Finding | None:
    """Flag events from known suspicious actor patterns."""
    suspicious_patterns = ["root", "admin", "anonymous", "guest", "system"]
    actor = (event.actor or "").lower()
    if actor in suspicious_patterns:
        return Finding(
            finding_id=str(uuid.uuid4()),
            originating_event_id=event.event_id,
            agent_id="blue_team",
            severity=Severity.MEDIUM,
            title=f"Event from privileged/suspicious actor: {event.actor!r}",
            detail=(
                f"Actor {event.actor!r} triggered action {event.action or 'N/A'!r} "
                f"on host {event.host or 'N/A'}. Privileged accounts warrant review."
            ),
            asset=event.host,
            created_at=datetime.now(timezone.utc),
        )
    return None


def _rule_port_scan(event: NormalizedEvent, ctx: dict[str, Any]) -> Finding | None:
    """Detect port-scanning indicators in the event attributes."""
    msg = event.attributes.get("msg", "").lower()
    action = (event.action or "").lower()
    port_scan_signals = ["port scan", "portscan", "nmap", "masscan", "scan detected"]
    if any(sig in msg for sig in port_scan_signals) or any(sig in action for sig in port_scan_signals):
        return Finding(
            finding_id=str(uuid.uuid4()),
            originating_event_id=event.event_id,
            agent_id="blue_team",
            severity=Severity.HIGH,
            title="Port scanning activity detected",
            detail=(
                f"Port scan indicator found in event from {event.host or 'unknown'}. "
                f"Actor: {event.actor or 'N/A'}. Message: {event.attributes.get('msg', 'N/A')}"
            ),
            asset=event.host,
            created_at=datetime.now(timezone.utc),
        )
    return None


def _rule_privilege_escalation(event: NormalizedEvent, ctx: dict[str, Any]) -> Finding | None:
    """Detect privilege escalation keywords in the event."""
    msg = event.attributes.get("msg", "").lower()
    action = (event.action or "").lower()
    escalation_signals = ["sudo", "su root", "privilege escalat", "setuid", "chmod 777", "passwd"]
    if any(sig in msg for sig in escalation_signals) or any(sig in action for sig in escalation_signals):
        return Finding(
            finding_id=str(uuid.uuid4()),
            originating_event_id=event.event_id,
            agent_id="blue_team",
            severity=Severity.CRITICAL,
            title="Potential privilege escalation detected",
            detail=(
                f"Privilege escalation indicator found in event from {event.host or 'unknown'}. "
                f"Actor: {event.actor or 'N/A'}. Action: {event.action or 'N/A'}."
            ),
            asset=event.host,
            created_at=datetime.now(timezone.utc),
        )
    return None


_ANOMALY_RULES = [
    _rule_failed_login,
    _rule_high_severity_event,
    _rule_suspicious_actor,
    _rule_port_scan,
    _rule_privilege_escalation,
]


# ---------------------------------------------------------------------------
# BlueTeamAgent
# ---------------------------------------------------------------------------

class BlueTeamAgent:
    """Blue Team Agent: telemetry ingestion, anomaly detection, and containment.

    Constructor Args:
        audit_log: Append-only log; every parse error and containment decision
            is recorded here (Req 7.2, 8.3).
        config: Platform configuration supplying ``severity_threshold`` and
            ``containment_requires_approval``.
        retriever: RAG retriever for playbook context retrieval (Req 8.4).
        authz: Authorization service for containment gating (Req 8.2, 8.3).
        actor_id: Actor identifier recorded on audit records.
    """

    def __init__(
        self,
        audit_log: AuditLog,
        config: PlatformConfig,
        retriever: Retriever,
        authz: AuthorizationService,
        *,
        actor_id: str = "blue_team",
    ) -> None:
        self._audit_log = audit_log
        self._config = config
        self._retriever = retriever
        self._authz = authz
        self._actor_id = actor_id
        # Per-agent mutable state for stateful anomaly rules (e.g. failure counters).
        self._rule_ctx: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Telemetry parse / serialize  (Req 7.1, 7.2, 7.3, 7.4)
    # ------------------------------------------------------------------

    def parse(self, raw: RawTelemetry) -> NormalizedEvent:
        """Parse ``raw`` telemetry into a :class:`~acdp.models.NormalizedEvent`.

        Supported formats are ``"json"``, ``"syslog"`` (RFC 3164/5424), and
        ``"cef"`` (ArcSight CEF). Malformed records cause a
        :class:`TelemetryParseError` to be raised **after** appending a
        ``PARSE_ERROR`` audit record naming the record (Req 7.2).

        Returns the :class:`~acdp.models.NormalizedEvent` on success.
        """
        raw = raw.strip()
        try:
            if raw.startswith("{"):
                return self._parse_json(raw)
            if raw.startswith("CEF:"):
                return self._parse_cef(raw)
            return self._parse_syslog(raw)
        except TelemetryParseError:
            raise
        except Exception as exc:
            self._audit_parse_error(raw, str(exc))
            raise TelemetryParseError(raw, str(exc)) from exc

    def _audit_parse_error(self, raw: str, reason: str) -> None:
        """Append a PARSE_ERROR audit record naming the malformed record."""
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            actor_id=self._actor_id,
            action=AuditAction.PARSE_ERROR,
            outcome="rejected",
            target=raw[:120],  # truncate very long records for the audit trail
            detail={"reason": reason, "raw_preview": raw[:200]},
        )
        self._audit_log.append(record)

    def _parse_json(self, raw: str) -> NormalizedEvent:
        """Parse a JSON-formatted telemetry record."""
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._audit_parse_error(raw, f"invalid JSON: {exc}")
            raise TelemetryParseError(raw, f"invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            self._audit_parse_error(raw, "JSON root must be an object")
            raise TelemetryParseError(raw, "JSON root must be an object")

        try:
            event_id = str(data.get("event_id") or str(uuid.uuid4()))
            ts_raw = data.get("timestamp") or data.get("ts")
            if ts_raw is None:
                timestamp = datetime.now(timezone.utc)
            elif isinstance(ts_raw, (int, float)):
                timestamp = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
            else:
                timestamp = datetime.fromisoformat(str(ts_raw))

            severity_raw = str(data.get("severity", "info")).lower()
            try:
                severity = Severity(severity_raw)
            except ValueError:
                severity = Severity.INFO

            # Canonical attributes: all remaining string-valued keys, sorted.
            reserved = {"event_id", "timestamp", "ts", "host", "actor", "action", "severity", "source_format"}
            attributes = {
                k: str(v)
                for k, v in sorted(data.items())
                if k not in reserved and v is not None
            }

            return NormalizedEvent(
                event_id=event_id,
                source_format="json",
                timestamp=timestamp,
                host=str(data["host"]) if data.get("host") else None,
                actor=str(data["actor"]) if data.get("actor") else None,
                action=str(data["action"]) if data.get("action") else None,
                severity=severity,
                attributes=attributes,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._audit_parse_error(raw, f"JSON field error: {exc}")
            raise TelemetryParseError(raw, f"JSON field error: {exc}") from exc

    def _parse_syslog(self, raw: str) -> NormalizedEvent:
        """Parse an RFC 3164 or 5424 syslog record."""
        # Try RFC 5424 first (has version number after priority)
        m5 = _SYSLOG_RFC5424.match(raw)
        if m5:
            try:
                ts_str = m5.group("timestamp")
                if ts_str == "-":
                    timestamp = datetime.now(timezone.utc)
                else:
                    timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                host = m5.group("host") if m5.group("host") != "-" else None
                app = m5.group("app") if m5.group("app") != "-" else None
                msg = m5.group("msg").strip()
                attributes = dict(sorted({
                    "pri": m5.group("pri"),
                    "version": m5.group("version"),
                    "app": app or "",
                    "procid": m5.group("procid"),
                    "msgid": m5.group("msgid"),
                    "msg": msg,
                }.items()))
                # Filter empty values
                attributes = {k: v for k, v in attributes.items() if v and v != "-"}
                return NormalizedEvent(
                    event_id=str(uuid.uuid4()),
                    source_format="syslog",
                    timestamp=timestamp,
                    host=host,
                    actor=app,
                    action=None,
                    severity=Severity.INFO,
                    attributes=attributes,
                )
            except (ValueError, AttributeError) as exc:
                # Fall through to RFC 3164
                pass

        # Try RFC 3164
        m3 = _SYSLOG_RFC3164.match(raw)
        if m3:
            try:
                month_str = m3.group("month").lower()[:3]
                month = _MONTH_MAP.get(month_str, 1)
                day = int(m3.group("day"))
                time_str = m3.group("time")
                year = datetime.now(timezone.utc).year
                timestamp = datetime.strptime(
                    f"{year}-{month:02d}-{day:02d}T{time_str}",
                    "%Y-%m-%dT%H:%M:%S"
                ).replace(tzinfo=timezone.utc)
                host = m3.group("host")
                tag_raw = m3.group("tag")
                # Strip [pid] from tag if present; pid is stored in attributes
                tag_pid_match = re.match(r'^(\w[\w.\-]*)(?:\[(\d+)\])?$', tag_raw)
                if tag_pid_match:
                    tag = tag_pid_match.group(1)
                    pid = tag_pid_match.group(2) or m3.group("pid")
                else:
                    tag = tag_raw
                    pid = m3.group("pid")
                msg = m3.group("msg").strip()
                attributes = dict(sorted({
                    "tag": tag,
                    "msg": msg,
                    **({"pid": pid} if pid else {}),
                    **({"pri": m3.group("pri").strip("<>") if m3.group("pri") else ""})
                }.items()))
                attributes = {k: v for k, v in attributes.items() if v}
                return NormalizedEvent(
                    event_id=str(uuid.uuid4()),
                    source_format="syslog",
                    timestamp=timestamp,
                    host=host,
                    actor=tag,
                    action=None,
                    severity=Severity.INFO,
                    attributes=attributes,
                )
            except (ValueError, AttributeError) as exc:
                self._audit_parse_error(raw, f"syslog parse error: {exc}")
                raise TelemetryParseError(raw, f"syslog parse error: {exc}") from exc

        # Neither pattern matched
        self._audit_parse_error(raw, "unrecognized syslog format")
        raise TelemetryParseError(raw, "unrecognized syslog format")

    def _parse_cef(self, raw: str) -> NormalizedEvent:
        """Parse an ArcSight CEF telemetry record."""
        m = _CEF_HEADER.match(raw)
        if not m:
            self._audit_parse_error(raw, "invalid CEF header")
            raise TelemetryParseError(raw, "invalid CEF header")

        try:
            severity = _cef_severity_to_enum(m.group("severity"))
            ext = _parse_cef_extension(m.group("extension"))

            # Extract standard CEF extension fields
            host = ext.pop("dhost", None) or ext.pop("src", None)
            actor = ext.pop("suser", None) or ext.pop("duser", None)
            action = m.group("name") or ext.pop("act", None)

            # Build canonical attributes from all remaining extension fields + header fields
            attributes = dict(sorted({
                "cef_version": m.group("version"),
                "vendor": m.group("vendor"),
                "product": m.group("product"),
                "dev_version": m.group("dev_version"),
                "sig_id": m.group("sig_id"),
                **ext,
            }.items()))
            attributes = {k: v for k, v in attributes.items() if v}

            # Timestamp from rt (receipt time) or start fields in extension
            ts_str = ext.get("rt") or ext.get("start") or ext.get("end")
            if ts_str:
                try:
                    timestamp = datetime.fromtimestamp(int(ts_str) / 1000, tz=timezone.utc)
                except (ValueError, TypeError):
                    try:
                        timestamp = datetime.fromisoformat(ts_str)
                    except ValueError:
                        timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            return NormalizedEvent(
                event_id=str(uuid.uuid4()),
                source_format="cef",
                timestamp=timestamp,
                host=host,
                actor=actor,
                action=action,
                severity=severity,
                attributes=attributes,
            )
        except (KeyError, ValueError, AttributeError) as exc:
            self._audit_parse_error(raw, f"CEF parse error: {exc}")
            raise TelemetryParseError(raw, f"CEF parse error: {exc}") from exc

    def serialize(self, event: NormalizedEvent) -> RawTelemetry:
        """Serialize a :class:`~acdp.models.NormalizedEvent` back to canonical telemetry.

        The canonical representation is the format determined by
        ``event.source_format``. For round-trip determinism, the ``attributes``
        dict is serialized with sorted keys (Req 7.3, 7.4).
        """
        fmt = event.source_format.lower()
        if fmt == "json":
            return self._serialize_json(event)
        if fmt == "syslog":
            return self._serialize_syslog(event)
        if fmt == "cef":
            return self._serialize_cef(event)
        # Unknown format: fall back to JSON
        return self._serialize_json(event)

    def _serialize_json(self, event: NormalizedEvent) -> str:
        """Serialize to a canonical JSON string."""
        data: dict[str, Any] = {
            "event_id": event.event_id,
            "source_format": event.source_format,
            "timestamp": event.timestamp.isoformat(),
            "severity": event.severity.value,
        }
        if event.host is not None:
            data["host"] = event.host
        if event.actor is not None:
            data["actor"] = event.actor
        if event.action is not None:
            data["action"] = event.action
        # Merge attributes with canonical sorted ordering
        for k, v in sorted(event.attributes.items()):
            data[k] = v
        return json.dumps(data, sort_keys=True)

    def _serialize_syslog(self, event: NormalizedEvent) -> str:
        """Serialize to a canonical RFC 3164-style syslog line."""
        # Format: MMM DD HH:MM:SS host tag: msg
        ts = event.timestamp
        month_abbr = ts.strftime("%b")
        day = f"{ts.day:2d}"
        time_str = ts.strftime("%H:%M:%S")
        host = event.host or "localhost"
        tag = event.actor or event.attributes.get("tag", "blue_team")
        msg = event.attributes.get("msg", event.action or "")
        # Include pid if present
        pid = event.attributes.get("pid", "")
        tag_part = f"{tag}[{pid}]" if pid else tag
        return f"{month_abbr} {day} {time_str} {host} {tag_part}: {msg}"

    def _serialize_cef(self, event: NormalizedEvent) -> str:
        """Serialize to a canonical ArcSight CEF line."""
        # Map severity enum back to CEF numeric
        sev_map = {
            Severity.INFO: "0",
            Severity.LOW: "3",
            Severity.MEDIUM: "5",
            Severity.HIGH: "8",
            Severity.CRITICAL: "10",
        }
        cef_sev = sev_map.get(event.severity, "0")
        vendor = event.attributes.get("vendor", "ACDP")
        product = event.attributes.get("product", "BlueTeam")
        dev_version = event.attributes.get("dev_version", "1.0")
        sig_id = event.attributes.get("sig_id", event.event_id)
        name = event.action or "event"
        cef_version = event.attributes.get("cef_version", "0")

        # Build extension from remaining attributes + standard fields
        ext_parts: dict[str, str] = {}
        reserved_attr_keys = {"vendor", "product", "dev_version", "sig_id", "cef_version"}
        for k, v in sorted(event.attributes.items()):
            if k not in reserved_attr_keys:
                ext_parts[k] = v
        if event.host:
            ext_parts["dhost"] = event.host
        if event.actor:
            ext_parts["suser"] = event.actor
        # Timestamp as rt (epoch ms)
        ext_parts["rt"] = str(int(event.timestamp.timestamp() * 1000))

        ext_str = " ".join(f"{k}={v}" for k, v in sorted(ext_parts.items()))
        return (
            f"CEF:{cef_version}|{vendor}|{product}|{dev_version}"
            f"|{sig_id}|{name}|{cef_sev}|{ext_str}"
        )

    # ------------------------------------------------------------------
    # Anomaly detection  (Req 8.1)
    # ------------------------------------------------------------------

    def analyze(self, event: NormalizedEvent) -> list[Finding]:
        """Apply anomaly rules to ``event`` and return severity-classified findings.

        Each built-in anomaly rule (failed logins, high-severity escalation,
        suspicious actor, port scan, privilege escalation) is applied in order.
        Rules that match the event produce a :class:`~acdp.models.Finding`
        classified by severity (Req 8.1).
        """
        findings: list[Finding] = []
        for rule in _ANOMALY_RULES:
            finding = rule(event, self._rule_ctx)
            if finding is not None:
                findings.append(finding)
        return findings

    # ------------------------------------------------------------------
    # Containment request  (Req 8.2, 8.4, 8.5)
    # ------------------------------------------------------------------

    def request_containment(
        self, finding: Finding, scope: TargetScope
    ) -> ContainmentRequest:
        """Request containment for a finding against ``scope``.

        Only proceeds when ``finding.severity`` strictly exceeds the configured
        ``severity_threshold`` (Req 8.2). Retrieves playbook context from the
        RAG Core and attaches non-empty context refs to the finding (Req 8.4).
        Routes through :class:`~acdp.authz.AuthorizationService`; when
        ``containment_requires_approval`` is configured, holds the request
        pending approval rather than auto-approving (Req 8.5).

        Returns a :class:`ContainmentRequest` describing the outcome.
        """
        asset = finding.asset or (scope.assets[0] if scope.assets else "unknown")

        # Req 8.2: only proceed if severity strictly exceeds the threshold
        if not _severity_exceeds(finding.severity, self._config.severity_threshold):
            return ContainmentRequest(
                finding_id=finding.finding_id,
                asset=asset,
                status="denied",
                denied_reason=(
                    f"Severity {finding.severity.value!r} does not exceed "
                    f"threshold {self._config.severity_threshold.value!r}"
                ),
            )

        # Req 8.4: retrieve playbook context; must be non-empty
        query = f"containment playbook for {finding.title}: {finding.detail[:120]}"
        playbook_chunks = self._retriever.retrieve(
            query, category=SourceCategory.PLAYBOOK
        )
        if not playbook_chunks:
            # Also try without category filter as a fallback
            playbook_chunks = self._retriever.retrieve(query)

        playbook_refs = [c.chunk.source_id for c in playbook_chunks]

        # Route through Authorization Service (Req 8.3)
        req = ActionRequest(
            agent_id=self._actor_id,
            asset=asset,
            action="containment",
        )
        decision = self._authz.authorize(req, at=datetime.now(timezone.utc))

        if not decision.grant:
            return ContainmentRequest(
                finding_id=finding.finding_id,
                asset=asset,
                status="denied",
                playbook_refs=playbook_refs,
                denied_reason=decision.reason,
            )

        # Req 8.5: hold pending approval if configured
        if self._config.containment_requires_approval:
            status = "pending_approval"
        else:
            status = "approved"

        return ContainmentRequest(
            finding_id=finding.finding_id,
            asset=asset,
            status=status,
            playbook_refs=playbook_refs,
        )
