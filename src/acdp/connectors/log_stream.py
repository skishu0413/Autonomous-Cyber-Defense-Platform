"""LogStreamConnector — continuous log ingestion pipeline feeding BlueTeamAgent.

Supports three ingestion modes:

* ``tail``      — follow a log file, reading new lines as they appear
* ``syslog_udp``— receive UDP datagrams on a configurable host:port
* ``poll_dir``  — periodically scan a directory for new/modified log files

All modes share the same ``_process_line()`` pipeline:
  BlueTeamAgent.parse(raw)  →  BlueTeamAgent.analyze(event)
      → for HIGH/CRITICAL findings: BlueTeamAgent.request_containment()
                                     + audit FINDING_RECORDED record
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from acdp.audit import AuditLog
from acdp.connectors.base import BaseConnector
from acdp.connectors.config import LogStreamConfig
from acdp.models import AuditAction, AuditRecord, Severity, TargetScope

if TYPE_CHECKING:
    from acdp.agents.blue_team_agent import BlueTeamAgent

__all__ = ["LogStreamConnector"]

_HIGH_CRITICAL = {Severity.HIGH, Severity.CRITICAL}


class LogStreamConnector(BaseConnector):
    """Continuous log ingestion connector that feeds raw log lines to BlueTeamAgent.

    Constructor Args:
        config: LogStreamConfig instance.
        blue_team_agent: The BlueTeamAgent to parse/analyze log lines with.
        audit_log: Shared AuditLog instance from the Platform.
        scope: The TargetScope to use for containment requests.
    """

    def __init__(
        self,
        config: LogStreamConfig,
        blue_team_agent: "BlueTeamAgent",
        audit_log: AuditLog,
        scope: TargetScope,
    ) -> None:
        self._config = config
        self._agent = blue_team_agent
        self._audit_log = audit_log
        self._scope = scope
        self._stop_event = asyncio.Event()
        self._thread_stop = threading.Event()
        self._task: asyncio.Task | None = None
        self._thread: threading.Thread | None = None
        self._transport = None

    async def start(self) -> None:
        """Start the appropriate ingestion mode."""
        self._stop_event.clear()
        self._thread_stop.clear()

        mode = self._config.mode
        if mode == "tail":
            self._task = asyncio.create_task(self._run_tail())
        elif mode == "syslog_udp":
            await self._start_syslog_udp()
        elif mode == "poll_dir":
            self._thread = threading.Thread(
                target=self._run_poll_dir,
                daemon=True,
                name="log-stream-poll",
            )
            self._thread.start()
        else:
            raise ValueError(f"Unknown log stream mode: {mode!r}")

    async def stop(self) -> None:
        """Stop the ingestion loop and clean up resources."""
        self._stop_event.set()
        self._thread_stop.set()

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

        if self._transport is not None:
            self._transport.close()
            self._transport = None

        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ------------------------------------------------------------------
    # Shared processing pipeline
    # ------------------------------------------------------------------

    def _process_line(self, raw: str) -> None:
        """Parse, analyze, and optionally trigger containment for one log line.

        1. BlueTeamAgent.parse(raw)  → NormalizedEvent (on TelemetryParseError: audit + continue)
        2. BlueTeamAgent.analyze(event) → list[Finding]
        3. For HIGH/CRITICAL findings: request_containment() + audit FINDING_RECORDED
        """
        from acdp.agents.blue_team_agent import TelemetryParseError

        try:
            event = self._agent.parse(raw)
        except TelemetryParseError:
            # parse() already writes the PARSE_ERROR audit record; just continue
            return
        except Exception as exc:
            # Unexpected error — write audit and continue
            self._audit_log.append(AuditRecord(
                timestamp=datetime.now(timezone.utc),
                actor_id="log_stream",
                action=AuditAction.PARSE_ERROR,
                outcome="error",
                detail={"reason": str(exc), "raw_preview": raw[:200]},
            ))
            return

        try:
            findings = self._agent.analyze(event)
        except Exception:
            # Analysis failure should not halt ingestion
            return

        for finding in findings:
            if finding.severity in _HIGH_CRITICAL:
                try:
                    containment = self._agent.request_containment(finding, self._scope)
                    self._audit_log.append(AuditRecord(
                        timestamp=datetime.now(timezone.utc),
                        actor_id="log_stream",
                        action=AuditAction.FINDING_RECORDED,
                        outcome="containment_triggered",
                        target=finding.asset,
                        detail={
                            "finding_id": finding.finding_id,
                            "severity": finding.severity.value,
                            "containment_status": containment.status,
                        },
                    ))
                except Exception as exc:
                    # Containment failure should not halt ingestion
                    self._audit_log.append(AuditRecord(
                        timestamp=datetime.now(timezone.utc),
                        actor_id="log_stream",
                        action=AuditAction.TASK_FAILED,
                        outcome="containment_error",
                        detail={"finding_id": finding.finding_id, "error": str(exc)},
                    ))

    # ------------------------------------------------------------------
    # Tail mode
    # ------------------------------------------------------------------

    async def _run_tail(self) -> None:
        """Continuously tail the configured log file."""
        log_path = self._config.log_path
        if not log_path:
            raise ValueError("log_path must be set for tail mode")

        retry_count = 0
        fh = None

        while not self._stop_event.is_set():
            try:
                if fh is None:
                    fh = open(log_path, "r", encoding="utf-8", errors="replace")
                    fh.seek(0, 2)  # seek to end
                    retry_count = 0  # reset on successful open

                line = fh.readline()
                if line:
                    self._process_line(line.rstrip("\n"))
                else:
                    await asyncio.sleep(0.1)

            except OSError as exc:
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
                    fh = None

                retry_count += 1
                self._audit_log.append(AuditRecord(
                    timestamp=datetime.now(timezone.utc),
                    actor_id="log_stream",
                    action=AuditAction.FINDING_RECORDED,
                    outcome="tail_warning",
                    detail={
                        "path": str(log_path),
                        "attempt": retry_count,
                        "error": str(exc),
                    },
                ))

                if retry_count > self._config.max_retries:
                    # Exceeded max retries — write final error and raise
                    self._audit_log.append(AuditRecord(
                        timestamp=datetime.now(timezone.utc),
                        actor_id="log_stream",
                        action=AuditAction.TASK_FAILED,
                        outcome="tail_failed",
                        detail={
                            "path": str(log_path),
                            "error": str(exc),
                            "max_retries_exceeded": True,
                        },
                    ))
                    raise

                await asyncio.sleep(self._config.retry_interval_seconds)

        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Syslog UDP mode
    # ------------------------------------------------------------------

    async def _start_syslog_udp(self) -> None:
        """Start a UDP server that processes each datagram as one log line."""
        loop = asyncio.get_event_loop()

        connector = self

        class _SyslogProtocol(asyncio.DatagramProtocol):
            def datagram_received(self, data: bytes, addr):
                try:
                    line = data.decode("utf-8", errors="replace").rstrip("\n")
                    connector._process_line(line)
                except Exception:
                    pass

            def error_received(self, exc):
                connector._audit_log.append(AuditRecord(
                    timestamp=datetime.now(timezone.utc),
                    actor_id="log_stream",
                    action=AuditAction.TASK_FAILED,
                    outcome="udp_error",
                    detail={"error": str(exc)},
                ))

        transport, _ = await loop.create_datagram_endpoint(
            _SyslogProtocol,
            local_addr=(self._config.udp_host, self._config.udp_port),
        )
        self._transport = transport

    # ------------------------------------------------------------------
    # Poll directory mode
    # ------------------------------------------------------------------

    def _run_poll_dir(self) -> None:
        """Periodically scan a directory for new/modified log files."""
        log_path = self._config.log_path or "."
        last_processed: dict[str, float] = {}

        while not self._thread_stop.is_set():
            try:
                for entry in os.scandir(log_path):
                    if not entry.is_file():
                        continue
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        continue

                    last_mtime = last_processed.get(entry.path, 0.0)
                    if mtime > last_mtime:
                        last_processed[entry.path] = mtime
                        try:
                            with open(entry.path, "r", encoding="utf-8", errors="replace") as fh:
                                for line in fh:
                                    self._process_line(line.rstrip("\n"))
                        except OSError as exc:
                            self._audit_log.append(AuditRecord(
                                timestamp=datetime.now(timezone.utc),
                                actor_id="log_stream",
                                action=AuditAction.TASK_FAILED,
                                outcome="poll_warning",
                                detail={"path": entry.path, "error": str(exc)},
                            ))
            except OSError as exc:
                self._audit_log.append(AuditRecord(
                    timestamp=datetime.now(timezone.utc),
                    actor_id="log_stream",
                    action=AuditAction.TASK_FAILED,
                    outcome="poll_dir_warning",
                    detail={"path": str(log_path), "error": str(exc)},
                ))

            self._thread_stop.wait(self._config.poll_interval_seconds)
