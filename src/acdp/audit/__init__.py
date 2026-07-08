"""Append-only audit logging (Layer 1 — Foundation).

Every action-taking component in the platform routes through an
:class:`AuditLog`. The contract is deliberately narrow:

* :meth:`AuditLog.append` stores **exactly one** record, preserving all prior
  records and their order, and returns the stored record with a monotonically
  increasing ``seq`` assigned (Req 12.1, 12.2).
* A failed append raises :class:`~acdp.exceptions.AuditWriteError` rather than
  silently dropping the record (Req 12.4).

The default :class:`JsonlAuditLog` writes one JSON object per line and calls
``os.fsync`` so a committed record survives a crash. The :func:`guarded_action`
helper enforces *audit-before-act*: the audit write must succeed before the
wrapped side effect runs, so a failed append aborts the action (Req 12.4).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Protocol, TypeVar, runtime_checkable

from acdp.exceptions import AuditWriteError
from acdp.models import AuditRecord

__all__ = ["AuditLog", "JsonlAuditLog", "guarded_action"]

T = TypeVar("T")


@runtime_checkable
class AuditLog(Protocol):
    """Append-only audit log interface (Req 12.1, 12.2, 12.4)."""

    def append(self, record: AuditRecord) -> AuditRecord:
        """Append exactly one record, preserving all prior records.

        Returns the stored record with its assigned monotonic sequence id.
        Raises :class:`~acdp.exceptions.AuditWriteError` on failure.
        """
        ...

    def read_all(self) -> list[AuditRecord]:
        """Return all records in append order (read-only, for review/tests)."""
        ...


class JsonlAuditLog:
    """A JSONL-backed append-only audit log.

    Each :meth:`append` writes exactly one JSON line and flushes it to durable
    storage with ``os.fsync``. Sequence numbers are assigned monotonically
    starting at ``1`` and continue across process restarts by inspecting any
    records already present in the target file.

    The instance is safe to share across threads: sequence assignment and the
    write are serialized under a lock so concurrent appends cannot interleave
    partial lines or reuse a ``seq``.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        # Continue the monotonic sequence after any pre-existing records so a
        # restart never reuses or rewinds a seq.
        self._last_seq = self._max_existing_seq()

    def append(self, record: AuditRecord) -> AuditRecord:
        """Append ``record`` as a single fsync'd JSON line (Req 12.1, 12.2, 12.4)."""
        with self._lock:
            stored = record.model_copy(update={"seq": self._last_seq + 1})
            line = json.dumps(stored.model_dump(mode="json"), sort_keys=True)
            try:
                # Line-buffered append; fsync guarantees the record is durable
                # before we acknowledge the write.
                with open(self._path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise AuditWriteError(
                    f"Failed to append audit record to {str(self._path)!r}: {exc}"
                ) from exc

            # Only advance the counter once the write is durably committed.
            self._last_seq += 1
            return stored

    def read_all(self) -> list[AuditRecord]:
        """Return every stored record in append order."""
        try:
            raw_text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise AuditWriteError(
                f"Failed to read audit log {str(self._path)!r}: {exc}"
            ) from exc

        records: list[AuditRecord] = []
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            records.append(AuditRecord.model_validate_json(line))
        return records

    def _max_existing_seq(self) -> int:
        """Return the highest ``seq`` already present, or ``0`` if the log is empty."""
        try:
            existing = self.read_all()
        except AuditWriteError:
            # An unreadable pre-existing file should surface on append, not here.
            return 0
        return max((r.seq for r in existing if r.seq is not None), default=0)


def guarded_action(
    audit_log: AuditLog,
    record: AuditRecord,
    action: Callable[[], T],
) -> T:
    """Run ``action`` only after ``record`` is successfully appended.

    Enforces the *audit-before-act* rule (Req 12.4): the audit append must
    succeed before the side effect runs. If :meth:`AuditLog.append` raises,
    ``action`` is **not** invoked and the :class:`~acdp.exceptions.AuditWriteError`
    is surfaced to the caller.

    Args:
        audit_log: The audit log to append to.
        record: The audit record describing the action about to be taken.
        action: A zero-argument callable performing the side effect.

    Returns:
        The value returned by ``action``.

    Raises:
        AuditWriteError: If the audit append fails; ``action`` is not run.
    """
    audit_log.append(record)  # raises AuditWriteError -> action never runs
    return action()
