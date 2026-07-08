"""Autonomous Cyber Defense Platform (acdp).

A Python multi-agent system that pairs Retrieval-Augmented Generation (RAG)
with agentic AI to defend an organization's digital footprint.

This top-level package re-exports the shared data models and exception types
so downstream modules can import them from a single location.
"""

from __future__ import annotations

from acdp.exceptions import (
    ACDPError,
    AuditWriteError,
    ConfigError,
    IngestionError,
    LLMTimeoutError,
    ModelUnavailableError,
)
from acdp.models import (
    AuditAction,
    AuditRecord,
    EmbeddedChunk,
    Finding,
    KnowledgeSource,
    NormalizedEvent,
    PlatformConfig,
    ScoredChunk,
    Severity,
    SourceCategory,
    Task,
    TaskPlan,
    TaskState,
    TargetScope,
)

__all__ = [
    # models
    "SourceCategory",
    "KnowledgeSource",
    "EmbeddedChunk",
    "ScoredChunk",
    "Severity",
    "Finding",
    "TaskState",
    "Task",
    "TaskPlan",
    "TargetScope",
    "AuditAction",
    "AuditRecord",
    "NormalizedEvent",
    "PlatformConfig",
    # exceptions
    "ACDPError",
    "ConfigError",
    "AuditWriteError",
    "ModelUnavailableError",
    "LLMTimeoutError",
    "IngestionError",
]
