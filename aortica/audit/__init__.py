"""Tamper-evident audit trail for clinical decision-support (US-121)."""

from __future__ import annotations

from aortica.audit.logger import (
    EVENT_TYPES,
    AuditEvent,
    AuditLogger,
    IntegrityReport,
    verify_integrity,
)

__all__ = [
    "EVENT_TYPES",
    "AuditEvent",
    "AuditLogger",
    "IntegrityReport",
    "verify_integrity",
]
