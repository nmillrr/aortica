"""Urgent-finding notification pipeline (US-126).

Monitors AI results for critical findings and pushes alerts to configured
EHR channels — FHIR ``CommunicationRequest``, HL7 v2.x, a generic webhook,
and SMTP email.  Trigger rules (conditions, minimum confidence, urgency
threshold, de-duplication window) are configurable.  Every delivery attempt
is tracked in SQLite so :func:`GET /api/v1/notifications` can report history.

Channel senders are injectable callables so the notifier is fully testable
with mock EHR endpoints and the app never hard-depends on an SMTP/FHIR
server.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

CHANNELS = ("fhir", "hl7", "webhook", "email")

#: A channel sender takes the notification payload and returns True on success.
ChannelSender = Callable[[Dict[str, Any]], bool]


@dataclass
class NotificationRules:
    """Trigger rules for urgent-finding notifications.

    Attributes:
        conditions: Only notify for these finding names (empty = any).
        min_confidence: Minimum finding confidence (0–1).
        urgency_threshold: Minimum urgency score (0–100).
        dedup_window_hours: Suppress a repeat notification for the same
            ``patient + finding`` within this many hours.
        channels: Which channels to deliver on.
    """

    conditions: List[str] = field(default_factory=list)
    min_confidence: float = 0.5
    urgency_threshold: int = 0
    dedup_window_hours: float = 24.0
    channels: List[str] = field(default_factory=lambda: ["webhook"])

    def __post_init__(self) -> None:
        unknown = set(self.channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"Unknown channels: {sorted(unknown)}")


def load_notification_rules(path: str | Path) -> NotificationRules:
    """Load ``notification_rules.yaml`` into :class:`NotificationRules`."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")
    text = path_obj.read_text()
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
    except ImportError:
        from aortica.federated.fl_server import _simple_yaml_load

        data = _simple_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError("notification_rules.yaml must be a mapping")
    return NotificationRules(
        conditions=list(data.get("conditions", [])),
        min_confidence=float(data.get("min_confidence", 0.5)),
        urgency_threshold=int(data.get("urgency_threshold", 0)),
        dedup_window_hours=float(data.get("dedup_window_hours", 24.0)),
        channels=list(data.get("channels", ["webhook"])),
    )


@dataclass
class NotificationRecord:
    """A tracked notification delivery."""

    id: str
    patient_id: Optional[str]
    ecg_id: str
    finding: str
    confidence: float
    urgency_score: int
    channel: str
    status: str  # sent | failed
    created_at: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "ecg_id": self.ecg_id,
            "finding": self.finding,
            "confidence": self.confidence,
            "urgency_score": self.urgency_score,
            "channel": self.channel,
            "status": self.status,
            "created_at": self.created_at,
            "error": self.error,
        }


class UrgentFindingNotifier:
    """Delivers urgent-finding alerts across channels with de-duplication.

    Args:
        rules: Trigger rules.
        db_path: SQLite path for delivery tracking (``:memory:`` allowed).
        senders: Optional per-channel sender overrides (for testing). Any
            channel without an override uses a stdlib default.
        result_url_base: Base URL used to build the Aortica result link.
    """

    def __init__(
        self,
        rules: NotificationRules,
        *,
        db_path: str = ":memory:",
        senders: Optional[Dict[str, ChannelSender]] = None,
        result_url_base: str = "https://aortica.local/results",
    ) -> None:
        self._rules = rules
        self._senders = dict(senders or {})
        self._result_url_base = result_url_base
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                patient_id TEXT,
                ecg_id TEXT NOT NULL,
                finding TEXT NOT NULL,
                confidence REAL NOT NULL,
                urgency_score INTEGER NOT NULL,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                error TEXT
            )
            """
        )
        self._conn.commit()

    # -- Public API ---------------------------------------------------------

    def notify(
        self,
        multi_task_output: Any,
        ecg_id: str,
        *,
        patient_id: Optional[str] = None,
    ) -> List[NotificationRecord]:
        """Evaluate a result and deliver notifications for matching findings.

        Returns the delivery records created (one per matched finding ×
        channel).  De-duplicated findings and non-matching results produce
        no records.
        """
        from aortica.integration.worklist import WorklistPrioritizer

        worklist = WorklistPrioritizer().prioritize(
            [multi_task_output], ecg_ids=[ecg_id]
        )
        item = worklist.items[0]
        if item.urgency_score < self._rules.urgency_threshold:
            return []

        records: List[NotificationRecord] = []
        for finding in item.active_findings:
            name = finding.get("class_name", "")
            confidence = float(finding.get("confidence", 0.0))
            if confidence < self._rules.min_confidence:
                continue
            if self._rules.conditions and name not in self._rules.conditions:
                continue
            if self._is_duplicate(patient_id, name):
                continue

            payload = self._build_payload(
                item, finding, ecg_id, patient_id
            )
            for channel in self._rules.channels:
                records.append(
                    self._deliver(channel, payload, item, finding, ecg_id, patient_id)
                )
        return records

    def history(
        self, *, patient_id: Optional[str] = None, limit: int = 500
    ) -> List[NotificationRecord]:
        """Return delivery history (newest first)."""
        with self._lock:
            if patient_id:
                rows = self._conn.execute(
                    "SELECT * FROM notifications WHERE patient_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (patient_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM notifications ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # -- Internals ----------------------------------------------------------

    def _is_duplicate(self, patient_id: Optional[str], finding: str) -> bool:
        if patient_id is None:
            return False
        cutoff = time.time() - self._rules.dedup_window_hours * 3600
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM notifications "
                "WHERE patient_id = ? AND finding = ? AND status = 'sent' "
                "AND created_at >= ?",
                (patient_id, finding, cutoff),
            ).fetchone()
        return bool(row and row[0] > 0)

    def _build_payload(
        self,
        item: Any,
        finding: Dict[str, Any],
        ecg_id: str,
        patient_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "patient_id": patient_id,
            "finding": finding.get("class_name"),
            "confidence": finding.get("confidence"),
            "urgency_score": item.urgency_score,
            "recommended_action": item.recommended_action,
            "result_url": f"{self._result_url_base}/{ecg_id}",
            "timestamp": time.time(),
        }

    def _deliver(
        self,
        channel: str,
        payload: Dict[str, Any],
        item: Any,
        finding: Dict[str, Any],
        ecg_id: str,
        patient_id: Optional[str],
    ) -> NotificationRecord:
        sender = self._senders.get(channel) or self._default_sender(channel)
        error: Optional[str] = None
        try:
            ok = bool(sender(payload))
        except Exception as exc:  # noqa: BLE001 - track, don't raise
            ok = False
            error = str(exc)

        record = NotificationRecord(
            id=f"ntf-{uuid.uuid4().hex[:12]}",
            patient_id=patient_id,
            ecg_id=ecg_id,
            finding=str(finding.get("class_name", "")),
            confidence=float(finding.get("confidence", 0.0)),
            urgency_score=item.urgency_score,
            channel=channel,
            status="sent" if ok else "failed",
            created_at=time.time(),
            error=error,
        )
        self._persist(record)
        return record

    def _persist(self, record: NotificationRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO notifications (id, patient_id, ecg_id, finding, "
                "confidence, urgency_score, channel, status, created_at, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id, record.patient_id, record.ecg_id, record.finding,
                    record.confidence, record.urgency_score, record.channel,
                    record.status, record.created_at, record.error,
                ),
            )
            self._conn.commit()

    # -- Default channel senders (stdlib) -----------------------------------

    def _default_sender(self, channel: str) -> ChannelSender:
        if channel == "fhir":
            return self._send_fhir
        if channel == "hl7":
            return self._send_hl7
        if channel == "webhook":
            return self._send_webhook
        if channel == "email":
            return self._send_email
        raise ValueError(f"Unknown channel {channel!r}")

    @staticmethod
    def _send_fhir(payload: Dict[str, Any]) -> bool:  # pragma: no cover - needs server
        # A real deployment POSTs a CommunicationRequest to the FHIR server.
        logger.info("FHIR CommunicationRequest (noop default): %s", payload["finding"])
        return False

    @staticmethod
    def _send_hl7(payload: Dict[str, Any]) -> bool:  # pragma: no cover - needs MLLP
        logger.info("HL7 alert (noop default): %s", payload["finding"])
        return False

    @staticmethod
    def _send_webhook(payload: Dict[str, Any]) -> bool:  # pragma: no cover - needs URL
        logger.info("Webhook alert (noop default): %s", payload["finding"])
        return False

    @staticmethod
    def _send_email(payload: Dict[str, Any]) -> bool:  # pragma: no cover - needs SMTP
        logger.info("Email alert (noop default): %s", payload["finding"])
        return False

    @staticmethod
    def _row_to_record(row: Any) -> NotificationRecord:
        return NotificationRecord(
            id=row[0], patient_id=row[1], ecg_id=row[2], finding=row[3],
            confidence=row[4], urgency_score=row[5], channel=row[6],
            status=row[7], created_at=row[8], error=row[9],
        )

    @staticmethod
    def build_communication_request(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a FHIR R4 CommunicationRequest resource for *payload*."""
        return {
            "resourceType": "CommunicationRequest",
            "status": "active",
            "priority": "urgent",
            "subject": (
                {"reference": f"Patient/{payload['patient_id']}"}
                if payload.get("patient_id")
                else None
            ),
            "payload": [
                {
                    "contentString": (
                        f"Urgent AI finding: {payload['finding']} "
                        f"(confidence {payload['confidence']}, urgency "
                        f"{payload['urgency_score']}). "
                        f"{payload['recommended_action']} "
                        f"Review: {payload['result_url']}"
                    )
                }
            ],
        }

    @staticmethod
    def json_payload(payload: Dict[str, Any]) -> str:
        return json.dumps(payload)
