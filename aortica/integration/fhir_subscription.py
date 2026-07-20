"""FHIR Subscription and webhook notification service (US-117).

Lets an EHR register interest in critical Aortica findings and receive a
push notification (FHIR ``subscription-notification`` Bundle) over a
``rest-hook`` channel whenever a prediction matches its criteria — no
polling required.

The :class:`SubscriptionManager` is self-contained (stdlib only) and holds
subscriptions plus per-subscription delivery history in memory behind a
lock, so it is safe to share across a threaded FastAPI server.  Webhook
delivery runs on a background thread with bounded exponential-backoff
retries and a dead-letter queue for persistently failed notifications.

Example::

    mgr = SubscriptionManager()
    sub = mgr.create_subscription(
        "https://ehr.example/hook",
        SubscriptionCriteria(min_severity="critical"),
    )
    mgr.process_result(multi_task_output, ecg_id="ecg_1")
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Severity model
# ---------------------------------------------------------------------------

#: Urgency tiers (from worklist) mapped to notification severities.
_TIER_SEVERITY: Dict[str, str] = {
    "critical": "critical",
    "high": "warning",
    "moderate": "warning",
    "low": "info",
}

#: Severity ordering for the ``min_severity`` filter (higher = more severe).
_SEVERITY_RANK: Dict[str, int] = {"info": 0, "warning": 1, "critical": 2}

VALID_SEVERITIES = ("info", "warning", "critical")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SubscriptionCriteria:
    """Filter describing which findings a subscription cares about.

    Attributes:
        min_severity: Minimum finding severity to notify on
            (``"info"``/``"warning"``/``"critical"``).  A finding must be at
            least this severe.
        conditions: If non-empty, only notify when at least one of these
            condition names is among the active findings (e.g. ``["STEMI",
            "VT", "VF", "brugada_pattern"]``).
        urgency_threshold: Minimum urgency score (0–100) required to notify.
    """

    min_severity: str = "critical"
    conditions: List[str] = field(default_factory=list)
    urgency_threshold: int = 0

    def __post_init__(self) -> None:
        if self.min_severity not in VALID_SEVERITIES:
            raise ValueError(
                f"min_severity must be one of {VALID_SEVERITIES}, "
                f"got {self.min_severity!r}"
            )
        if not 0 <= self.urgency_threshold <= 100:
            raise ValueError("urgency_threshold must be in 0..100")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_severity": self.min_severity,
            "conditions": list(self.conditions),
            "urgency_threshold": self.urgency_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubscriptionCriteria:
        return cls(
            min_severity=data.get("min_severity", "critical"),
            conditions=list(data.get("conditions", [])),
            urgency_threshold=int(data.get("urgency_threshold", 0)),
        )


@dataclass
class Subscription:
    """A registered FHIR Subscription with a rest-hook channel."""

    id: str
    webhook_url: str
    criteria: SubscriptionCriteria
    channel_type: str = "rest-hook"
    active: bool = True
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "webhook_url": self.webhook_url,
            "criteria": self.criteria.to_dict(),
            "channel_type": self.channel_type,
            "active": self.active,
            "created_at": self.created_at,
        }

    def to_fhir(self) -> Dict[str, Any]:
        """Render as a FHIR R4 Subscription resource."""
        crit_parts = [f"severity>={self.criteria.min_severity}"]
        if self.criteria.conditions:
            crit_parts.append("conditions=" + ",".join(self.criteria.conditions))
        if self.criteria.urgency_threshold:
            crit_parts.append(f"urgency>={self.criteria.urgency_threshold}")
        return {
            "resourceType": "Subscription",
            "id": self.id,
            "status": "active" if self.active else "off",
            "reason": "Aortica critical-finding notification",
            "criteria": "DiagnosticReport?" + "&".join(crit_parts),
            "channel": {
                "type": self.channel_type,
                "endpoint": self.webhook_url,
                "payload": "application/fhir+json",
            },
        }


@dataclass
class NotificationRecord:
    """One webhook delivery attempt sequence for a matched result."""

    id: str
    subscription_id: str
    status: str  # "pending" | "sent" | "failed" | "dead-letter"
    ecg_id: str
    urgency_score: int
    matched_findings: List[Dict[str, Any]]
    attempts: int = 0
    created_at: str = field(default_factory=_now_iso)
    delivered_at: Optional[str] = None
    error: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subscription_id": self.subscription_id,
            "status": self.status,
            "ecg_id": self.ecg_id,
            "urgency_score": self.urgency_score,
            "matched_findings": list(self.matched_findings),
            "attempts": self.attempts,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Default webhook transport
# ---------------------------------------------------------------------------


def _default_http_post(url: str, payload: Dict[str, Any], timeout: float) -> int:
    """POST *payload* as JSON to *url*; return the HTTP status code.

    Raises on transport error (caller treats any exception as a failure).
    """
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/fhir+json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return int(resp.status)


# ---------------------------------------------------------------------------
# Subscription manager
# ---------------------------------------------------------------------------


class SubscriptionManager:
    """Manages FHIR subscriptions and delivers webhook notifications.

    Args:
        http_post: Callable ``(url, payload, timeout) -> status_code`` used
            to deliver webhooks.  Defaults to a urllib-based POST; override
            in tests.  A non-2xx status or any raised exception counts as a
            failed attempt.
        max_retries: Total delivery attempts before dead-lettering.
        backoff_base: Base seconds for exponential backoff between retries.
        sleep: Sleep function (injectable so tests avoid real delays).
        timeout: Per-request timeout in seconds.
        prioritizer: Optional worklist prioritizer for scoring results.
    """

    def __init__(
        self,
        *,
        http_post: Optional[Callable[[str, Dict[str, Any], float], int]] = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 5.0,
        prioritizer: Optional[Any] = None,
    ) -> None:
        self._http_post = http_post or _default_http_post
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._timeout = timeout
        self._prioritizer = prioritizer

        self._lock = threading.RLock()
        self._subs: Dict[str, Subscription] = {}
        self._notifications: Dict[str, List[NotificationRecord]] = {}
        self._dead_letter: List[NotificationRecord] = []
        self._threads: List[threading.Thread] = []

    # -- CRUD ---------------------------------------------------------------

    def create_subscription(
        self,
        webhook_url: str,
        criteria: SubscriptionCriteria,
        channel_type: str = "rest-hook",
    ) -> Subscription:
        """Register a new subscription."""
        if channel_type != "rest-hook":
            raise ValueError(
                f"Only 'rest-hook' channel supported, got {channel_type!r}"
            )
        if not webhook_url:
            raise ValueError("webhook_url must not be empty")
        sub = Subscription(
            id=_new_id("sub"),
            webhook_url=webhook_url,
            criteria=criteria,
            channel_type=channel_type,
        )
        with self._lock:
            self._subs[sub.id] = sub
            self._notifications[sub.id] = []
        return sub

    def list_subscriptions(self, active_only: bool = False) -> List[Subscription]:
        with self._lock:
            subs = list(self._subs.values())
        if active_only:
            subs = [s for s in subs if s.active]
        return subs

    def get_subscription(self, sub_id: str) -> Optional[Subscription]:
        with self._lock:
            return self._subs.get(sub_id)

    def delete_subscription(self, sub_id: str) -> bool:
        with self._lock:
            if sub_id in self._subs:
                del self._subs[sub_id]
                return True
        return False

    def get_notifications(self, sub_id: str) -> List[NotificationRecord]:
        with self._lock:
            return list(self._notifications.get(sub_id, []))

    def dead_letter_queue(self) -> List[NotificationRecord]:
        with self._lock:
            return list(self._dead_letter)

    # -- Matching -----------------------------------------------------------

    def matches(
        self,
        criteria: SubscriptionCriteria,
        active_findings: List[Dict[str, Any]],
        urgency_score: int,
    ) -> List[Dict[str, Any]]:
        """Return the findings that satisfy *criteria* (empty = no match).

        A finding matches when its severity is at least ``min_severity``.
        When ``conditions`` is set, at least one matching finding must be in
        that list.  The overall urgency must also meet ``urgency_threshold``.
        """
        if urgency_score < criteria.urgency_threshold:
            return []

        min_rank = _SEVERITY_RANK[criteria.min_severity]
        matched: List[Dict[str, Any]] = []
        for finding in active_findings:
            severity = _TIER_SEVERITY.get(finding.get("tier", "low"), "info")
            if _SEVERITY_RANK[severity] < min_rank:
                continue
            if criteria.conditions and finding.get("class_name") not in criteria.conditions:
                continue
            matched.append(finding)
        return matched

    # -- Result processing --------------------------------------------------

    def process_result(
        self,
        multi_task_output: Any,
        ecg_id: str,
        *,
        diagnostic_report_ref: Optional[str] = None,
        sync: bool = False,
    ) -> List[NotificationRecord]:
        """Score a result and notify every subscription it matches.

        Args:
            multi_task_output: A ``MultiTaskOutput``-like object/dict.
            ecg_id: Identifier for the ECG.
            diagnostic_report_ref: Optional FHIR DiagnosticReport reference
                to embed in the notification bundle.
            sync: Deliver synchronously (blocks) instead of on a background
                thread.  Useful for tests and small deployments.

        Returns:
            The :class:`NotificationRecord` objects created (one per matched
            subscription), in their initial ``pending`` state.
        """
        prioritizer = self._prioritizer
        if prioritizer is None:
            from aortica.integration.worklist import WorklistPrioritizer

            prioritizer = WorklistPrioritizer()
            self._prioritizer = prioritizer

        worklist = prioritizer.prioritize([multi_task_output], ecg_ids=[ecg_id])
        item = worklist.items[0]

        created: List[NotificationRecord] = []
        for sub in self.list_subscriptions(active_only=True):
            matched = self.matches(
                sub.criteria, item.active_findings, item.urgency_score
            )
            if not matched:
                continue
            payload = self._build_bundle(
                sub, item, matched, diagnostic_report_ref
            )
            record = NotificationRecord(
                id=_new_id("ntf"),
                subscription_id=sub.id,
                status="pending",
                ecg_id=ecg_id,
                urgency_score=item.urgency_score,
                matched_findings=matched,
                payload=payload,
            )
            with self._lock:
                self._notifications.setdefault(sub.id, []).append(record)
            created.append(record)

            if sync:
                self._deliver(sub, record)
            else:
                thread = threading.Thread(
                    target=self._deliver, args=(sub, record), daemon=True
                )
                with self._lock:
                    self._threads.append(thread)
                thread.start()

        return created

    def wait_for_deliveries(self, timeout: Optional[float] = None) -> None:
        """Join outstanding delivery threads (mainly for tests)."""
        with self._lock:
            threads = list(self._threads)
        for t in threads:
            t.join(timeout)

    # -- Delivery -----------------------------------------------------------

    def _deliver(self, sub: Subscription, record: NotificationRecord) -> None:
        """Deliver a notification with bounded exponential-backoff retries."""
        for attempt in range(1, self._max_retries + 1):
            record.attempts = attempt
            try:
                status = self._http_post(
                    sub.webhook_url, record.payload or {}, self._timeout
                )
                if 200 <= status < 300:
                    record.status = "sent"
                    record.delivered_at = _now_iso()
                    record.error = None
                    return
                record.error = f"HTTP {status}"
            except Exception as exc:  # noqa: BLE001 - any failure → retry
                record.error = str(exc)

            if attempt < self._max_retries:
                self._sleep(self._backoff_base * (2 ** (attempt - 1)))

        # All attempts exhausted → dead-letter.
        record.status = "dead-letter"
        with self._lock:
            self._dead_letter.append(record)

    # -- FHIR bundle --------------------------------------------------------

    def _build_bundle(
        self,
        sub: Subscription,
        item: Any,
        matched: List[Dict[str, Any]],
        diagnostic_report_ref: Optional[str],
    ) -> Dict[str, Any]:
        """Build a FHIR ``subscription-notification`` Bundle payload."""
        entries: List[Dict[str, Any]] = [
            {
                "resource": {
                    "resourceType": "Parameters",
                    "parameter": [
                        {"name": "subscription", "valueString": sub.id},
                        {"name": "type", "valueCode": "event-notification"},
                        {"name": "ecg_id", "valueString": item.ecg_id},
                        {"name": "urgency_score", "valueInteger": item.urgency_score},
                        {"name": "timestamp", "valueString": _now_iso()},
                    ],
                }
            }
        ]
        if diagnostic_report_ref:
            entries.append(
                {
                    "resource": {
                        "resourceType": "DiagnosticReport",
                        "id": diagnostic_report_ref,
                    }
                }
            )
        for finding in matched:
            entries.append(
                {
                    "resource": {
                        "resourceType": "Observation",
                        "status": "final",
                        "code": {"text": finding.get("class_name")},
                        "valueQuantity": {"value": finding.get("confidence")},
                        "interpretation": [{"text": finding.get("tier")}],
                    }
                }
            )
        return {
            "resourceType": "Bundle",
            "type": "subscription-notification",
            "timestamp": _now_iso(),
            "entry": entries,
        }
