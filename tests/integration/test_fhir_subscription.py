"""Tests for aortica.integration.fhir_subscription (US-117)."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from aortica.integration.fhir_subscription import (
    NotificationRecord,
    Subscription,
    SubscriptionCriteria,
    SubscriptionManager,
)

# ---------------------------------------------------------------------------
# Synthetic multi-task outputs (dict-of-dicts form accepted by the worklist)
# ---------------------------------------------------------------------------


def _critical_result() -> Dict[str, Dict[str, float]]:
    # STEMI is a critical ischaemia finding.
    return {
        "rhythm": {"normal_sinus_rhythm": 0.1},
        "structural": {},
        "ischaemia": {"STEMI": 0.95},
        "risk": {},
    }


def _moderate_result() -> Dict[str, Dict[str, float]]:
    return {
        "rhythm": {"PAC": 0.9},
        "structural": {},
        "ischaemia": {},
        "risk": {},
    }


def _normal_result() -> Dict[str, Dict[str, float]]:
    return {
        "rhythm": {"normal_sinus_rhythm": 0.99},
        "structural": {},
        "ischaemia": {},
        "risk": {},
    }


class _RecordingPoster:
    """Records webhook POSTs and returns a scripted status sequence."""

    def __init__(self, statuses: List[Any]) -> None:
        self.statuses = statuses
        self.calls: List[Dict[str, Any]] = []
        self._i = 0
        self._lock = threading.Lock()

    def __call__(self, url: str, payload: Dict[str, Any], timeout: float) -> int:
        with self._lock:
            self.calls.append({"url": url, "payload": payload})
            status = self.statuses[min(self._i, len(self.statuses) - 1)]
            self._i += 1
        if isinstance(status, Exception):
            raise status
        return int(status)


def _manager(poster: _RecordingPoster, **kwargs: Any) -> SubscriptionManager:
    return SubscriptionManager(
        http_post=poster,
        sleep=lambda _s: None,  # no real backoff delay in tests
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────


class TestCrud:
    def test_create_and_get(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        sub = mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        assert isinstance(sub, Subscription)
        assert mgr.get_subscription(sub.id) is sub

    def test_list(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        mgr.create_subscription("https://a", SubscriptionCriteria())
        mgr.create_subscription("https://b", SubscriptionCriteria())
        assert len(mgr.list_subscriptions()) == 2

    def test_delete(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        sub = mgr.create_subscription("https://a", SubscriptionCriteria())
        assert mgr.delete_subscription(sub.id) is True
        assert mgr.get_subscription(sub.id) is None
        assert mgr.delete_subscription(sub.id) is False

    def test_rejects_non_rest_hook(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        with pytest.raises(ValueError):
            mgr.create_subscription("https://a", SubscriptionCriteria(), "websocket")

    def test_criteria_validation(self) -> None:
        with pytest.raises(ValueError):
            SubscriptionCriteria(min_severity="urgent")
        with pytest.raises(ValueError):
            SubscriptionCriteria(urgency_threshold=200)

    def test_subscription_to_fhir(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        sub = mgr.create_subscription(
            "https://ehr/hook",
            SubscriptionCriteria(min_severity="critical", conditions=["STEMI"]),
        )
        fhir = sub.to_fhir()
        assert fhir["resourceType"] == "Subscription"
        assert fhir["channel"]["endpoint"] == "https://ehr/hook"


# ─────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────


class TestMatching:
    def test_critical_matches_critical_criteria(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        findings = [{"class_name": "STEMI", "confidence": 0.95, "tier": "critical"}]
        matched = mgr.matches(
            SubscriptionCriteria(min_severity="critical"), findings, 90
        )
        assert len(matched) == 1

    def test_moderate_does_not_match_critical_criteria(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        findings = [{"class_name": "PAC", "confidence": 0.9, "tier": "moderate"}]
        matched = mgr.matches(
            SubscriptionCriteria(min_severity="critical"), findings, 50
        )
        assert matched == []

    def test_warning_criteria_matches_moderate(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        findings = [{"class_name": "PAC", "confidence": 0.9, "tier": "moderate"}]
        matched = mgr.matches(
            SubscriptionCriteria(min_severity="warning"), findings, 50
        )
        assert len(matched) == 1

    def test_condition_filter(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        findings = [{"class_name": "STEMI", "confidence": 0.95, "tier": "critical"}]
        assert mgr.matches(
            SubscriptionCriteria(min_severity="critical", conditions=["VT", "VF"]),
            findings, 90,
        ) == []
        assert len(mgr.matches(
            SubscriptionCriteria(min_severity="critical", conditions=["STEMI"]),
            findings, 90,
        )) == 1

    def test_urgency_threshold(self) -> None:
        mgr = _manager(_RecordingPoster([200]))
        findings = [{"class_name": "STEMI", "confidence": 0.95, "tier": "critical"}]
        assert mgr.matches(
            SubscriptionCriteria(min_severity="critical", urgency_threshold=95),
            findings, 90,
        ) == []


# ─────────────────────────────────────────────────────────────────
# Delivery
# ─────────────────────────────────────────────────────────────────


class TestDelivery:
    def test_critical_result_delivers(self) -> None:
        poster = _RecordingPoster([200])
        mgr = _manager(poster)
        mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        created = mgr.process_result(_critical_result(), "ecg_1", sync=True)
        assert len(created) == 1
        assert created[0].status == "sent"
        assert len(poster.calls) == 1
        # Payload is a FHIR subscription-notification bundle.
        payload = poster.calls[0]["payload"]
        assert payload["type"] == "subscription-notification"

    def test_normal_result_no_delivery(self) -> None:
        poster = _RecordingPoster([200])
        mgr = _manager(poster)
        mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        created = mgr.process_result(_normal_result(), "ecg_2", sync=True)
        assert created == []
        assert poster.calls == []

    def test_moderate_below_critical_no_delivery(self) -> None:
        poster = _RecordingPoster([200])
        mgr = _manager(poster)
        mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        created = mgr.process_result(_moderate_result(), "ecg_3", sync=True)
        assert created == []

    def test_diagnostic_report_ref_in_payload(self) -> None:
        poster = _RecordingPoster([200])
        mgr = _manager(poster)
        mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        mgr.process_result(
            _critical_result(), "ecg_1",
            diagnostic_report_ref="DiagnosticReport/abc", sync=True,
        )
        payload = poster.calls[0]["payload"]
        resource_types = [e["resource"]["resourceType"] for e in payload["entry"]]
        assert "DiagnosticReport" in resource_types


# ─────────────────────────────────────────────────────────────────
# Retry / dead-letter
# ─────────────────────────────────────────────────────────────────


class TestRetry:
    def test_retries_then_succeeds(self) -> None:
        # Fail twice, then succeed on the third attempt.
        poster = _RecordingPoster([500, 500, 200])
        mgr = _manager(poster, max_retries=3)
        mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        created = mgr.process_result(_critical_result(), "ecg_1", sync=True)
        assert created[0].status == "sent"
        assert created[0].attempts == 3
        assert len(poster.calls) == 3

    def test_dead_letter_after_exhausting_retries(self) -> None:
        poster = _RecordingPoster([500, 500, 500])
        mgr = _manager(poster, max_retries=3)
        sub = mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        created = mgr.process_result(_critical_result(), "ecg_1", sync=True)
        assert created[0].status == "dead-letter"
        assert created[0].attempts == 3
        assert len(mgr.dead_letter_queue()) == 1
        # History records the failed delivery.
        history = mgr.get_notifications(sub.id)
        assert history[0].status == "dead-letter"

    def test_exception_counts_as_failure(self) -> None:
        poster = _RecordingPoster([ConnectionError("boom"), 200])
        mgr = _manager(poster, max_retries=3)
        mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        created = mgr.process_result(_critical_result(), "ecg_1", sync=True)
        assert created[0].status == "sent"
        assert created[0].attempts == 2


# ─────────────────────────────────────────────────────────────────
# Notification history
# ─────────────────────────────────────────────────────────────────


class TestHistory:
    def test_history_records_delivery(self) -> None:
        poster = _RecordingPoster([200])
        mgr = _manager(poster)
        sub = mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        mgr.process_result(_critical_result(), "ecg_1", sync=True)
        history = mgr.get_notifications(sub.id)
        assert len(history) == 1
        assert isinstance(history[0], NotificationRecord)
        assert history[0].ecg_id == "ecg_1"
        assert history[0].status == "sent"

    def test_async_delivery_completes(self) -> None:
        poster = _RecordingPoster([200])
        mgr = _manager(poster)
        sub = mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        mgr.process_result(_critical_result(), "ecg_1", sync=False)
        mgr.wait_for_deliveries(timeout=5)
        history = mgr.get_notifications(sub.id)
        assert history[0].status == "sent"
