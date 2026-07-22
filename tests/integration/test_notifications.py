"""Tests for the urgent-finding notification pipeline (US-126)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from aortica.integration.notifications import (
    NotificationRules,
    UrgentFindingNotifier,
    load_notification_rules,
)


def _critical() -> Dict[str, Any]:
    return {"ischaemia": {"STEMI": 0.95}}


def _routine() -> Dict[str, Any]:
    return {"rhythm": {"normal_sinus_rhythm": 0.99}}


class _RecordingSender:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, payload: Dict[str, Any]) -> bool:
        self.calls.append(payload)
        return self.ok


def _notifier(rules: NotificationRules, senders=None) -> UrgentFindingNotifier:
    return UrgentFindingNotifier(rules, db_path=":memory:", senders=senders or {})


# ─────────────────────────────────────────────────────────────────
# Rules
# ─────────────────────────────────────────────────────────────────


class TestRules:
    def test_unknown_channel_raises(self) -> None:
        with pytest.raises(ValueError):
            NotificationRules(channels=["webhook", "carrier_pigeon"])

    def test_load_rules(self, tmp_path: Path) -> None:
        f = tmp_path / "notification_rules.yaml"
        f.write_text(
            "conditions:\n  - STEMI\n  - VT\n"
            "min_confidence: 0.7\nurgency_threshold: 60\n"
            "dedup_window_hours: 12\nchannels:\n  - webhook\n  - email\n"
        )
        rules = load_notification_rules(f)
        assert rules.conditions == ["STEMI", "VT"]
        assert rules.min_confidence == 0.7
        assert rules.channels == ["webhook", "email"]


# ─────────────────────────────────────────────────────────────────
# Trigger logic
# ─────────────────────────────────────────────────────────────────


class TestTrigger:
    def test_critical_triggers(self) -> None:
        sender = _RecordingSender()
        notifier = _notifier(
            NotificationRules(channels=["webhook"]), {"webhook": sender}
        )
        records = notifier.notify(_critical(), "ecg_1")
        assert len(records) == 1
        assert records[0].finding == "STEMI"
        assert len(sender.calls) == 1

    def test_routine_no_trigger(self) -> None:
        sender = _RecordingSender()
        notifier = _notifier(
            NotificationRules(urgency_threshold=50, channels=["webhook"]),
            {"webhook": sender},
        )
        assert notifier.notify(_routine(), "ecg_1") == []
        assert sender.calls == []

    def test_condition_filter(self) -> None:
        sender = _RecordingSender()
        notifier = _notifier(
            NotificationRules(conditions=["VT", "VF"], channels=["webhook"]),
            {"webhook": sender},
        )
        # STEMI is critical but not in the allow-list.
        assert notifier.notify(_critical(), "ecg_1") == []

    def test_confidence_filter(self) -> None:
        sender = _RecordingSender()
        notifier = _notifier(
            NotificationRules(min_confidence=0.99, channels=["webhook"]),
            {"webhook": sender},
        )
        assert notifier.notify(_critical(), "ecg_1") == []


# ─────────────────────────────────────────────────────────────────
# Multi-channel delivery
# ─────────────────────────────────────────────────────────────────


class TestMultiChannel:
    def test_delivers_to_all_channels(self) -> None:
        webhook, email = _RecordingSender(), _RecordingSender()
        notifier = _notifier(
            NotificationRules(channels=["webhook", "email"]),
            {"webhook": webhook, "email": email},
        )
        records = notifier.notify(_critical(), "ecg_1")
        assert {r.channel for r in records} == {"webhook", "email"}
        assert len(webhook.calls) == 1
        assert len(email.calls) == 1

    def test_failed_channel_tracked(self) -> None:
        notifier = _notifier(
            NotificationRules(channels=["webhook"]),
            {"webhook": _RecordingSender(ok=False)},
        )
        records = notifier.notify(_critical(), "ecg_1")
        assert records[0].status == "failed"

    def test_exception_in_sender_tracked(self) -> None:
        def _boom(_p: Dict[str, Any]) -> bool:
            raise RuntimeError("smtp down")

        notifier = _notifier(
            NotificationRules(channels=["email"]), {"email": _boom}
        )
        records = notifier.notify(_critical(), "ecg_1")
        assert records[0].status == "failed"
        assert "smtp down" in (records[0].error or "")


# ─────────────────────────────────────────────────────────────────
# De-duplication
# ─────────────────────────────────────────────────────────────────


class TestDedup:
    def test_dedup_same_patient_finding(self) -> None:
        sender = _RecordingSender()
        notifier = _notifier(
            NotificationRules(dedup_window_hours=24, channels=["webhook"]),
            {"webhook": sender},
        )
        notifier.notify(_critical(), "ecg_1", patient_id="P1")
        # Second STEMI for the same patient within the window is suppressed.
        second = notifier.notify(_critical(), "ecg_2", patient_id="P1")
        assert second == []
        assert len(sender.calls) == 1

    def test_no_dedup_different_patient(self) -> None:
        sender = _RecordingSender()
        notifier = _notifier(
            NotificationRules(channels=["webhook"]), {"webhook": sender}
        )
        notifier.notify(_critical(), "ecg_1", patient_id="P1")
        notifier.notify(_critical(), "ecg_2", patient_id="P2")
        assert len(sender.calls) == 2

    def test_no_dedup_without_patient(self) -> None:
        sender = _RecordingSender()
        notifier = _notifier(
            NotificationRules(channels=["webhook"]), {"webhook": sender}
        )
        notifier.notify(_critical(), "ecg_1")
        notifier.notify(_critical(), "ecg_2")
        assert len(sender.calls) == 2


# ─────────────────────────────────────────────────────────────────
# History + payload builders
# ─────────────────────────────────────────────────────────────────


class TestHistory:
    def test_history_records(self, tmp_path: Path) -> None:
        notifier = UrgentFindingNotifier(
            NotificationRules(channels=["webhook"]),
            db_path=str(tmp_path / "n.db"),
            senders={"webhook": _RecordingSender()},
        )
        notifier.notify(_critical(), "ecg_1", patient_id="P1")
        history = notifier.history()
        assert len(history) == 1
        assert history[0].ecg_id == "ecg_1"
        assert notifier.history(patient_id="P1")[0].patient_id == "P1"

    def test_communication_request_builder(self) -> None:
        payload = {
            "patient_id": "P1", "finding": "STEMI", "confidence": 0.95,
            "urgency_score": 90, "recommended_action": "Immediate review",
            "result_url": "https://x/ecg_1",
        }
        cr = UrgentFindingNotifier.build_communication_request(payload)
        assert cr["resourceType"] == "CommunicationRequest"
        assert cr["priority"] == "urgent"
        assert "STEMI" in cr["payload"][0]["contentString"]
