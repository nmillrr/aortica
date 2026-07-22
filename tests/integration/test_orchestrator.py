"""Tests for the integration orchestrator (US-125)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from aortica.integration.orchestrator import (
    CHANNELS,
    IntegrationConfig,
    IntegrationOrchestrator,
    load_integration_config,
)


def _processor(critical: bool = False):
    def _proc(payload: Any) -> Dict[str, Any]:
        if critical:
            return {"ischaemia": {"STEMI": 0.95}}
        return {"rhythm": {"normal_sinus_rhythm": 0.99}}
    return _proc


class _RecordingStore:
    def __init__(self) -> None:
        self.stored: List[str] = []

    def store_result(self, ecg_id: str, result: Dict[str, Any]) -> None:
        self.stored.append(ecg_id)


class _FailingWriter:
    def __init__(self, fail_times: int = 1) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def __call__(self, ecg_id: str, result: Dict[str, Any]) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("PACS down")


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_all_channels(self) -> None:
        cfg = IntegrationConfig()
        assert set(cfg.enabled_channels) == set(CHANNELS)

    def test_unknown_channel_raises(self) -> None:
        with pytest.raises(ValueError):
            IntegrationConfig(enabled_channels=["storage", "bogus"])

    def test_load_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "integration.yaml"
        cfg_file.write_text(
            "enabled_channels:\n  - storage\n  - ehr\n"
            "max_retries: 5\nerror_rate_threshold: 0.1\n"
        )
        cfg = load_integration_config(cfg_file)
        assert cfg.enabled_channels == ["storage", "ehr"]
        assert cfg.max_retries == 5
        assert cfg.error_rate_threshold == 0.1


# ─────────────────────────────────────────────────────────────────
# Full workflow
# ─────────────────────────────────────────────────────────────────


class TestWorkflow:
    def test_all_steps_run(self) -> None:
        store = _RecordingStore()
        pacs: List[str] = []
        ehr: List[str] = []
        orch = IntegrationOrchestrator(
            IntegrationConfig(),
            processor=_processor(),
            result_store=store,
            pacs_writer=lambda i, r: pacs.append(i),
            ehr_submitter=lambda i, r: ehr.append(i),
        )
        result = orch.process_ecg("ecg_1", object())
        assert result.ok
        assert store.stored == ["ecg_1"]
        assert pacs == ["ecg_1"]
        assert ehr == ["ecg_1"]
        # worklist/notification skipped (no store/manager configured).
        skipped = {s.channel for s in result.steps if s.skipped}
        assert {"worklist", "notification"} <= skipped

    def test_disabled_channel_skipped(self) -> None:
        pacs: List[str] = []
        orch = IntegrationOrchestrator(
            IntegrationConfig(enabled_channels=["storage"]),
            processor=_processor(),
            result_store=_RecordingStore(),
            pacs_writer=lambda i, r: pacs.append(i),
        )
        result = orch.process_ecg("ecg_1", object())
        assert pacs == []  # pacs not enabled
        pacs_step = next(s for s in result.steps if s.channel == "pacs")
        assert pacs_step.skipped

    def test_processing_failure_short_circuits(self) -> None:
        def _boom(_p: Any) -> Dict[str, Any]:
            raise ValueError("model exploded")

        orch = IntegrationOrchestrator(IntegrationConfig(), processor=_boom)
        result = orch.process_ecg("ecg_1", object())
        assert not result.ok
        assert result.steps[0].channel == "processing"
        assert not result.steps[0].ok


# ─────────────────────────────────────────────────────────────────
# Per-step failure isolation
# ─────────────────────────────────────────────────────────────────


class TestFailureIsolation:
    def test_pacs_failure_does_not_block_ehr(self) -> None:
        ehr: List[str] = []
        orch = IntegrationOrchestrator(
            IntegrationConfig(),
            processor=_processor(),
            result_store=_RecordingStore(),
            pacs_writer=_FailingWriter(fail_times=99),
            ehr_submitter=lambda i, r: ehr.append(i),
        )
        result = orch.process_ecg("ecg_1", object())
        pacs_step = next(s for s in result.steps if s.channel == "pacs")
        ehr_step = next(s for s in result.steps if s.channel == "ehr")
        assert not pacs_step.ok
        assert ehr_step.ok  # EHR still ran despite PACS failure
        assert ehr == ["ecg_1"]

    def test_failure_enqueues_retry(self) -> None:
        orch = IntegrationOrchestrator(
            IntegrationConfig(),
            processor=_processor(),
            pacs_writer=_FailingWriter(fail_times=99),
        )
        orch.process_ecg("ecg_1", object())
        assert orch.status()["queue_depth"] == 1


# ─────────────────────────────────────────────────────────────────
# Retry + dead-letter
# ─────────────────────────────────────────────────────────────────


class TestRetry:
    def test_retry_succeeds_after_transient_failure(self) -> None:
        writer = _FailingWriter(fail_times=1)  # fails first, ok on retry
        orch = IntegrationOrchestrator(
            IntegrationConfig(max_retries=3),
            processor=_processor(),
            pacs_writer=writer,
            sleep=lambda _s: None,
        )
        orch.process_ecg("ecg_1", object())  # enqueues
        succeeded = orch.retry_failed()
        assert succeeded == 1
        assert orch.dead_letter() == []

    def test_dead_letter_after_persistent_failure(self) -> None:
        orch = IntegrationOrchestrator(
            IntegrationConfig(max_retries=2),
            processor=_processor(),
            pacs_writer=_FailingWriter(fail_times=99),
            sleep=lambda _s: None,
        )
        orch.process_ecg("ecg_1", object())
        orch.retry_failed()
        dead = orch.dead_letter()
        assert len(dead) == 1
        assert dead[0]["channel"] == "pacs"


# ─────────────────────────────────────────────────────────────────
# Status + health
# ─────────────────────────────────────────────────────────────────


class TestStatus:
    def test_status_counts(self) -> None:
        orch = IntegrationOrchestrator(
            IntegrationConfig(),
            processor=_processor(),
            result_store=_RecordingStore(),
        )
        orch.process_ecg("ecg_1", object())
        status = orch.status()
        assert status["processed"] == 1
        assert status["channels"]["storage"]["success"] == 1

    def test_health_alert_on_high_error_rate(self) -> None:
        orch = IntegrationOrchestrator(
            IntegrationConfig(error_rate_threshold=0.05),
            processor=_processor(),
            pacs_writer=_FailingWriter(fail_times=99),
        )
        for i in range(5):
            orch.process_ecg(f"ecg_{i}", object())
        alerts = orch.health_alerts()
        assert any(a["channel"] == "pacs" for a in alerts)

    def test_worklist_and_notification_integration(self) -> None:
        from aortica.integration.fhir_subscription import (
            SubscriptionCriteria,
            SubscriptionManager,
        )
        from aortica.integration.worklist_store import WorklistStore

        wl = WorklistStore()
        mgr = SubscriptionManager(
            http_post=lambda u, p, t: 200, sleep=lambda _s: None
        )
        mgr.create_subscription(
            "https://ehr/hook", SubscriptionCriteria(min_severity="critical")
        )
        orch = IntegrationOrchestrator(
            IntegrationConfig(),
            processor=_processor(critical=True),
            worklist_store=wl,
            subscription_manager=mgr,
        )
        orch.process_ecg("ecg_crit", object())
        assert wl.get("ecg_crit") is not None
        # Critical result matched the subscription → a notification was sent.
        subs = mgr.list_subscriptions()
        assert len(mgr.get_notifications(subs[0].id)) == 1
