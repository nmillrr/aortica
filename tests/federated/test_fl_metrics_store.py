"""Tests for aortica.federated.fl_metrics_store — SQLite persistence (US-113)."""

from __future__ import annotations

import time

import pytest

from aortica.federated.fl_metrics_store import (
    CampaignStatus,
    ConvergenceIndicators,
    FLMetricsStore,
    RoundRecord,
    SiteRecord,
)


# ---------------------------------------------------------------------------
# Campaign lifecycle
# ---------------------------------------------------------------------------


class TestCampaignLifecycle:
    """Test campaign start / complete / fail / status."""

    def test_default_status_is_idle(self) -> None:
        store = FLMetricsStore()
        status = store.get_campaign_status()
        assert status.status == "idle"
        assert status.current_round == 0

    def test_start_campaign(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(
            name="test-run",
            total_rounds=20,
            strategy="fedprox",
            epsilon_budget=2.0,
        )
        status = store.get_campaign_status()
        assert status.campaign_name == "test-run"
        assert status.total_rounds == 20
        assert status.strategy == "fedprox"
        assert status.status == "running"
        assert status.start_timestamp > 0
        assert status.elapsed_seconds >= 0

    def test_complete_campaign(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(name="c1", total_rounds=5)
        store.complete_campaign()
        assert store.get_campaign_status().status == "completed"

    def test_fail_campaign(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(name="c2", total_rounds=5)
        store.fail_campaign()
        assert store.get_campaign_status().status == "failed"

    def test_restart_resets(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(name="first", total_rounds=5)
        store.complete_campaign()
        store.start_campaign(name="second", total_rounds=10)
        status = store.get_campaign_status()
        assert status.campaign_name == "second"
        assert status.total_rounds == 10
        assert status.status == "running"

    def test_to_dict(self) -> None:
        status = CampaignStatus(campaign_name="x", current_round=3)
        d = status.to_dict()
        assert d["campaign_name"] == "x"
        assert d["current_round"] == 3


# ---------------------------------------------------------------------------
# Round recording
# ---------------------------------------------------------------------------


class TestRoundRecording:
    """Test persisting and retrieving per-round metrics."""

    def test_record_and_retrieve(self) -> None:
        store = FLMetricsStore()
        store.record_round(
            round_number=1,
            loss=0.5,
            metrics={"rhythm_f1": 0.87, "structural_f1": 0.82},
            num_clients=3,
            gradient_norm=1.2,
        )
        store.record_round(round_number=2, loss=0.4, num_clients=4)

        rounds = store.get_rounds()
        assert len(rounds) == 2
        assert rounds[0].round_number == 1
        assert rounds[0].loss == 0.5
        assert rounds[0].metrics["rhythm_f1"] == 0.87
        assert rounds[0].num_clients == 3
        assert rounds[0].gradient_norm == 1.2
        assert rounds[1].round_number == 2
        assert rounds[1].loss == 0.4

    def test_empty_rounds(self) -> None:
        store = FLMetricsStore()
        assert store.get_rounds() == []

    def test_replace_round(self) -> None:
        store = FLMetricsStore()
        store.record_round(round_number=1, loss=0.5)
        store.record_round(round_number=1, loss=0.3)  # Replace
        rounds = store.get_rounds()
        assert len(rounds) == 1
        assert rounds[0].loss == 0.3

    def test_current_round_in_status(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(total_rounds=10)
        store.record_round(round_number=1, loss=0.5)
        store.record_round(round_number=2, loss=0.4)
        store.record_round(round_number=3, loss=0.35)
        status = store.get_campaign_status()
        assert status.current_round == 3

    def test_round_to_dict(self) -> None:
        r = RoundRecord(round_number=5, loss=0.3, num_clients=2)
        d = r.to_dict()
        assert d["round_number"] == 5
        assert d["loss"] == 0.3


# ---------------------------------------------------------------------------
# Site participation
# ---------------------------------------------------------------------------


class TestSiteParticipation:
    """Test per-site participation tracking."""

    def test_insert_new_site(self) -> None:
        store = FLMetricsStore()
        store.update_site(
            "site-A",
            status="online",
            samples_contributed=500,
            local_training_time_ms=1200.0,
            epsilon_spent=0.1,
        )
        sites = store.get_sites()
        assert len(sites) == 1
        assert sites[0].site_id == "site-A"
        assert sites[0].status == "online"
        assert sites[0].samples_contributed == 500
        assert sites[0].epsilon_spent == 0.1

    def test_update_existing_site(self) -> None:
        store = FLMetricsStore()
        store.update_site("site-B", status="online", samples_contributed=100)
        store.update_site("site-B", status="offline", epsilon_spent=0.5)
        sites = store.get_sites()
        assert len(sites) == 1
        assert sites[0].status == "offline"
        assert sites[0].epsilon_spent == 0.5
        # samples_contributed unchanged from insert
        assert sites[0].samples_contributed == 100

    def test_multiple_sites(self) -> None:
        store = FLMetricsStore()
        store.update_site("alpha", status="online")
        store.update_site("beta", status="online")
        store.update_site("gamma", status="offline")
        sites = store.get_sites()
        assert len(sites) == 3
        ids = [s.site_id for s in sites]
        assert "alpha" in ids
        assert "beta" in ids
        assert "gamma" in ids

    def test_empty_sites(self) -> None:
        store = FLMetricsStore()
        assert store.get_sites() == []

    def test_site_to_dict(self) -> None:
        s = SiteRecord(site_id="x", status="online")
        d = s.to_dict()
        assert d["site_id"] == "x"


# ---------------------------------------------------------------------------
# Privacy budget
# ---------------------------------------------------------------------------


class TestPrivacyBudget:
    """Test epsilon budget retrieval."""

    def test_default_budget(self) -> None:
        store = FLMetricsStore()
        assert store.get_epsilon_budget() == 1.0

    def test_custom_budget(self) -> None:
        store = FLMetricsStore()
        store.start_campaign(epsilon_budget=5.0)
        assert store.get_epsilon_budget() == 5.0


# ---------------------------------------------------------------------------
# Convergence indicators
# ---------------------------------------------------------------------------


class TestConvergence:
    """Test plateau detection and convergence indicators."""

    def test_no_data(self) -> None:
        store = FLMetricsStore()
        ci = store.get_convergence_indicators()
        assert ci.gradient_norms == []
        assert ci.plateau_detected is False
        assert ci.early_stop_recommended is False

    def test_plateau_detection(self) -> None:
        store = FLMetricsStore()
        # Create 7 rounds with plateaued loss
        for i in range(1, 8):
            store.record_round(round_number=i, loss=0.5, gradient_norm=0.1)

        ci = store.get_convergence_indicators(plateau_window=5)
        assert ci.plateau_detected is True
        assert ci.early_stop_recommended is True
        assert len(ci.gradient_norms) == 7

    def test_no_plateau_when_improving(self) -> None:
        store = FLMetricsStore()
        for i in range(1, 8):
            store.record_round(round_number=i, loss=1.0 - i * 0.1, gradient_norm=1.0)

        ci = store.get_convergence_indicators(plateau_window=5)
        assert ci.plateau_detected is False

    def test_convergence_to_dict(self) -> None:
        ci = ConvergenceIndicators(plateau_detected=True)
        d = ci.to_dict()
        assert d["plateau_detected"] is True


# ---------------------------------------------------------------------------
# Close / cleanup
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    """Test close and repr."""

    def test_close(self) -> None:
        store = FLMetricsStore()
        store.close()
        # After close, operations should raise
        with pytest.raises(Exception):
            store.get_rounds()

    def test_repr(self) -> None:
        store = FLMetricsStore()
        assert "FLMetricsStore" in repr(store)
