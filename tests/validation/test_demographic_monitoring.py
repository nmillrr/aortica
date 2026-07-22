"""Tests for demographic-stratified production monitoring (US-130)."""

from __future__ import annotations

from pathlib import Path

from aortica.validation.performance_monitor import (
    PerformanceMonitor,
    SubgroupStatus,
)


def _monitor(tmp_path: Path, **kw) -> PerformanceMonitor:
    return PerformanceMonitor(db_dir=str(tmp_path), **kw)


def _seed(
    monitor: PerformanceMonitor,
    n: int = 80,
    *,
    with_demographics: bool = True,
    female_worse: bool = False,
) -> None:
    for i in range(n):
        sex = "M" if i % 2 else "F"
        age = 30 + (i % 5) * 15
        label = 1 if i % 2 else 0
        # Base case: confident-correct predictions.
        pred = 0.9 if label else 0.1
        if female_worse and sex == "F":
            pred = 0.5  # degrade the female subgroup
        monitor.record_prediction(
            ecg_id=f"e{i}",
            task="rhythm",
            predictions={"AF": pred},
            ground_truth={"AF": label},
            age=age if with_demographics else None,
            sex=sex if with_demographics else None,
        )


# ─────────────────────────────────────────────────────────────────
# record_prediction persists demographics
# ─────────────────────────────────────────────────────────────────


def test_record_prediction_stores_age_sex(tmp_path: Path) -> None:
    m = _monitor(tmp_path)
    m.record_prediction(
        ecg_id="e1", task="rhythm", predictions={"AF": 0.9},
        ground_truth={"AF": 1}, age=55, sex="F",
    )
    row = m._conn.execute(
        "SELECT age, sex FROM monitor_predictions WHERE ecg_id='e1'"
    ).fetchone()
    assert row == (55, "F")


def test_record_prediction_demographics_optional(tmp_path: Path) -> None:
    m = _monitor(tmp_path)
    # Should not raise when age/sex omitted.
    m.record_prediction(ecg_id="e1", task="rhythm", predictions={"AF": 0.9})
    row = m._conn.execute(
        "SELECT age, sex FROM monitor_predictions WHERE ecg_id='e1'"
    ).fetchone()
    assert row == (None, None)


# ─────────────────────────────────────────────────────────────────
# get_subgroup_status
# ─────────────────────────────────────────────────────────────────


def test_subgroup_status_stratifies_by_sex_and_age(tmp_path: Path) -> None:
    m = _monitor(tmp_path)
    _seed(m, 80)
    status = m.get_subgroup_status(min_samples=10)
    assert isinstance(status, SubgroupStatus)
    assert status.has_demographics
    names = {s.subgroup for s in status.subgroups}
    assert "sex_M" in names and "sex_F" in names
    assert any(n.startswith("age_") for n in names)


def test_subgroup_min_sample_guard(tmp_path: Path) -> None:
    m = _monitor(tmp_path)
    _seed(m, 80)
    # A very high threshold excludes every subgroup.
    status = m.get_subgroup_status(min_samples=1000)
    assert status.has_demographics
    assert status.subgroups == []


def test_subgroup_default_min_samples(tmp_path: Path) -> None:
    m = _monitor(tmp_path, subgroup_min_samples=30)
    _seed(m, 80)  # 40 M + 40 F ≥ 30 each
    status = m.get_subgroup_status()
    assert status.min_samples == 30
    sex_groups = {s.subgroup for s in status.subgroups if s.subgroup.startswith("sex_")}
    assert sex_groups == {"sex_M", "sex_F"}


def test_no_demographics_path(tmp_path: Path) -> None:
    m = _monitor(tmp_path)
    _seed(m, 40, with_demographics=False)
    status = m.get_subgroup_status(min_samples=5)
    assert status.has_demographics is False
    assert status.subgroups == []
    assert "No demographic" in status.note


def test_subgroup_equity_drift_flagged(tmp_path: Path) -> None:
    m = _monitor(tmp_path)
    _seed(m, 80, female_worse=True)
    status = m.get_subgroup_status(min_samples=10)
    # The degraded female subgroup should trip an equity-deviation alert.
    assert any(
        a.alert_type == "subgroup_equity_deviation"
        and "sex_F" in a.message
        for a in status.drift_alerts
    )


# ─────────────────────────────────────────────────────────────────
# Quarterly report integration
# ─────────────────────────────────────────────────────────────────


def test_quarterly_report_renders_demographics(tmp_path: Path) -> None:
    from aortica.validation.quarterly_report import generate_quarterly_report

    m = _monitor(tmp_path / "mon")
    _seed(m, 80)
    result = generate_quarterly_report(m, str(tmp_path / "out"), quarter=1, year=2026)
    md = Path(result.markdown_path).read_text()
    assert "Demographic Subgroup Breakdown" in md
    assert "sex_M" in md or "sex_F" in md
    csv_text = Path(result.csv_path).read_text()
    assert "subgroup" in csv_text.splitlines()[0]
    assert "sex_" in csv_text


def test_quarterly_report_states_no_demographics(tmp_path: Path) -> None:
    from aortica.validation.quarterly_report import generate_quarterly_report

    m = _monitor(tmp_path / "mon")
    _seed(m, 40, with_demographics=False)
    result = generate_quarterly_report(m, str(tmp_path / "out"), quarter=1, year=2026)
    md = Path(result.markdown_path).read_text()
    assert "Demographic Subgroup Breakdown" in md
    assert "No demographic" in md


# ─────────────────────────────────────────────────────────────────
# Migration: an old DB without age/sex columns still works
# ─────────────────────────────────────────────────────────────────


def test_migration_adds_columns(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "monitor.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE monitor_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ecg_id TEXT NOT NULL, task TEXT NOT NULL, class_name TEXT NOT NULL,
            prediction REAL NOT NULL, ground_truth INTEGER, timestamp REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    # Opening the monitor should migrate the schema.
    m = _monitor(tmp_path)
    cols = {r[1] for r in m._conn.execute("PRAGMA table_info(monitor_predictions)")}
    assert "age" in cols and "sex" in cols
    m.record_prediction(
        ecg_id="e1", task="rhythm", predictions={"AF": 0.9},
        ground_truth={"AF": 1}, age=60, sex="M",
    )
    assert m.get_subgroup_status(min_samples=1).has_demographics
