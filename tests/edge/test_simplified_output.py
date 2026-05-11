"""Tests for aortica.edge.simplified_output — CHW-facing simplified output."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aortica.edge.simplified_output import (
    TIER_LOW,
    TIER_REFER,
    TIER_URGENT,
    VALID_TIERS,
    KeyFinding,
    SimplifiedReport,
    TierThresholds,
    _DEFAULT_LOCALE,
    _ISCHAEMIA_CLASSES,
    _REFER_CONDITIONS,
    _RHYTHM_CLASSES,
    _RISK_OUTPUTS,
    _RISK_THRESHOLDS,
    _STRUCTURAL_CLASSES,
    _URGENT_CONDITIONS,
    _extract_predictions,
    _get_locale,
    load_locale,
    simplify_output,
)


# ── Constants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_tier_names(self) -> None:
        assert TIER_LOW == "low"
        assert TIER_REFER == "refer"
        assert TIER_URGENT == "urgent"

    def test_valid_tiers_ordered(self) -> None:
        assert VALID_TIERS == ["low", "refer", "urgent"]

    def test_rhythm_classes_count(self) -> None:
        assert len(_RHYTHM_CLASSES) == 28

    def test_structural_classes_count(self) -> None:
        assert len(_STRUCTURAL_CLASSES) == 15

    def test_ischaemia_classes_count(self) -> None:
        assert len(_ISCHAEMIA_CLASSES) == 15

    def test_risk_outputs_count(self) -> None:
        assert len(_RISK_OUTPUTS) == 3

    def test_urgent_conditions_nonempty(self) -> None:
        assert len(_URGENT_CONDITIONS) > 0

    def test_refer_conditions_nonempty(self) -> None:
        assert len(_REFER_CONDITIONS) > 0

    def test_risk_thresholds_keys(self) -> None:
        for name in _RISK_OUTPUTS:
            assert name in _RISK_THRESHOLDS

    def test_default_locale_keys(self) -> None:
        assert "tier_labels" in _DEFAULT_LOCALE
        assert "summaries" in _DEFAULT_LOCALE
        assert "actions" in _DEFAULT_LOCALE


# ── Dataclasses ────────────────────────────────────────────────────────────

class TestTierThresholds:
    def test_defaults(self) -> None:
        t = TierThresholds()
        assert len(t.urgent_conditions) > 0
        assert len(t.refer_conditions) > 0
        assert len(t.risk_thresholds) == 3

    def test_custom(self) -> None:
        t = TierThresholds(
            urgent_conditions={"VF": 0.20},
            refer_conditions={"AF": 0.30},
            risk_thresholds={},
        )
        assert t.urgent_conditions == {"VF": 0.20}
        assert t.refer_conditions == {"AF": 0.30}


class TestKeyFinding:
    def test_construction(self) -> None:
        f = KeyFinding("VT", 0.85, "rhythm", "urgent")
        assert f.class_name == "VT"
        assert f.confidence == 0.85
        assert f.task == "rhythm"
        assert f.tier_contribution == "urgent"


class TestSimplifiedReport:
    def test_construction(self) -> None:
        r = SimplifiedReport(
            tier="low",
            tier_label="Low risk",
            summary="All clear.",
            actions=["Continue monitoring"],
        )
        assert r.tier == "low"
        assert r.locale == "en"

    def test_to_dict(self) -> None:
        finding = KeyFinding("VT", 0.9, "rhythm", "urgent")
        r = SimplifiedReport(
            tier="urgent",
            tier_label="Urgent",
            summary="Danger.",
            actions=["Act now"],
            key_findings=[finding],
            locale="es",
        )
        d = r.to_dict()
        assert d["tier"] == "urgent"
        assert d["locale"] == "es"
        assert len(d["key_findings"]) == 1
        assert d["key_findings"][0]["class_name"] == "VT"

    def test_to_dict_empty_findings(self) -> None:
        r = SimplifiedReport("low", "Low", "Ok", [])
        d = r.to_dict()
        assert d["key_findings"] == []


# ── Locale ─────────────────────────────────────────────────────────────────

class TestLocale:
    def test_default_locale(self) -> None:
        loc = _get_locale(None)
        assert "tier_labels" in loc

    def test_load_locale_file(self, tmp_path: Path) -> None:
        locale_data = {
            "tier_labels": {"low": "Bajo riesgo", "refer": "Referir", "urgent": "Urgente"},
            "summaries": {"low": "Todo bien.", "refer": "Consultar.", "urgent": "Emergencia."},
            "actions": {"low": ["Nada"], "refer": ["Consultar"], "urgent": ["Actuar"]},
        }
        locale_file = tmp_path / "es.json"
        locale_file.write_text(json.dumps(locale_data), encoding="utf-8")
        loaded = load_locale(locale_file)
        assert loaded["tier_labels"]["low"] == "Bajo riesgo"

    def test_load_locale_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_locale("/nonexistent/locale.json")

    def test_get_locale_with_file(self, tmp_path: Path) -> None:
        locale_file = tmp_path / "fr.json"
        locale_file.write_text(json.dumps(_DEFAULT_LOCALE), encoding="utf-8")
        loc = _get_locale(str(locale_file))
        assert "tier_labels" in loc

    def test_get_locale_unknown_code(self) -> None:
        loc = _get_locale("nonexistent_code")
        assert loc == _DEFAULT_LOCALE


# ── Prediction extraction ──────────────────────────────────────────────────

class TestExtractPredictions:
    def test_dict_with_lists(self) -> None:
        output = {
            "rhythm": [0.1] * 28,
            "structural": [0.2] * 15,
            "ischaemia": [0.3] * 15,
            "risk": [0.4] * 3,
        }
        preds = _extract_predictions(output)
        assert len(preds["rhythm"]) == 28
        assert preds["ischaemia"]["STEMI"] == pytest.approx(0.3)
        assert preds["risk"]["mortality_1y"] == pytest.approx(0.4)

    def test_dict_with_nested_dict(self) -> None:
        output = {
            "rhythm": {"AF": 0.9, "VT": 0.1},
        }
        preds = _extract_predictions(output)
        assert preds["rhythm"]["AF"] == pytest.approx(0.9)

    def test_dict_with_none_task(self) -> None:
        output = {"rhythm": None, "risk": [0.1, 0.2, 0.3]}
        preds = _extract_predictions(output)
        assert "rhythm" not in preds
        assert "risk" in preds

    def test_dict_wrong_length_skipped(self) -> None:
        output = {"rhythm": [0.1, 0.2]}  # wrong length
        preds = _extract_predictions(output)
        assert "rhythm" not in preds

    def test_empty_dict(self) -> None:
        preds = _extract_predictions({})
        assert preds == {}


# ── Core simplify_output ───────────────────────────────────────────────────

class TestSimplifyOutput:
    def _make_output(
        self,
        rhythm: list[float] | None = None,
        structural: list[float] | None = None,
        ischaemia: list[float] | None = None,
        risk: list[float] | None = None,
    ) -> dict[str, list[float] | None]:
        return {
            "rhythm": rhythm or [0.0] * 28,
            "structural": structural or [0.0] * 15,
            "ischaemia": ischaemia or [0.0] * 15,
            "risk": risk or [0.0] * 3,
        }

    def test_all_low(self) -> None:
        output = self._make_output()
        report = simplify_output(output)
        assert report.tier == TIER_LOW
        assert len(report.key_findings) == 0

    def test_returns_simplified_report(self) -> None:
        report = simplify_output(self._make_output())
        assert isinstance(report, SimplifiedReport)

    def test_tier_label_present(self) -> None:
        report = simplify_output(self._make_output())
        assert len(report.tier_label) > 0

    def test_summary_present(self) -> None:
        report = simplify_output(self._make_output())
        assert len(report.summary) > 0

    def test_actions_present(self) -> None:
        report = simplify_output(self._make_output())
        assert len(report.actions) > 0

    def test_urgent_from_vt(self) -> None:
        rhythm = [0.0] * 28
        vt_idx = _RHYTHM_CLASSES.index("VT")
        rhythm[vt_idx] = 0.95
        report = simplify_output(self._make_output(rhythm=rhythm))
        assert report.tier == TIER_URGENT
        assert any(f.class_name == "VT" for f in report.key_findings)

    def test_urgent_from_stemi(self) -> None:
        ischaemia = [0.0] * 15
        stemi_idx = _ISCHAEMIA_CLASSES.index("STEMI")
        ischaemia[stemi_idx] = 0.80
        report = simplify_output(self._make_output(ischaemia=ischaemia))
        assert report.tier == TIER_URGENT

    def test_urgent_from_vf(self) -> None:
        rhythm = [0.0] * 28
        vf_idx = _RHYTHM_CLASSES.index("VF")
        rhythm[vf_idx] = 0.50
        report = simplify_output(self._make_output(rhythm=rhythm))
        assert report.tier == TIER_URGENT

    def test_refer_from_af(self) -> None:
        rhythm = [0.0] * 28
        af_idx = _RHYTHM_CLASSES.index("AF")
        rhythm[af_idx] = 0.70
        report = simplify_output(self._make_output(rhythm=rhythm))
        assert report.tier == TIER_REFER
        assert any(f.class_name == "AF" for f in report.key_findings)

    def test_refer_from_lvsd(self) -> None:
        structural = [0.0] * 15
        lvsd_idx = _STRUCTURAL_CLASSES.index("LVSD")
        structural[lvsd_idx] = 0.60
        report = simplify_output(self._make_output(structural=structural))
        assert report.tier == TIER_REFER

    def test_urgent_overrides_refer(self) -> None:
        rhythm = [0.0] * 28
        af_idx = _RHYTHM_CLASSES.index("AF")
        vf_idx = _RHYTHM_CLASSES.index("VF")
        rhythm[af_idx] = 0.80  # refer
        rhythm[vf_idx] = 0.50  # urgent
        report = simplify_output(self._make_output(rhythm=rhythm))
        assert report.tier == TIER_URGENT
        # Both findings should be present
        names = {f.class_name for f in report.key_findings}
        assert "AF" in names
        assert "VF" in names

    def test_below_threshold_stays_low(self) -> None:
        rhythm = [0.0] * 28
        vt_idx = _RHYTHM_CLASSES.index("VT")
        rhythm[vt_idx] = 0.30  # below 0.40 urgent threshold
        report = simplify_output(self._make_output(rhythm=rhythm))
        assert report.tier == TIER_LOW

    def test_risk_urgent(self) -> None:
        risk = [0.0, 0.0, 0.0]
        risk[0] = 0.85  # mortality_1y > 0.70 urgent
        report = simplify_output(self._make_output(risk=risk))
        assert report.tier == TIER_URGENT
        assert any(f.class_name == "mortality_1y" for f in report.key_findings)

    def test_risk_refer(self) -> None:
        risk = [0.0, 0.0, 0.0]
        risk[1] = 0.50  # hf_hosp_12m >= 0.40 refer
        report = simplify_output(self._make_output(risk=risk))
        assert report.tier == TIER_REFER

    def test_findings_sorted_urgent_first(self) -> None:
        rhythm = [0.0] * 28
        af_idx = _RHYTHM_CLASSES.index("AF")
        vt_idx = _RHYTHM_CLASSES.index("VT")
        rhythm[af_idx] = 0.80  # refer
        rhythm[vt_idx] = 0.50  # urgent
        report = simplify_output(self._make_output(rhythm=rhythm))
        if len(report.key_findings) >= 2:
            assert report.key_findings[0].tier_contribution == TIER_URGENT

    def test_custom_thresholds(self) -> None:
        thresholds = TierThresholds(
            urgent_conditions={"normal_sinus_rhythm": 0.50},
            refer_conditions={},
            risk_thresholds={},
        )
        rhythm = [0.0] * 28
        nsr_idx = _RHYTHM_CLASSES.index("normal_sinus_rhythm")
        rhythm[nsr_idx] = 0.90
        report = simplify_output(
            self._make_output(rhythm=rhythm),
            thresholds=thresholds,
        )
        assert report.tier == TIER_URGENT

    def test_locale_file(self, tmp_path: Path) -> None:
        locale_data = {
            "tier_labels": {"low": "Bajo", "refer": "Referir", "urgent": "Urgente"},
            "summaries": {"low": "Bien.", "refer": "Consultar.", "urgent": "Emergencia."},
            "actions": {"low": ["Nada"], "refer": ["Ver médico"], "urgent": ["Hospital"]},
        }
        locale_file = tmp_path / "es.json"
        locale_file.write_text(json.dumps(locale_data), encoding="utf-8")
        report = simplify_output(self._make_output(), locale=str(locale_file))
        assert report.tier_label == "Bajo"
        assert report.locale == "es"

    def test_none_tasks_handled(self) -> None:
        output = {"rhythm": None, "structural": None, "ischaemia": None, "risk": None}
        report = simplify_output(output)
        assert report.tier == TIER_LOW

    def test_partial_tasks(self) -> None:
        output = {"rhythm": [0.0] * 28}
        report = simplify_output(output)
        assert report.tier == TIER_LOW

    def test_multiple_urgent_findings(self) -> None:
        rhythm = [0.0] * 28
        vt_idx = _RHYTHM_CLASSES.index("VT")
        vf_idx = _RHYTHM_CLASSES.index("VF")
        rhythm[vt_idx] = 0.90
        rhythm[vf_idx] = 0.80
        report = simplify_output(self._make_output(rhythm=rhythm))
        assert report.tier == TIER_URGENT
        urgent_findings = [f for f in report.key_findings if f.tier_contribution == TIER_URGENT]
        assert len(urgent_findings) >= 2

    def test_hyperkalaemia_urgent(self) -> None:
        ischaemia = [0.0] * 15
        hk_idx = _ISCHAEMIA_CLASSES.index("hyperkalaemia")
        ischaemia[hk_idx] = 0.60
        report = simplify_output(self._make_output(ischaemia=ischaemia))
        assert report.tier == TIER_URGENT

    def test_av_block_3rd_urgent(self) -> None:
        rhythm = [0.0] * 28
        idx = _RHYTHM_CLASSES.index("av_block_3rd")
        rhythm[idx] = 0.70
        report = simplify_output(self._make_output(rhythm=rhythm))
        assert report.tier == TIER_URGENT

    def test_default_locale_en(self) -> None:
        report = simplify_output(self._make_output())
        assert report.locale == "en"


# ── Imports ────────────────────────────────────────────────────────────────

class TestImports:
    def test_edge_package_exports(self) -> None:
        import aortica.edge
        assert hasattr(aortica.edge, "simplify_output")
        assert hasattr(aortica.edge, "SimplifiedReport")

    def test_module_importable(self) -> None:
        from aortica.edge import simplified_output
        assert hasattr(simplified_output, "simplify_output")
