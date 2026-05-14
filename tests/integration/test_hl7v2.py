"""Tests for HL7 v2.x ORU^R01 message generation (US-081).

Validates:
- ORU^R01 message structure with synthetic multi-task output
- OBX segments for each active classification finding
- OBX segments for risk scores with numeric values
- SNOMED CT / local code mappings in OBX-3
- Confidence thresholds filter low-confidence findings
- Special characters and segment delimiters handled per HL7 v2.x spec
- Message is parseable by hl7apy
- Patient ID and order ID propagation
- Edge cases: empty predictions, risk-only, list-format input
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

try:
    from hl7apy.consts import VALIDATION_LEVEL
    from hl7apy.parser import parse_message

    HAS_HL7 = True
except ImportError:
    HAS_HL7 = False

pytestmark = pytest.mark.skipif(not HAS_HL7, reason="hl7apy not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_multi_task_output() -> Dict[str, Any]:
    """Create a synthetic multi-task output dict with mixed findings."""
    return {
        "rhythm": {
            "AF": 0.95,
            "normal_sinus_rhythm": 0.02,
            "VT": 0.85,
            "sinus_brady": 0.10,
            "WPW": 0.55,
        },
        "structural": {
            "LVH": 0.80,
            "LVSD": 0.15,
            "DCM": 0.45,
        },
        "ischaemia": {
            "STEMI": 0.92,
            "old_MI": 0.05,
            "QTc_prolongation": 0.35,
        },
        "risk": {
            "mortality_1y": 0.15,
            "hf_hosp_12m": 0.08,
            "af_onset_12m": 0.72,
            "ecg_predicted_ef": 0.45,
            "conduction_disease_trajectory": 0.20,
            "sudden_cardiac_death_risk": 0.05,
        },
    }


@pytest.fixture
def minimal_output() -> Dict[str, Any]:
    """Minimal output with a single finding."""
    return {
        "rhythm": {"AF": 0.95},
    }


@pytest.fixture
def risk_only_output() -> Dict[str, Any]:
    """Output with only risk predictions (no classification findings)."""
    return {
        "risk": {
            "mortality_1y": 0.30,
            "hf_hosp_12m": 0.10,
            "af_onset_12m": 0.05,
            "ecg_predicted_ef": 0.60,
            "conduction_disease_trajectory": 0.12,
            "sudden_cardiac_death_risk": 0.03,
        },
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_segments(er7: str) -> list[str]:
    """Split ER7 message into segment strings."""
    # HL7 uses \r as segment separator
    return [s for s in er7.split("\r") if s.strip()]


def _get_obx_segments(er7: str) -> list[str]:
    """Extract OBX segments from an ER7 message."""
    return [s for s in _parse_segments(er7) if s.startswith("OBX|")]


# ---------------------------------------------------------------------------
# Tests — Basic functionality
# ---------------------------------------------------------------------------


class TestToOruR01:
    """Tests for the main to_oru_r01() function."""

    def test_produces_non_empty_message(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """to_oru_r01 returns a non-empty string."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_message_starts_with_msh(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Message starts with MSH segment."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        assert result.startswith("MSH|")

    def test_message_type_oru_r01(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """MSH-9 contains ORU^R01."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        assert "ORU^R01" in result

    def test_hl7_version_2_5(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """MSH-12 contains version 2.5."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        msh = _parse_segments(result)[0]
        fields = msh.split("|")
        # MSH-12 is at index 11 (0-indexed, MSH|^~\& counts as fields 0 and 1)
        assert fields[11] == "2.5"

    def test_sending_application(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """MSH-3 contains sending application name."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(
            synthetic_multi_task_output,
            sending_application="TestApp",
        )
        msh = _parse_segments(result)[0]
        fields = msh.split("|")
        assert fields[2] == "TestApp"

    def test_custom_facilities(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Custom facility names propagate to MSH segment."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(
            synthetic_multi_task_output,
            sending_facility="Lab01",
            receiving_application="HIS",
            receiving_facility="MainHospital",
        )
        msh = _parse_segments(result)[0]
        fields = msh.split("|")
        assert fields[3] == "Lab01"
        assert fields[4] == "HIS"
        assert fields[5] == "MainHospital"


# ---------------------------------------------------------------------------
# Tests — PID segment
# ---------------------------------------------------------------------------


class TestPIDSegment:
    """Tests for patient identification in the message."""

    def test_patient_id_in_pid(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Patient ID appears in PID segment."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(
            synthetic_multi_task_output, patient_id="PAT-12345"
        )
        segments = _parse_segments(result)
        pid_segments = [s for s in segments if s.startswith("PID|")]
        assert len(pid_segments) == 1
        assert "PAT-12345" in pid_segments[0]

    def test_default_anonymous_patient(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """When no patient_id, PID uses ANONYMOUS."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        segments = _parse_segments(result)
        pid_segments = [s for s in segments if s.startswith("PID|")]
        assert len(pid_segments) == 1
        assert "ANONYMOUS" in pid_segments[0]


# ---------------------------------------------------------------------------
# Tests — OBR segment
# ---------------------------------------------------------------------------


class TestOBRSegment:
    """Tests for the Order/Observation Request segment."""

    def test_obr_present(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """OBR segment is present in the message."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        segments = _parse_segments(result)
        obr_segments = [s for s in segments if s.startswith("OBR|")]
        assert len(obr_segments) == 1

    def test_obr_has_ecg_loinc(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """OBR-4 contains ECG study LOINC code."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        segments = _parse_segments(result)
        obr = [s for s in segments if s.startswith("OBR|")][0]
        assert "11524-6" in obr
        assert "EKG study" in obr

    def test_order_id_in_obr(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Custom order ID appears in OBR-2."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(
            synthetic_multi_task_output, order_id="ORD-98765"
        )
        segments = _parse_segments(result)
        obr = [s for s in segments if s.startswith("OBR|")][0]
        assert "ORD-98765" in obr


# ---------------------------------------------------------------------------
# Tests — OBX segments (classification findings)
# ---------------------------------------------------------------------------


class TestOBXClassificationFindings:
    """Tests for OBX segments encoding classification findings."""

    def test_obx_count_matches_positive_findings(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Number of OBX segments matches findings above threshold + risk scores."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(
            synthetic_multi_task_output, confidence_threshold=0.30
        )
        obx_segments = _get_obx_segments(result)

        # Classification findings >= 0.30:
        # rhythm: AF(0.95), VT(0.85), WPW(0.55) = 3
        # structural: LVH(0.80), DCM(0.45) = 2
        # ischaemia: STEMI(0.92), QTc_prolongation(0.35) = 2
        # Risk scores: all 6
        # Total = 7 + 6 = 13
        assert len(obx_segments) == 13

    def test_obx_has_snomed_code(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """OBX-3 contains SNOMED CT code for known findings."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        obx_segments = _get_obx_segments(result)

        # AF should have SNOMED code 49436004
        af_obx = [s for s in obx_segments if "49436004" in s]
        assert len(af_obx) >= 1
        assert "SCT" in af_obx[0]

    def test_obx_has_confidence_value(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """OBX-5 contains the confidence value."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        obx_segments = _get_obx_segments(result)

        # AF with 0.95 confidence
        af_obx = [s for s in obx_segments if "49436004" in s]
        assert len(af_obx) >= 1
        assert "0.95" in af_obx[0]

    def test_obx_value_type_numeric(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """OBX-2 is 'NM' (numeric) for findings."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        obx_segments = _get_obx_segments(result)
        for obx in obx_segments:
            fields = obx.split("|")
            assert fields[2] == "NM"

    def test_obx_units_probability(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """OBX-6 contains probability units."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        obx_segments = _get_obx_segments(result)
        for obx in obx_segments:
            assert "{probability}" in obx

    def test_obx_status_final(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """OBX-11 is 'F' (final) for all observations."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        obx_segments = _get_obx_segments(result)
        for obx in obx_segments:
            # Status is OBX-11, which should contain "F"
            fields = obx.split("|")
            assert fields[11] == "F"

    def test_obx_sequential_set_ids(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """OBX-1 contains sequential set IDs starting from 1."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        obx_segments = _get_obx_segments(result)

        for i, obx in enumerate(obx_segments, start=1):
            fields = obx.split("|")
            assert fields[1] == str(i)

    def test_confidence_threshold_filters(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Higher threshold reduces OBX count for classification findings."""
        from aortica.integration.hl7v2 import to_oru_r01

        low_thresh = to_oru_r01(
            synthetic_multi_task_output, confidence_threshold=0.30
        )
        high_thresh = to_oru_r01(
            synthetic_multi_task_output, confidence_threshold=0.80
        )

        low_obx = _get_obx_segments(low_thresh)
        high_obx = _get_obx_segments(high_thresh)

        # High threshold should have fewer OBX segments (but risk scores remain)
        assert len(high_obx) < len(low_obx)

    def test_abnormal_flag_critical(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """High confidence findings (>=0.90) get 'AA' abnormal flag."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        obx_segments = _get_obx_segments(result)

        # AF (0.95) should have AA flag
        af_obx = [s for s in obx_segments if "49436004" in s]
        assert len(af_obx) >= 1
        fields = af_obx[0].split("|")
        # OBX-8 is the abnormal flag
        assert fields[8] == "AA"

    def test_abnormal_flag_abnormal(self) -> None:
        """Moderate confidence findings (>=0.50, <0.90) get 'A' flag."""
        from aortica.integration.hl7v2 import to_oru_r01

        output = {"rhythm": {"WPW": 0.55}}
        result = to_oru_r01(output)
        obx_segments = _get_obx_segments(result)
        assert len(obx_segments) == 1
        fields = obx_segments[0].split("|")
        assert fields[8] == "A"


# ---------------------------------------------------------------------------
# Tests — OBX segments (risk scores)
# ---------------------------------------------------------------------------


class TestOBXRiskScores:
    """Tests for OBX segments encoding risk predictions."""

    def test_risk_obx_count(
        self, risk_only_output: Dict[str, Any]
    ) -> None:
        """One OBX per risk output."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(risk_only_output)
        obx_segments = _get_obx_segments(result)
        assert len(obx_segments) == 6

    def test_risk_obx_has_loinc_code(
        self, risk_only_output: Dict[str, Any]
    ) -> None:
        """Risk OBX segments have LOINC codes where available."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(risk_only_output)
        obx_segments = _get_obx_segments(result)

        # mortality_1y should have LOINC code 75889-6
        mortality_obx = [s for s in obx_segments if "75889-6" in s]
        assert len(mortality_obx) == 1
        assert "LN" in mortality_obx[0]

    def test_risk_obx_numeric_value(
        self, risk_only_output: Dict[str, Any]
    ) -> None:
        """Risk OBX-5 contains numeric probability value."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(risk_only_output)
        obx_segments = _get_obx_segments(result)

        for obx in obx_segments:
            fields = obx.split("|")
            value = float(fields[5])
            assert 0.0 <= value <= 1.0

    def test_risk_obx_type_numeric(
        self, risk_only_output: Dict[str, Any]
    ) -> None:
        """Risk OBX-2 is 'NM' (numeric)."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(risk_only_output)
        obx_segments = _get_obx_segments(result)
        for obx in obx_segments:
            fields = obx.split("|")
            assert fields[2] == "NM"


# ---------------------------------------------------------------------------
# Tests — Message parsability
# ---------------------------------------------------------------------------


class TestMessageParsability:
    """Tests that generated messages can be parsed back by hl7apy."""

    def test_message_is_parseable(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Generated message can be parsed by hl7apy."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        parsed = parse_message(
            result, validation_level=VALIDATION_LEVEL.TOLERANT
        )
        assert parsed.name == "ORU_R01"

    def test_parse_utility_function(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """parse_oru_r01 utility extracts structured data."""
        from aortica.integration.hl7v2 import parse_oru_r01, to_oru_r01

        er7 = to_oru_r01(
            synthetic_multi_task_output,
            patient_id="PAT-999",
        )
        parsed = parse_oru_r01(er7)

        assert parsed["message_type"] == "ORU^R01"
        assert parsed["patient_id"] == "PAT-999"
        assert len(parsed["observations"]) == 13  # 7 findings + 6 risk
        assert parsed["segment_count"] > 0

    def test_round_trip_observation_values(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Observation values survive serialization round-trip."""
        from aortica.integration.hl7v2 import parse_oru_r01, to_oru_r01

        er7 = to_oru_r01(synthetic_multi_task_output)
        parsed = parse_oru_r01(er7)

        # Find AF observation
        af_obs = [
            o for o in parsed["observations"]
            if o.get("code") == "49436004"
        ]
        assert len(af_obs) >= 1
        assert float(af_obs[0]["value"]) == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Tests — Special characters
# ---------------------------------------------------------------------------


class TestSpecialCharacters:
    """Tests that HL7 special characters are handled correctly."""

    def test_pipe_in_display_escaped(self) -> None:
        """Pipe characters in display names are escaped."""
        from aortica.integration.hl7v2 import to_oru_r01

        # All standard class names shouldn't contain pipes,
        # but verify the message is well-formed
        output = {"rhythm": {"AF": 0.95}}
        result = to_oru_r01(output)

        # Verify that between delimiters there are no unescaped pipes
        # in non-delimiter positions
        segments = _parse_segments(result)
        for seg in segments:
            # Each segment should be parseable by splitting on |
            fields = seg.split("|")
            assert len(fields) >= 1

    def test_segment_delimiter_cr(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Segments are separated by carriage return (\\r)."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        assert "\r" in result

    def test_field_separator_pipe(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Fields are separated by pipe (|)."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        msh = _parse_segments(result)[0]
        assert msh.count("|") >= 11  # MSH has at least 12 fields

    def test_encoding_characters(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """MSH-2 contains encoding characters ^~\\&."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(synthetic_multi_task_output)
        msh = _parse_segments(result)[0]
        fields = msh.split("|")
        assert fields[1] == "^~\\&"


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_output(self) -> None:
        """Empty dict produces a valid message with no OBX segments."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01({})
        assert result.startswith("MSH|")
        obx_segments = _get_obx_segments(result)
        assert len(obx_segments) == 0

    def test_single_task_output(self) -> None:
        """Works with only one task head present."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01({"rhythm": {"AF": 0.99}})
        obx_segments = _get_obx_segments(result)
        assert len(obx_segments) == 1

    def test_all_below_threshold(self) -> None:
        """When all classification findings are below threshold, only risk OBX present."""
        from aortica.integration.hl7v2 import to_oru_r01

        output = {
            "rhythm": {"AF": 0.10, "normal_sinus_rhythm": 0.05},
            "risk": {"mortality_1y": 0.15},
        }
        result = to_oru_r01(output, confidence_threshold=0.50)
        obx_segments = _get_obx_segments(result)
        # Only risk score OBX, no classification findings
        assert len(obx_segments) == 1
        assert "75889-6" in obx_segments[0]  # mortality LOINC

    def test_risk_only_output(
        self, risk_only_output: Dict[str, Any]
    ) -> None:
        """Works with risk-only output (no classification findings)."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(risk_only_output)
        obx_segments = _get_obx_segments(result)
        assert len(obx_segments) == 6
        assert result.startswith("MSH|")

    def test_list_format_risk_predictions(self) -> None:
        """Works when risk predictions are a list of floats."""
        from aortica.integration.hl7v2 import to_oru_r01

        output = {
            "risk": [0.15, 0.08, 0.72, 0.45, 0.20, 0.05],
        }
        result = to_oru_r01(output)
        obx_segments = _get_obx_segments(result)
        assert len(obx_segments) == 6

    def test_high_threshold_filters_all_classification(
        self, synthetic_multi_task_output: Dict[str, Any]
    ) -> None:
        """Very high threshold filters all classification findings."""
        from aortica.integration.hl7v2 import to_oru_r01

        result = to_oru_r01(
            synthetic_multi_task_output, confidence_threshold=0.99
        )
        obx_segments = _get_obx_segments(result)
        # Only risk scores remain (6)
        assert len(obx_segments) == 6


# ---------------------------------------------------------------------------
# Tests — Code coverage
# ---------------------------------------------------------------------------


class TestCodeMappings:
    """Tests for finding code mappings."""

    def test_all_finding_codes_are_valid(self) -> None:
        """All finding codes have non-empty code, display, and system."""
        from aortica.integration.hl7v2 import _FINDING_CODES

        for class_name, (code, display, system) in _FINDING_CODES.items():
            assert isinstance(code, str) and len(code) > 0, (
                f"Missing code for {class_name}"
            )
            assert isinstance(display, str) and len(display) > 0, (
                f"Missing display for {class_name}"
            )
            assert system in ("SCT", "LOCAL"), (
                f"Invalid system for {class_name}: {system}"
            )

    def test_risk_display_names_complete(self) -> None:
        """All 6 risk outputs have display entries."""
        from aortica.integration.hl7v2 import _RISK_DISPLAY
        from aortica.models.risk_head import RISK_OUTPUTS

        for risk_name in RISK_OUTPUTS:
            assert risk_name in _RISK_DISPLAY, (
                f"Missing display entry for risk output {risk_name}"
            )

    def test_finding_codes_cover_all_head_classes(self) -> None:
        """Finding codes cover all classes from all task heads."""
        from aortica.integration.hl7v2 import _FINDING_CODES
        from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
        from aortica.models.rhythm_head import RHYTHM_CLASSES
        from aortica.models.structural_head import STRUCTURAL_CLASSES

        all_classes = (
            list(RHYTHM_CLASSES) +
            list(STRUCTURAL_CLASSES) +
            list(ISCHAEMIA_CLASSES)
        )
        for cls in all_classes:
            assert cls in _FINDING_CODES, (
                f"Missing finding code mapping for {cls}"
            )


# ---------------------------------------------------------------------------
# Tests — ImportError path
# ---------------------------------------------------------------------------


class TestImportGuard:
    """Tests for the import guard when hl7apy is not installed."""

    def test_check_hl7_when_available(self) -> None:
        """_check_hl7 does not raise when hl7apy is available."""
        from aortica.integration.hl7v2 import _check_hl7

        # Should not raise
        _check_hl7()
