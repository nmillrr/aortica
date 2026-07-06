"""Tests for aortica.edge.power_validation — sustained-power validation (US-061b)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from aortica.edge.power_validation import (
    DEFAULT_POWER_BUDGET_MW,
    PowerValidationReport,
    _resolve_input_shape,
    compute_sustained_power,
    validate_power_consumption,
)

# ---------------------------------------------------------------------------
# compute_sustained_power — pure math, no onnxruntime
# ---------------------------------------------------------------------------


def test_compute_sustained_power_basic() -> None:
    # 300 ms active every 300 s → duty cycle 0.001 → 4 W × 0.001 = 4 mW
    duty, mw = compute_sustained_power(300.0, 4.0, 300.0)
    assert duty == pytest.approx(0.001, rel=1e-6)
    assert mw == pytest.approx(4.0, rel=1e-6)


def test_compute_sustained_power_under_budget() -> None:
    _, mw = compute_sustained_power(350.0, 4.0, 300.0)
    assert mw < DEFAULT_POWER_BUDGET_MW


def test_compute_sustained_power_continuous_is_full_tdp() -> None:
    # latency == interval → duty cycle capped at 1.0 → full TDP in mW
    duty, mw = compute_sustained_power(1000.0, 4.0, 1.0)
    assert duty == 1.0
    assert mw == pytest.approx(4000.0)


@pytest.mark.parametrize(
    "latency,tdp,interval",
    [(-1.0, 4.0, 300.0), (300.0, 0.0, 300.0), (300.0, 4.0, 0.0)],
)
def test_compute_sustained_power_validates_inputs(
    latency: float, tdp: float, interval: float
) -> None:
    with pytest.raises(ValueError):
        compute_sustained_power(latency, tdp, interval)


# ---------------------------------------------------------------------------
# _resolve_input_shape
# ---------------------------------------------------------------------------


def test_resolve_input_shape_dynamic_dims() -> None:
    assert _resolve_input_shape(["N", 12, 5000]) == (1, 12, 5000)
    assert _resolve_input_shape([None, None, None]) == (1, 12, 5000)
    assert _resolve_input_shape([2, 6, 1000]) == (2, 6, 1000)


# ---------------------------------------------------------------------------
# PowerValidationReport
# ---------------------------------------------------------------------------


def test_report_to_dict_and_summary() -> None:
    rep = PowerValidationReport(
        hardware_profile="rpi4",
        tdp_watts=4.0,
        n_inferences=50,
        mean_latency_ms=10.0,
        sustained_power_mw=0.13,
        threshold_mw=200.0,
        passed=True,
    )
    d = rep.to_dict()
    assert d["hardware_profile"] == "rpi4"
    assert d["passed"] is True
    assert "PASS" in rep.summary()


# ---------------------------------------------------------------------------
# validate_power_consumption — end-to-end with a real minimal ONNX model
# ---------------------------------------------------------------------------

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required")


def _make_identity_model(path: str, shape: list[object]) -> str:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("ecg_input", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    node = helper.make_node("Identity", ["ecg_input"], ["output"])
    graph = helper.make_graph([node], "m", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


@pytest.fixture()
def identity_model() -> str:
    path = os.path.join(tempfile.mkdtemp(), "identity.onnx")
    return _make_identity_model(path, [1, 12, 5000])


def test_validate_power_consumption_passes(identity_model: str) -> None:
    report = validate_power_consumption(identity_model, "rpi4", n_inferences=10)
    assert report.passed
    assert report.sustained_power_mw < 200.0
    assert report.n_inferences == 10
    assert report.tdp_watts == 4.0
    assert report.mean_latency_ms >= 0.0


def test_validate_power_consumption_dynamic_shape() -> None:
    path = os.path.join(tempfile.mkdtemp(), "dyn.onnx")
    _make_identity_model(path, ["N", 12, 5000])
    report = validate_power_consumption(path, "jetson_nano", n_inferences=5)
    assert report.hardware_profile == "jetson_nano"
    assert report.tdp_watts == 5.0
    assert report.passed


def test_validate_power_consumption_custom_input(identity_model: str) -> None:
    x = np.zeros((1, 12, 5000), dtype=np.float32)
    report = validate_power_consumption(
        identity_model, "rpi4", n_inferences=5, input_data=x
    )
    assert report.passed


def test_validate_power_consumption_high_duty_fails(identity_model: str) -> None:
    # Very short interval → duty cycle near 1 → power exceeds budget.
    report = validate_power_consumption(
        identity_model,
        "rpi4",
        n_inferences=5,
        inference_interval_seconds=1e-6,
    )
    assert not report.passed
    assert report.sustained_power_mw >= report.threshold_mw


def test_validate_power_consumption_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        validate_power_consumption("/nonexistent/model.onnx", "rpi4")


def test_validate_power_consumption_bad_n(identity_model: str) -> None:
    with pytest.raises(ValueError):
        validate_power_consumption(identity_model, "rpi4", n_inferences=0)
