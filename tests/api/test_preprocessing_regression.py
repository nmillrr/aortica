"""End-to-end regression: serving must feed the model what training fed it.

Training reads PTB-XL's 100 Hz records, unfiltered, in µV, windowed to
1,000 samples.  For the whole life of v0.3.0 the serving path instead fed
the model 500 Hz, denoised, 5,000-sample input.  The backbone pools
globally, so nothing raised — the model just scored near chance (PTB-XL
fold 10 macro-AUROC 0.525 through the API vs 0.943 on its training input).

These tests run real PTB-XL records through the real reader and the real
pipeline into the released v0.3.0 FP32 edge model, and compare against the
same model on the training data path's tensor for the same patient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from aortica.api.predict import run_inference_pipeline  # noqa: E402
from aortica.data.dataset import ECGDataset  # noqa: E402
from aortica.io import read_ecg as real_read_ecg  # noqa: E402
from aortica.models.aortica_model import MultiTaskOutput  # noqa: E402
from aortica.models.output_gating import default_trained_outputs  # noqa: E402
from aortica.signal.preprocess import (  # noqa: E402
    MODEL_NUM_SAMPLES,
    MODEL_SAMPLE_RATE_HZ,
    ModelInputError,
    model_input,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "ptbxl"
ONNX_FP32 = REPO_ROOT / "artifacts_combined" / "aortica_edge_fp32_v0.3.0.onnx"
RECORDS = ["00009", "00318", "00180", "00172"]
TASKS = ["rhythm", "structural", "ischaemia", "risk"]

#: Largest acceptable per-output disagreement between serving and training
#: paths.  The two differ only in who downsampled 500 -> 100 Hz (PTB-XL's
#: authors vs scipy), which moves outputs by ~0.002 on average.  The old
#: serving path moved them by ~0.08 on average and far more on many records.
TOLERANCE = 0.05


class OnnxAorticaModel:
    """Duck-types :class:`AorticaModel` over an exported ONNX graph."""

    def __init__(self, path: Path) -> None:
        self.session = ort.InferenceSession(str(path))
        self.enabled_tasks = list(TASKS)
        self.inputs_seen: List[np.ndarray] = []

    def eval(self) -> "OnnxAorticaModel":
        return self

    def __call__(self, x: Any, tasks: Optional[List[str]] = None) -> MultiTaskOutput:
        arr = x.detach().cpu().numpy().astype(np.float32)
        self.inputs_seen.append(arr)
        outs = self.session.run(None, {"ecg_input": arr})
        return MultiTaskOutput(**{t: torch.from_numpy(o) for t, o in zip(TASKS, outs)})


@pytest.fixture(scope="module")
def model() -> OnnxAorticaModel:
    if not ONNX_FP32.exists():
        pytest.skip(f"{ONNX_FP32.name} not present")
    return OnnxAorticaModel(ONNX_FP32)


def _serve(model: OnnxAorticaModel, record_id: str) -> dict[str, float]:
    """Run the 500 Hz file through the full serving pipeline."""
    hr = FIXTURES / "records500" / f"{record_id}_hr"
    with patch(
        "aortica.api.predict.read_ecg",
        side_effect=lambda _path, **kw: real_read_ecg(hr, **kw),
    ):
        resp = run_inference_pipeline(b"", f"{record_id}_hr.hea", model=model)
    return {
        name: prob
        for task in resp.predictions
        for name, prob in zip(task.class_names, task.probabilities)
    }


def _train_path(model: OnnxAorticaModel, record_id: str) -> dict[str, float]:
    """The same patient through the training data path."""
    from aortica.data.ptbxl import _read_record
    from aortica.models.ischaemia_head import ISCHAEMIA_CLASSES
    from aortica.models.rhythm_head import RHYTHM_CLASSES
    from aortica.models.structural_head import STRUCTURAL_CLASSES

    record = _read_record(FIXTURES / "records100" / f"{record_id}_lr", 100.0)
    ds = ECGDataset([record], np.zeros((1, 72), np.float32), window_seconds=10.0)
    x, _ = ds[0]
    out = model(x.unsqueeze(0)).as_dict()
    names = {"rhythm": RHYTHM_CLASSES, "structural": STRUCTURAL_CLASSES,
             "ischaemia": ISCHAEMIA_CLASSES}
    return {n: float(p) for t, ns in names.items() for n, p in zip(ns, out[t][0].numpy())}


@pytest.mark.parametrize("record_id", RECORDS)
def test_model_receives_training_shaped_input(model: OnnxAorticaModel, record_id: str) -> None:
    model.inputs_seen.clear()
    _serve(model, record_id)
    (x,) = model.inputs_seen
    assert x.shape == (1, 12, MODEL_NUM_SAMPLES)


@pytest.mark.parametrize("record_id", RECORDS)
def test_serving_matches_training_path(model: OnnxAorticaModel, record_id: str) -> None:
    served = _serve(model, record_id)
    trained = _train_path(model, record_id)
    assert set(served) == set(default_trained_outputs())
    diffs = {n: abs(served[n] - trained[n]) for n in served}
    worst = max(diffs, key=diffs.get)  # type: ignore[arg-type]
    assert diffs[worst] < TOLERANCE, (
        f"{record_id}: serving and training paths disagree on {worst} "
        f"({served[worst]:.3f} vs {trained[worst]:.3f})"
    )


@pytest.mark.parametrize(
    "record_id, finding",
    [("00318", "AF"), ("00180", "LBBB"), ("00172", "RBBB")],
)
def test_served_finding_is_detected(model: OnnxAorticaModel, record_id: str,
                                    finding: str) -> None:
    """Each labelled finding clears its released operating point."""
    from aortica.models.output_gating import default_class_thresholds

    served = _serve(model, record_id)
    assert served[finding] >= default_class_thresholds()[finding]


def test_old_serving_path_would_fail(model: OnnxAorticaModel) -> None:
    """Guard the guard: the pre-fix input really does fall outside tolerance."""
    from aortica.signal import denoise

    hr = real_read_ecg(FIXTURES / "records500" / "00180_hr")  # 500 Hz, as served
    old = denoise(hr).signals.astype(np.float32)[None]
    new = model_input(hr)[None]
    old_out = model(torch.from_numpy(old)).as_dict()["rhythm"][0].numpy()
    new_out = model(torch.from_numpy(new)).as_dict()["rhythm"][0].numpy()
    assert np.abs(old_out - new_out).max() > TOLERANCE


def test_model_input_contract() -> None:
    hr = real_read_ecg(FIXTURES / "records500" / "00009_hr", resample=False)
    assert hr.sample_rate == 500.0
    x = model_input(hr)
    assert x.shape == (12, MODEL_NUM_SAMPLES) and x.dtype == np.float32
    # Same record already at 100 Hz goes through untouched apart from dtype.
    lr = real_read_ecg(FIXTURES / "records100" / "00009_lr", resample=False)
    np.testing.assert_allclose(model_input(lr), lr.signals.astype(np.float32))


def test_model_input_rejects_unknown_units() -> None:
    hr = real_read_ecg(FIXTURES / "records500" / "00009_hr")
    hr.units = "bpm"
    with pytest.raises(ModelInputError):
        model_input(hr)


def test_model_input_converts_millivolts() -> None:
    hr = real_read_ecg(FIXTURES / "records500" / "00009_hr")
    as_mv = type(hr)(signals=hr.signals / 1000.0, sample_rate=hr.sample_rate,
                     lead_names=hr.lead_names, units="mV")
    np.testing.assert_allclose(model_input(as_mv), model_input(hr), rtol=1e-5)


def test_contract_matches_packaged_sidecar() -> None:
    import json

    import aortica.models.output_gating as gating

    sidecar = json.loads(gating._TRAINED_OUTPUTS_PATH.read_text())
    assert float(sidecar["sampling_rate"]) == MODEL_SAMPLE_RATE_HZ
