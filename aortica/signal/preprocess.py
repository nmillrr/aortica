"""The model's input contract — the one place training and inference agree.

Every released checkpoint was trained on the same thing: twelve leads in
standard order, **100 Hz**, a **10-second** window (1,000 samples), in
**micro-volts**, and **not** denoised.  The corpus loaders produce exactly
that, and nothing else in the training path transforms the signal.

Inference used to diverge from it.  ``read_ecg`` resamples to 500 Hz by
default and the API denoised before inference, so a 100 Hz model received
5,000 filtered samples per lead.  The backbone pools globally and accepted
the wrong shape without complaint, and its output was close to chance: on
PTB-XL fold 10 the v0.3.0 FP32 edge model scores macro-AUROC 0.943 on its
training-time input and 0.525 through the old API path.  The sample rate
accounts for nearly all of the loss; denoising alone moves it by <0.001.

So the contract lives here, and both sides call it:

* :class:`~aortica.data.dataset.ECGDataset` windows with
  :func:`window_signals`.
* The inference pipeline builds model input with :func:`model_input`.

Denoising remains part of the pipeline — signal quality scoring and the
display waveform use it — but it is deliberately not part of the model
input, because it was not part of the training input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aortica.io.ecg_record import ECGRecord

#: Sampling rate every released checkpoint was trained at.  Mirrored by
#: ``sampling_rate`` in ``aortica/models/artifacts/trained_outputs.json``;
#: a test keeps the two in step.
MODEL_SAMPLE_RATE_HZ: float = 100.0

#: Window length, in seconds, every training example was padded or cut to.
MODEL_WINDOW_SECONDS: float = 10.0

#: Samples per lead the model sees.
MODEL_NUM_SAMPLES: int = int(round(MODEL_SAMPLE_RATE_HZ * MODEL_WINDOW_SECONDS))

#: Leads the backbone expects, in order.
MODEL_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)

_TO_MICROVOLTS = {
    "µv": 1.0,
    "μv": 1.0,  # Greek mu, which some writers emit instead of the micro sign
    "uv": 1.0,
    "mv": 1e3,
    "v": 1e6,
}


class ModelInputError(ValueError):
    """A record cannot be turned into the input the model was trained on."""


def window_signals(signals: NDArray[np.floating], target_samples: int) -> NDArray[np.float64]:
    """Right-pad with zeros or right-truncate to exactly *target_samples*.

    This is the windowing the training dataset applies.  Keep it the only
    implementation: if training and inference window differently, a record
    shorter than ten seconds is padded one way and scored another.

    Args:
        signals: ``[leads, samples]``.
        target_samples: Desired samples per lead; must be positive.

    Returns:
        A new ``[leads, target_samples]`` array.
    """
    if target_samples <= 0:
        raise ValueError(f"target_samples must be > 0, got {target_samples}")
    n = signals.shape[1]
    if n >= target_samples:
        return np.array(signals[:, :target_samples], dtype=np.float64, copy=True)
    out = np.zeros((signals.shape[0], target_samples), dtype=np.float64)
    out[:, :n] = signals
    return out


def to_microvolts(record: ECGRecord) -> NDArray[np.float64]:
    """Return *record*'s signals scaled to micro-volts.

    Raises:
        ModelInputError: The record's units are not a voltage the model can
            be fed.  Guessing a scale factor would silently produce a 1000x
            error, which the model will happily score.
    """
    factor = _TO_MICROVOLTS.get(str(record.units).strip().lower())
    if factor is None:
        raise ModelInputError(
            f"Unrecognised signal units '{record.units}'; expected µV, mV or V."
        )
    return np.asarray(record.signals, dtype=np.float64) * factor


def model_record(record: ECGRecord) -> ECGRecord:
    """Return *record* as the model sees it: 100 Hz, 10 s, µV, 12 leads.

    The returned record is what attribution and retrieval should operate on,
    so their sample indices line up with the tensor the model scored.

    Raises:
        ModelInputError: Wrong lead set, or unusable units.
    """
    if tuple(record.lead_names) != MODEL_LEADS:
        raise ModelInputError(
            f"Model expects leads {list(MODEL_LEADS)} in that order; got "
            f"{list(record.lead_names)}."
        )
    signals = to_microvolts(record)
    rec = ECGRecord(
        signals=signals,
        sample_rate=record.sample_rate,
        lead_names=list(record.lead_names),
        patient_metadata=dict(record.patient_metadata) if record.patient_metadata else None,
        source_format=record.source_format,
        units="µV",
    )
    if rec.sample_rate != MODEL_SAMPLE_RATE_HZ:
        rec = rec.resample(MODEL_SAMPLE_RATE_HZ)
    windowed = window_signals(rec.signals, MODEL_NUM_SAMPLES)
    if not np.isfinite(windowed).all():
        raise ModelInputError("Signal contains non-finite samples.")
    return ECGRecord(
        signals=windowed,
        sample_rate=MODEL_SAMPLE_RATE_HZ,
        lead_names=list(MODEL_LEADS),
        patient_metadata=rec.patient_metadata,
        source_format=rec.source_format,
        units="µV",
    )


def model_input(record: ECGRecord) -> NDArray[np.float32]:
    """Return the ``[12, 1000]`` float32 array the model was trained on.

    Pass the record as read, **before** denoising.
    """
    return model_record(record).signals.astype(np.float32)
