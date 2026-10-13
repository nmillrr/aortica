"""Inbox-watching edge inference service — what the Raspberry Pi unit runs.

Polls a directory for ECG files, scores each with the INT8 ONNX edge model
and writes one JSON result per ECG.  This is the process behind
``aortica-edge.service`` (``aortica edge watch``).

Each ECG goes through the same path as the server:

* input built by :func:`aortica.signal.preprocess.model_input` — 100 Hz,
  10 s, µV, unfiltered, exactly what the model was trained on;
* outputs the model never trained are withheld
  (:mod:`aortica.models.output_gating`);
* findings are positive at their per-class operating point, and the ECG is
  given the worklist urgency tier.

The ONNX session is loaded on demand and released when idle
(:class:`~aortica.edge.deploy_profiles.DutyCycledModelLoader`), and the
service pings the systemd watchdog when run under one.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TASKS = ("rhythm", "structural", "ischaemia", "risk")


def sd_notify(message: str) -> bool:
    """Send *message* to systemd's notify socket, if there is one.

    Implements the datagram protocol from ``sd_notify(3)`` with the standard
    library, so ``Type=notify`` and ``WatchdogSec=`` work without the
    ``systemd`` Python bindings.  Returns ``False`` when not under systemd.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):  # abstract namespace
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(addr)
            sock.sendall(message.encode())
        return True
    except OSError:
        logger.warning("sd_notify(%s) failed", message, exc_info=True)
        return False


@dataclass
class EdgeScorer:
    """Score one ECG record with an ONNX edge model.

    Attributes:
        model_path: The ``.onnx`` file.
        thresholds: Per-class operating points; defaults to the packaged
            sidecar, which belongs to the shipped edge model.
        unload_after_seconds: Release the session after this much idle time.
    """

    model_path: Path
    thresholds: Optional[Dict[str, float]] = None
    unload_after_seconds: float = 30.0
    _loader: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from aortica.edge.deploy_profiles import DutyCycledModelLoader
        from aortica.models.output_gating import default_class_thresholds

        self.model_path = Path(self.model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Edge model not found: {self.model_path}")
        if self.thresholds is None:
            self.thresholds = default_class_thresholds()

        def _load() -> Any:
            import onnxruntime as ort

            return ort.InferenceSession(str(self.model_path))

        self._loader = DutyCycledModelLoader(_load, self.unload_after_seconds)

    def maybe_unload(self) -> bool:
        """Release the ONNX session if it has been idle long enough."""
        return bool(self._loader.maybe_unload())

    @property
    def is_loaded(self) -> bool:
        return bool(self._loader.is_loaded)

    def score(self, record: Any) -> Dict[str, Any]:
        """Return the result document for one :class:`ECGRecord`."""
        from aortica.integration.worklist import WorklistPrioritizer
        from aortica.models.output_gating import filter_task_outputs, is_positive
        from aortica.models.task_dims import TASK_CLASS_NAMES
        from aortica.signal import denoise, score_quality
        from aortica.signal.preprocess import model_input

        assert self.thresholds is not None
        x = model_input(record)[np.newaxis]
        with self._loader.session() as session:
            outputs = session.run(None, {session.get_inputs()[0].name: x})

        predictions: Dict[str, Dict[str, float]] = {}
        suppressed: List[str] = []
        positive: List[Dict[str, Any]] = []
        for task, out in zip(TASKS, outputs):
            kept, probs, dropped = filter_task_outputs(
                TASK_CLASS_NAMES[task], [float(v) for v in out[0]]
            )
            suppressed.extend(dropped)
            if kept:
                predictions[task] = dict(zip(kept, probs))
            for name, p in zip(kept, probs):
                if is_positive(name, p, self.thresholds):
                    positive.append({"finding": name, "task": task, "probability": p,
                                     "threshold": self.thresholds.get(name)})
        positive.sort(key=lambda f: -f["probability"])

        quality = score_quality(denoise(record))
        _, tier, _ = WorklistPrioritizer(thresholds=self.thresholds)._score_single(predictions)
        return {
            "model": self.model_path.name,
            "urgency_tier": tier,
            "positive_findings": positive,
            "predictions": predictions,
            "suppressed_classes": suppressed,
            "quality": {
                "overall_score": quality.overall_score,
                "classification": quality.overall_classification,
                "recommendation": quality.recommendation,
            },
            "scored_at": time.time(),
        }


def run_watch_loop(
    scorer: EdgeScorer,
    inbox: Path,
    outbox: Path,
    *,
    processed: Optional[Path] = None,
    interval_seconds: float = 5.0,
    max_cycles: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Poll *inbox*, score each new ECG, write ``<outbox>/<id>.json``.

    A file that cannot be read or scored gets an ``{"error": ...}`` result
    rather than stopping the service.

    Returns:
        Number of ECGs processed.
    """
    from aortica.integration.plugins.file_watcher import FileWatcherPlugin

    inbox.mkdir(parents=True, exist_ok=True)
    watcher = FileWatcherPlugin()
    config: Dict[str, Any] = {"watch_dir": str(inbox), "output_dir": str(outbox)}
    if processed is not None:
        config["processed_dir"] = str(processed)
    watcher.connect(config)

    sd_notify("READY=1")
    processed_count = 0
    cycles = 0
    while True:
        for ecg_id, record in watcher.poll_for_ecgs():
            try:
                result: Dict[str, Any] = {"ecg_id": ecg_id, **scorer.score(record)}
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the service
                logger.exception("Failed to score %s", ecg_id)
                result = {"ecg_id": ecg_id, "error": f"{type(exc).__name__}: {exc}"}
            watcher.submit_result(ecg_id, result)
            processed_count += 1
            logger.info("%s -> %s", ecg_id, result.get("urgency_tier", "error"))
        scorer.maybe_unload()
        sd_notify("WATCHDOG=1")
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        sleep(interval_seconds)
    return processed_count

