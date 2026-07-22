"""End-to-end EHR integration orchestrator (US-125).

Wires the standalone integration components into one automated loop::

    ECG ingested → AI processing → result storage → PACS write-back →
    EHR submission → worklist update → notification

Each step is isolated: a failure in one channel (e.g. PACS) does not block
the others.  Failed steps are queued for retry and, after exhausting
retries, moved to a dead-letter log.  Per-channel success/failure counts and
the last error feed the ``/api/v1/integration/status`` endpoint and the
error-rate health monitor.

The external-system steps (PACS write-back, EHR submission) are injected as
callables so the orchestrator is fully testable with mock endpoints and the
app never hard-depends on ``pynetdicom`` / a live EHR.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Ordered channels the orchestrator drives after AI processing.
CHANNELS = ("storage", "pacs", "ehr", "worklist", "notification")

#: A processor turns a polled ECG payload into a result dict.
Processor = Callable[[Any], Dict[str, Any]]


@dataclass
class IntegrationConfig:
    """Which channels are enabled and orchestrator tuning.

    Attributes:
        enabled_channels: Channels to run (subset of :data:`CHANNELS`).
        max_retries: Retry attempts per failed step before dead-lettering.
        error_rate_threshold: Per-channel error-rate alert threshold.
        error_rate_window_seconds: Window over which the rate is measured.
    """

    enabled_channels: List[str] = field(
        default_factory=lambda: list(CHANNELS)
    )
    max_retries: int = 3
    error_rate_threshold: float = 0.05
    error_rate_window_seconds: float = 3600.0

    def __post_init__(self) -> None:
        unknown = set(self.enabled_channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"Unknown channels: {sorted(unknown)}")


def load_integration_config(path: str | Path) -> IntegrationConfig:
    """Load ``integration.yaml`` into an :class:`IntegrationConfig`."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")
    text = path_obj.read_text()
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
    except ImportError:
        from aortica.federated.fl_server import _simple_yaml_load

        data = _simple_yaml_load(text)
    if not isinstance(data, dict):
        raise ValueError("integration.yaml must be a mapping")
    channels = data.get("enabled_channels") or list(CHANNELS)
    return IntegrationConfig(
        enabled_channels=list(channels),
        max_retries=int(data.get("max_retries", 3)),
        error_rate_threshold=float(data.get("error_rate_threshold", 0.05)),
        error_rate_window_seconds=float(
            data.get("error_rate_window_seconds", 3600.0)
        ),
    )


@dataclass
class StepResult:
    """Outcome of one channel step for one ECG."""

    channel: str
    ok: bool
    skipped: bool = False
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class ProcessResult:
    """Outcome of processing one ECG through the whole pipeline."""

    ecg_id: str
    steps: List[StepResult] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(s.ok or s.skipped for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ecg_id": self.ecg_id,
            "ok": self.ok,
            "steps": [
                {
                    "channel": s.channel,
                    "ok": s.ok,
                    "skipped": s.skipped,
                    "error": s.error,
                    "attempts": s.attempts,
                }
                for s in self.steps
            ],
        }


@dataclass
class ChannelStats:
    """Rolling success/failure counters for one channel."""

    name: str
    success: int = 0
    failure: int = 0
    last_error: Optional[str] = None
    recent: List[tuple[float, bool]] = field(default_factory=list)

    def record(self, ok: bool, error: Optional[str], window: float) -> None:
        now = time.time()
        if ok:
            self.success += 1
        else:
            self.failure += 1
            self.last_error = error
        self.recent.append((now, ok))
        cutoff = now - window
        self.recent = [(t, o) for t, o in self.recent if t >= cutoff]

    def error_rate(self) -> float:
        if not self.recent:
            return 0.0
        failures = sum(1 for _t, ok in self.recent if not ok)
        return failures / len(self.recent)


class IntegrationOrchestrator:
    """Drives the full ECG → EHR/PACS integration loop.

    Args:
        config: Enabled channels and tuning.
        processor: Callable mapping a polled ECG payload to a result dict
            (the AI processing step).
        result_store: Optional store with ``store_result(ecg_id, result)``
            or ``add_from_prioritized`` — persists the result (step 3).
        pacs_writer: Optional callable ``(ecg_id, result) -> None`` writing
            a DICOM SR to PACS (step 4).
        ehr_submitter: Optional callable ``(ecg_id, result) -> None``
            submitting a FHIR/HL7 report to the EHR (step 5).
        worklist_store: Optional ``WorklistStore`` (step 6).
        subscription_manager: Optional ``SubscriptionManager`` (step 7).
        sleep: Injectable sleep for retry backoff (tests).
    """

    def __init__(
        self,
        config: IntegrationConfig,
        *,
        processor: Processor,
        result_store: Optional[Any] = None,
        pacs_writer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        ehr_submitter: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        worklist_store: Optional[Any] = None,
        subscription_manager: Optional[Any] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._processor = processor
        self._result_store = result_store
        self._pacs_writer = pacs_writer
        self._ehr_submitter = ehr_submitter
        self._worklist_store = worklist_store
        self._subscription_manager = subscription_manager
        self._sleep = sleep

        self._lock = threading.RLock()
        self._stats: Dict[str, ChannelStats] = {
            c: ChannelStats(name=c) for c in CHANNELS
        }
        self._retry_queue: List[tuple[str, str, Dict[str, Any]]] = []
        self._dead_letter: List[Dict[str, Any]] = []
        self._processed = 0
        self._last_error: Optional[str] = None

    # -- Step handlers ------------------------------------------------------

    def _handler(self, channel: str) -> Optional[Callable[[str, Dict[str, Any]], None]]:
        if channel == "storage":
            return self._store if self._result_store is not None else None
        if channel == "pacs":
            return self._pacs_writer
        if channel == "ehr":
            return self._ehr_submitter
        if channel == "worklist":
            return self._to_worklist if self._worklist_store is not None else None
        if channel == "notification":
            return (
                self._notify
                if self._subscription_manager is not None
                else None
            )
        return None

    def _store(self, ecg_id: str, result: Dict[str, Any]) -> None:
        store: Any = self._result_store
        if hasattr(store, "store_result"):
            store.store_result(ecg_id, result)
        elif hasattr(store, "add_result"):
            store.add_result(ecg_id, result)
        else:  # pragma: no cover - defensive
            raise RuntimeError("result_store has no store_result/add_result")

    def _to_worklist(self, ecg_id: str, result: Dict[str, Any]) -> None:
        from aortica.integration.worklist import WorklistPrioritizer

        worklist = WorklistPrioritizer().prioritize([result], ecg_ids=[ecg_id])
        store: Any = self._worklist_store
        store.add_from_prioritized(worklist)

    def _notify(self, ecg_id: str, result: Dict[str, Any]) -> None:
        manager: Any = self._subscription_manager
        manager.process_result(result, ecg_id=ecg_id, sync=True)

    # -- Processing ---------------------------------------------------------

    def process_ecg(self, ecg_id: str, payload: Any) -> ProcessResult:
        """Run one ECG through AI processing + every enabled channel."""
        proc = ProcessResult(ecg_id=ecg_id)

        # Step 2 — AI processing (fatal: no result means nothing to route).
        try:
            result = self._processor(payload)
            proc.result = result
        except Exception as exc:  # noqa: BLE001
            logger.exception("AI processing failed for %s", ecg_id)
            proc.steps.append(
                StepResult(channel="processing", ok=False, error=str(exc))
            )
            with self._lock:
                self._last_error = f"processing: {exc}"
            return proc

        for channel in CHANNELS:
            proc.steps.append(self._run_step(channel, ecg_id, result))

        with self._lock:
            self._processed += 1
        return proc

    def _run_step(
        self, channel: str, ecg_id: str, result: Dict[str, Any]
    ) -> StepResult:
        if channel not in self._config.enabled_channels:
            return StepResult(channel=channel, ok=True, skipped=True)
        handler = self._handler(channel)
        if handler is None:
            return StepResult(channel=channel, ok=True, skipped=True)
        try:
            handler(ecg_id, result)
            with self._lock:
                self._stats[channel].record(
                    True, None, self._config.error_rate_window_seconds
                )
            return StepResult(channel=channel, ok=True, attempts=1)
        except Exception as exc:  # noqa: BLE001 - isolate per channel
            logger.warning("Channel %s failed for %s: %s", channel, ecg_id, exc)
            with self._lock:
                self._stats[channel].record(
                    False, str(exc), self._config.error_rate_window_seconds
                )
                self._last_error = f"{channel}: {exc}"
                self._retry_queue.append((channel, ecg_id, result))
            return StepResult(
                channel=channel, ok=False, error=str(exc), attempts=1
            )

    def retry_failed(self) -> int:
        """Drain the retry queue; dead-letter items that keep failing.

        Returns the number of items successfully retried.
        """
        with self._lock:
            queue = list(self._retry_queue)
            self._retry_queue.clear()

        succeeded = 0
        for channel, ecg_id, result in queue:
            handler = self._handler(channel)
            if handler is None:
                continue
            ok = False
            last_err: Optional[str] = None
            for attempt in range(1, self._config.max_retries + 1):
                try:
                    handler(ecg_id, result)
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    if attempt < self._config.max_retries:
                        self._sleep(0)  # backoff hook (injected no-op in tests)
            with self._lock:
                self._stats[channel].record(
                    ok, last_err, self._config.error_rate_window_seconds
                )
                if ok:
                    succeeded += 1
                else:
                    self._dead_letter.append(
                        {"channel": channel, "ecg_id": ecg_id, "error": last_err}
                    )
        return succeeded

    # -- Introspection ------------------------------------------------------

    def health_alerts(self) -> List[Dict[str, Any]]:
        """Return alerts for channels exceeding the error-rate threshold."""
        alerts: List[Dict[str, Any]] = []
        with self._lock:
            for channel, stats in self._stats.items():
                rate = stats.error_rate()
                if rate > self._config.error_rate_threshold and stats.recent:
                    alerts.append(
                        {
                            "channel": channel,
                            "error_rate": rate,
                            "threshold": self._config.error_rate_threshold,
                            "last_error": stats.last_error,
                        }
                    )
        return alerts

    def status(self) -> Dict[str, Any]:
        """Return orchestrator status for the API."""
        with self._lock:
            return {
                "status": "running",
                "processed": self._processed,
                "queue_depth": len(self._retry_queue),
                "dead_letter_count": len(self._dead_letter),
                "last_error": self._last_error,
                "enabled_channels": list(self._config.enabled_channels),
                "channels": {
                    name: {
                        "success": s.success,
                        "failure": s.failure,
                        "error_rate": s.error_rate(),
                        "last_error": s.last_error,
                    }
                    for name, s in self._stats.items()
                },
                "health_alerts": self.health_alerts(),
            }

    def dead_letter(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._dead_letter)
