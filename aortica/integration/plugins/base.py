"""Plugin base class for ECG management system integration (US-118).

Defines :class:`ECGSystemPlugin`, the abstract contract every ECG-system
plugin implements, plus the shared event-hook and orchestration machinery
that ties polling → inference → result submission together.

A plugin adapts a specific ECG management platform (GE MUSE, a FHIR server,
a watched directory, …) to a uniform interface so the daemon
(``aortica plugin run``) can drive any of them identically::

    plugin.connect(config)
    while running:
        plugin.process_once(processor)   # poll, infer, submit, fire hooks
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Event names fired during processing.
ON_ECG_RECEIVED = "on_ecg_received"
ON_RESULT_GENERATED = "on_result_generated"
ON_CRITICAL_FINDING = "on_critical_finding"

_VALID_EVENTS = (ON_ECG_RECEIVED, ON_RESULT_GENERATED, ON_CRITICAL_FINDING)

#: A processor turns a polled ECG (record or raw payload) into a result dict
#: (typically ``{task: {class: confidence}}``).
Processor = Callable[[Any], Dict[str, Any]]


@dataclass
class PluginHealth:
    """Result of a plugin health check."""

    healthy: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"healthy": self.healthy, "detail": self.detail}


@dataclass
class ProcessedECG:
    """Bookkeeping for one ECG processed in a cycle."""

    ecg_id: str
    result: Dict[str, Any]
    critical: bool = False
    submitted: bool = False


@dataclass
class ProcessReport:
    """Summary of a single :meth:`ECGSystemPlugin.process_once` cycle."""

    polled: int = 0
    processed: List[ProcessedECG] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for p in self.processed if p.critical)


class ECGSystemPlugin(ABC):
    """Abstract base class for an ECG management system plugin.

    Subclasses implement the four connection/data methods; the base class
    provides event-hook registration and the :meth:`process_once`
    orchestration shared by all plugins.
    """

    #: Registry name; subclasses should override.
    name: str = "base"

    def __init__(self) -> None:
        self._hooks: Dict[str, List[Callable[..., None]]] = {
            e: [] for e in _VALID_EVENTS
        }
        self._connected = False

    # -- Event hooks --------------------------------------------------------

    def register_hook(self, event: str, callback: Callable[..., None]) -> None:
        """Register *callback* for *event* (one of the ``on_*`` names)."""
        if event not in _VALID_EVENTS:
            raise ValueError(
                f"Unknown event {event!r}. Valid: {_VALID_EVENTS}"
            )
        self._hooks[event].append(callback)

    def _fire(self, event: str, *args: Any) -> None:
        """Invoke every callback registered for *event* (errors isolated)."""
        for cb in self._hooks.get(event, []):
            try:
                cb(*args)
            except Exception:  # noqa: BLE001 - a hook must not break the loop
                logger.exception("Hook %s raised for event %s", cb, event)

    # -- Abstract contract --------------------------------------------------

    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> None:
        """Establish a connection / initialise from *config*."""

    @abstractmethod
    def poll_for_ecgs(self) -> List[Tuple[str, Any]]:
        """Return newly available ``(ecg_id, payload)`` pairs.

        ``payload`` is whatever the processor consumes — typically an
        :class:`~aortica.io.ecg_record.ECGRecord`.
        """

    @abstractmethod
    def submit_result(self, ecg_id: str, result: Dict[str, Any]) -> None:
        """Write an inference *result* back to the source system."""

    @abstractmethod
    def get_worklist(self) -> List[Dict[str, Any]]:
        """Return the current pending worklist as plain dicts."""

    @abstractmethod
    def health_check(self) -> PluginHealth:
        """Return connectivity/health status."""

    # -- Orchestration ------------------------------------------------------

    def process_once(
        self,
        processor: Processor,
        *,
        critical_detector: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> ProcessReport:
        """Poll, process, and submit one batch of ECGs, firing hooks.

        For each polled ECG the cycle fires ``on_ecg_received``, runs
        *processor* to obtain a result, fires ``on_result_generated``,
        detects critical findings (firing ``on_critical_finding``), and
        submits the result back to the source system.  Per-ECG errors are
        collected without aborting the batch.

        Args:
            processor: Callable mapping a polled payload to a result dict.
            critical_detector: Optional predicate flagging a result as
                critical.  Defaults to :func:`default_critical_detector`.

        Returns:
            A :class:`ProcessReport` summarising the cycle.
        """
        detector = critical_detector or default_critical_detector
        report = ProcessReport()

        try:
            polled = self.poll_for_ecgs()
        except Exception as exc:  # noqa: BLE001
            logger.exception("poll_for_ecgs failed")
            report.errors.append({"stage": "poll", "error": str(exc)})
            return report

        report.polled = len(polled)
        for ecg_id, payload in polled:
            try:
                self._fire(ON_ECG_RECEIVED, ecg_id, payload)
                result = processor(payload)
                self._fire(ON_RESULT_GENERATED, ecg_id, result)

                critical = detector(result)
                if critical:
                    self._fire(ON_CRITICAL_FINDING, ecg_id, result)

                self.submit_result(ecg_id, result)
                report.processed.append(
                    ProcessedECG(
                        ecg_id=ecg_id,
                        result=result,
                        critical=critical,
                        submitted=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Processing ECG %s failed", ecg_id)
                report.errors.append({"ecg_id": ecg_id, "error": str(exc)})

        return report


# ---------------------------------------------------------------------------
# Default critical-finding detector
# ---------------------------------------------------------------------------

#: Conditions that, above threshold, mark a result critical.
_CRITICAL_CONDITIONS = {
    "STEMI",
    "VT",
    "VF",
    "occlusive_NSTEMI",
    "hyperkalaemia",
    "hyperkalaemia_severity_grade",
    "brugada_pattern",
    "de_Winter_T_wave",
    "Wellens_syndrome",
}


def default_critical_detector(
    result: Dict[str, Any], threshold: float = 0.5
) -> bool:
    """Return ``True`` if *result* contains a critical finding above threshold.

    Accepts a ``{task: {class: confidence}}`` mapping.
    """
    for task_preds in result.values():
        if not isinstance(task_preds, dict):
            continue
        for class_name, confidence in task_preds.items():
            try:
                conf = float(confidence)
            except (TypeError, ValueError):
                continue
            if class_name in _CRITICAL_CONDITIONS and conf >= threshold:
                return True
    return False
