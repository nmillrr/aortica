"""MusePlugin — GE MUSE (DICOM DIMSE) integration (US-118).

Wraps the DIMSE client (US-083): C-FIND to discover new ECG studies and
C-STORE to write DICOM SR results back to the archive (US-082).

Waveform retrieval (C-MOVE/C-GET) is out of scope for US-083, so
:meth:`poll_for_ecgs` returns study references; a deployment supplies the
actual retrieval + inference through the processor.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from aortica.integration.plugins.base import ECGSystemPlugin, PluginHealth

logger = logging.getLogger(__name__)


class MusePlugin(ECGSystemPlugin):
    """Poll a MUSE/DICOM archive via C-FIND and write results via C-STORE.

    Config keys:
        remote_ae, remote_host, remote_port: Remote SCP coordinates.
        local_ae_title, local_port: Local AE identity (optional).
        query: Optional C-FIND query dict (defaults to Modality=ECG).
    """

    name = "muse"

    def __init__(self) -> None:
        super().__init__()
        self._client: Any = None
        self._remote: Dict[str, Any] = {}
        self._query: Dict[str, str] = {}
        self._seen: set[str] = set()

    def connect(self, config: Dict[str, Any]) -> None:
        required = ("remote_ae", "remote_host", "remote_port")
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"MusePlugin missing config keys: {missing}")

        from aortica.integration.dimse import DicomClient

        self._client = DicomClient(
            local_ae_title=config.get("local_ae_title", "AORTICA"),
            local_port=int(config.get("local_port", 11113)),
            timeout=int(config.get("timeout", 30)),
        )
        self._remote = {
            "remote_ae": config["remote_ae"],
            "remote_host": config["remote_host"],
            "remote_port": int(config["remote_port"]),
        }
        self._query = config.get("query", {"Modality": "ECG"})
        self._connected = True

    def poll_for_ecgs(self) -> List[Tuple[str, Any]]:
        if self._client is None:
            raise RuntimeError("MusePlugin.connect() not called")
        find = self._client.c_find(self._query, **self._remote)
        studies = getattr(find, "matches", None) or getattr(find, "studies", [])
        out: List[Tuple[str, Any]] = []
        for study in studies:
            uid = str(study.get("StudyInstanceUID", study.get("SOPInstanceUID", "")))
            if not uid or uid in self._seen:
                continue
            self._seen.add(uid)
            out.append((uid, study))
        return out

    def submit_result(self, ecg_id: str, result: Dict[str, Any]) -> None:
        if self._client is None:
            raise RuntimeError("MusePlugin.connect() not called")
        from aortica.integration.dicom_sr import to_structured_report

        sr = to_structured_report(result, original_dicom_ref=ecg_id)
        self._client.c_store(sr, **self._remote)

    def get_worklist(self) -> List[Dict[str, Any]]:
        return [
            {"ecg_id": uid, "status": "seen"} for uid in sorted(self._seen)
        ]

    def health_check(self) -> PluginHealth:
        if self._client is None:
            return PluginHealth(False, "not connected")
        try:
            self._client.c_find(self._query, **self._remote)
            return PluginHealth(True, "c-find ok")
        except Exception as exc:  # noqa: BLE001
            return PluginHealth(False, f"c-find failed: {exc}")
