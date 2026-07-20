"""FHIRPlugin — FHIR R4 server integration (US-118).

Polls a FHIR server for ECG Observations and submits results back as
DiagnosticReport resources (wraps US-080's ``to_diagnostic_report``).
Uses stdlib ``urllib`` so no extra HTTP dependency is required.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from aortica.integration.plugins.base import ECGSystemPlugin, PluginHealth

logger = logging.getLogger(__name__)

#: Signature of an injectable HTTP transport ``(method, url, body) -> (status, json)``.
HttpFn = Callable[[str, str, Optional[Dict[str, Any]]], Tuple[int, Any]]


def _default_http(
    method: str, url: str, body: Optional[Dict[str, Any]]
) -> Tuple[int, Any]:
    import urllib.request

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/fhir+json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
        parsed = json.loads(raw) if raw else {}
        return int(resp.status), parsed


class FHIRPlugin(ECGSystemPlugin):
    """Poll a FHIR server for ECGs and submit DiagnosticReport results.

    Config keys:
        fhir_server_url: Base URL of the FHIR R4 server (required).
        search_query: Optional query string for the ECG search
            (default ``Observation?category=procedure&code=11524-6``).
        http: Optional injectable HTTP transport for testing.
    """

    name = "fhir"

    def __init__(self) -> None:
        super().__init__()
        self._base: str = ""
        self._search_query: str = ""
        self._http: HttpFn = _default_http
        self._seen: set[str] = set()

    def connect(self, config: Dict[str, Any]) -> None:
        base = config.get("fhir_server_url")
        if not base:
            raise ValueError("FHIRPlugin requires 'fhir_server_url'")
        self._base = base.rstrip("/")
        self._search_query = config.get(
            "search_query", "Observation?category=procedure&code=11524-6"
        )
        if "http" in config and callable(config["http"]):
            self._http = config["http"]
        self._connected = True

    def poll_for_ecgs(self) -> List[Tuple[str, Any]]:
        url = f"{self._base}/{self._search_query}"
        status, bundle = self._http("GET", url, None)
        if status < 200 or status >= 300:
            raise RuntimeError(f"FHIR search failed: HTTP {status}")
        out: List[Tuple[str, Any]] = []
        for entry in (bundle or {}).get("entry", []):
            resource = entry.get("resource", {})
            res_id = str(resource.get("id", ""))
            if not res_id or res_id in self._seen:
                continue
            self._seen.add(res_id)
            out.append((res_id, resource))
        return out

    def submit_result(self, ecg_id: str, result: Dict[str, Any]) -> None:
        from aortica.integration.fhir import to_diagnostic_report

        output = to_diagnostic_report(result, patient_ref=None)
        payload = json.loads(output.bundle_json) if output.bundle_json else {}
        status, _ = self._http("POST", f"{self._base}/DiagnosticReport", payload)
        if status < 200 or status >= 300:
            raise RuntimeError(f"DiagnosticReport POST failed: HTTP {status}")

    def get_worklist(self) -> List[Dict[str, Any]]:
        return [{"ecg_id": rid, "status": "seen"} for rid in sorted(self._seen)]

    def health_check(self) -> PluginHealth:
        try:
            status, _ = self._http("GET", f"{self._base}/metadata", None)
            if 200 <= status < 300:
                return PluginHealth(True, "capability statement ok")
            return PluginHealth(False, f"metadata HTTP {status}")
        except Exception as exc:  # noqa: BLE001
            return PluginHealth(False, f"metadata failed: {exc}")
