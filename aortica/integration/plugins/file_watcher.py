"""FileWatcherPlugin — directory-watching ECG ingestion (US-118).

Watches a configurable directory for new ECG files (SCP-ECG, WFDB, CSV,
etc.), reads each via the universal dispatcher, hands it to a processor,
writes the resulting JSON to an output directory, and (optionally) moves
the source file to a ``processed`` directory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from aortica.integration.plugins.base import ECGSystemPlugin, PluginHealth

logger = logging.getLogger(__name__)


class FileWatcherPlugin(ECGSystemPlugin):
    """Poll a directory for new ECG files and process them automatically.

    Config keys:
        watch_dir: Directory to scan for new ECG files (required).
        output_dir: Directory for result JSON files (required).
        processed_dir: Optional directory to move source files into after
            processing.  If omitted, files stay in place and are tracked as
            seen so they are not reprocessed.
        csv_config: Optional dict forwarded to the CSV reader (CSV needs a
            sample_rate).
    """

    name = "file_watcher"

    def __init__(self) -> None:
        super().__init__()
        self.watch_dir: Path | None = None
        self.output_dir: Path | None = None
        self.processed_dir: Path | None = None
        self._csv_config: Dict[str, Any] | None = None
        self._seen: set[str] = set()

    # -- Contract -----------------------------------------------------------

    def connect(self, config: Dict[str, Any]) -> None:
        watch = config.get("watch_dir")
        output = config.get("output_dir")
        if not watch or not output:
            raise ValueError(
                "FileWatcherPlugin requires 'watch_dir' and 'output_dir'"
            )
        self.watch_dir = Path(watch)
        self.output_dir = Path(output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        processed = config.get("processed_dir")
        if processed:
            self.processed_dir = Path(processed)
            self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._csv_config = config.get("csv_config")
        self._connected = True

    def _candidate_files(self) -> List[Path]:
        from aortica.io.dispatcher import _EXTENSION_MAP

        assert self.watch_dir is not None
        if not self.watch_dir.exists():
            return []
        files: List[Path] = []
        for p in sorted(self.watch_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in _EXTENSION_MAP:
                continue
            # WFDB: enter via the .hea header; skip the companion .dat.
            if p.suffix.lower() == ".dat" and p.with_suffix(".hea").exists():
                continue
            if str(p) in self._seen:
                continue
            files.append(p)
        return files

    def poll_for_ecgs(self) -> List[Tuple[str, Any]]:
        from aortica.io.dispatcher import read_ecg

        out: List[Tuple[str, Any]] = []
        for path in self._candidate_files():
            ecg_id = path.stem
            try:
                if self._csv_config and path.suffix.lower() in (".csv", ".tsv"):
                    from aortica.io.csv_reader import CSVConfig

                    record = read_ecg(
                        path, csv_config=CSVConfig(**self._csv_config)
                    )
                else:
                    record = read_ecg(path)
            except Exception:  # noqa: BLE001 - skip unreadable, mark seen
                logger.exception("Failed to read %s", path)
                self._seen.add(str(path))
                continue
            self._seen.add(str(path))
            out.append((ecg_id, record))
        return out

    def submit_result(self, ecg_id: str, result: Dict[str, Any]) -> None:
        assert self.output_dir is not None
        out_path = self.output_dir / f"{ecg_id}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))

        # Move the source file into processed_dir, if configured.
        if self.processed_dir is not None and self.watch_dir is not None:
            for ext in (".hea", ".dat"):
                src = self.watch_dir / f"{ecg_id}{ext}"
                if src.exists():
                    src.rename(self.processed_dir / src.name)
            for src in self.watch_dir.glob(f"{ecg_id}.*"):
                if src.exists():
                    src.rename(self.processed_dir / src.name)

    def get_worklist(self) -> List[Dict[str, Any]]:
        return [
            {"ecg_id": p.stem, "path": str(p), "status": "pending"}
            for p in self._candidate_files()
        ]

    def health_check(self) -> PluginHealth:
        if self.watch_dir is None or self.output_dir is None:
            return PluginHealth(False, "not connected")
        if not self.watch_dir.exists():
            return PluginHealth(False, f"watch_dir missing: {self.watch_dir}")
        if not os.access(self.output_dir, os.W_OK):
            return PluginHealth(False, f"output_dir not writable: {self.output_dir}")
        return PluginHealth(True, "ok")
