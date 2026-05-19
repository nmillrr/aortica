"""DICOM DIMSE C-STORE / C-FIND client for ECG management system integration.

Provides a high-level ``DicomClient`` that wraps ``pynetdicom`` for
sending DICOM datasets (ECG waveforms, Structured Reports) to remote
Application Entities and querying for ECG studies via C-FIND.

Designed for interoperability with GE MUSE-style ECG management systems
and other DICOM-compliant PACS/archives.

Requires ``pynetdicom``::

    pip install pynetdicom

Example usage::

    from aortica.integration.dimse import DicomClient

    client = DicomClient(local_ae_title="AORTICA", local_port=11113)

    # Store a DICOM SR
    result = client.c_store(
        dataset=sr_dataset,
        remote_ae="MUSE_AE",
        remote_host="192.168.1.100",
        remote_port=4006,
    )

    # Query for ECG studies
    matches = client.c_find(
        query={"PatientID": "12345", "Modality": "ECG"},
        remote_ae="MUSE_AE",
        remote_host="192.168.1.100",
        remote_port=4006,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import UID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Well-known SOP Class UIDs for presentation context negotiation
# ---------------------------------------------------------------------------

# ECG Waveform Storage
_12_LEAD_ECG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.9.1.1"
_GENERAL_ECG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.9.1.2"
_AMBULATORY_ECG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.9.1.3"
_HEMODYNAMIC_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.9.2.1"

# SR Storage
_COMPREHENSIVE_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.33"
_ENHANCED_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.22"
_BASIC_TEXT_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.11"

# Verification
_VERIFICATION_SOP_CLASS = "1.2.840.10008.1.1"

# Query/Retrieve — Study Root
_STUDY_ROOT_QR_FIND = "1.2.840.10008.5.1.4.1.2.2.1"
_STUDY_ROOT_QR_MOVE = "1.2.840.10008.5.1.4.1.2.2.2"
_STUDY_ROOT_QR_GET = "1.2.840.10008.5.1.4.1.2.2.3"

# Patient Root
_PATIENT_ROOT_QR_FIND = "1.2.840.10008.5.1.4.1.2.1.1"

# Transfer Syntaxes
_IMPLICIT_VR_LE = "1.2.840.10008.1.2"
_EXPLICIT_VR_LE = "1.2.840.10008.1.2.1"
_EXPLICIT_VR_BE = "1.2.840.10008.1.2.2"

# Default supported transfer syntaxes for negotiation
_DEFAULT_TRANSFER_SYNTAXES = [
    _EXPLICIT_VR_LE,
    _IMPLICIT_VR_LE,
]

# Storage SOP classes supported by default
_DEFAULT_STORE_SOP_CLASSES = [
    _12_LEAD_ECG_SOP_CLASS,
    _GENERAL_ECG_SOP_CLASS,
    _AMBULATORY_ECG_SOP_CLASS,
    _COMPREHENSIVE_SR_SOP_CLASS,
    _ENHANCED_SR_SOP_CLASS,
    _BASIC_TEXT_SR_SOP_CLASS,
]

# Query SOP classes
_DEFAULT_FIND_SOP_CLASSES = [
    _STUDY_ROOT_QR_FIND,
    _PATIENT_ROOT_QR_FIND,
]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CStoreResult:
    """Result of a C-STORE operation.

    Attributes:
        success: Whether the C-STORE was accepted by the remote SCP.
        status_code: DICOM status code returned by the SCP.
        status_category: Human-readable status category
            (``'Success'``, ``'Warning'``, ``'Failure'``, etc.).
        status_description: Detailed description of the status.
        sop_instance_uid: SOP Instance UID of the stored dataset.
        error_message: Error description if the operation failed.
    """

    success: bool = False
    status_code: int = 0
    status_category: str = ""
    status_description: str = ""
    sop_instance_uid: str = ""
    error_message: str = ""


@dataclass
class CStoreMultiResult:
    """Result of storing multiple datasets.

    Attributes:
        results: Per-dataset C-STORE results.
        total: Total number of datasets submitted.
        succeeded: Number of successful stores.
        failed: Number of failed stores.
    """

    results: List[CStoreResult] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0


@dataclass
class CFindResult:
    """Result of a C-FIND query.

    Attributes:
        success: Whether the C-FIND completed without error.
        matches: List of matching DICOM datasets (study-level).
        match_count: Number of matches found.
        error_message: Error description if the operation failed.
    """

    success: bool = False
    matches: List[Dict[str, Any]] = field(default_factory=list)
    match_count: int = 0
    error_message: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_category(status_code: int) -> str:
    """Map a DICOM status code to a human-readable category."""
    if status_code == 0x0000:
        return "Success"
    elif 0xB000 <= status_code <= 0xBFFF:
        return "Warning"
    elif status_code == 0xFF00 or status_code == 0xFF01:
        return "Pending"
    elif 0xC000 <= status_code <= 0xCFFF:
        return "Failure"
    elif 0xA000 <= status_code <= 0xAFFF:
        return "Failure"
    elif status_code == 0xFE00:
        return "Cancel"
    else:
        return "Unknown"


def _dataset_to_dict(ds: Dataset) -> Dict[str, Any]:
    """Convert a pydicom Dataset to a plain dict for serialisation."""
    result: Dict[str, Any] = {}
    for elem in ds:
        if elem.VR == "SQ":
            # Skip sequences for simplicity
            continue
        keyword = elem.keyword
        if keyword:
            result[keyword] = str(elem.value)
    return result


def _build_find_dataset(
    query: Dict[str, str],
    query_retrieve_level: str = "STUDY",
) -> Dataset:
    """Build a C-FIND identifier dataset from a simple query dict.

    Maps common field names to DICOM tags.  Unrecognised keys are
    silently skipped.

    Args:
        query: Search criteria.  Supported keys include:
            ``PatientID``, ``PatientName``, ``StudyDate``,
            ``Modality``, ``AccessionNumber``,
            ``StudyInstanceUID``, ``StudyDescription``,
            ``SeriesInstanceUID``, ``SOPInstanceUID``.
        query_retrieve_level: DICOM Q/R level.  Default ``STUDY``.

    Returns:
        A ``pydicom.Dataset`` suitable for C-FIND.
    """
    ds = Dataset()
    ds.QueryRetrieveLevel = query_retrieve_level

    # Map friendly names to DICOM keywords (all are standard keywords)
    _SUPPORTED_KEYS = {
        "PatientID",
        "PatientName",
        "StudyDate",
        "Modality",
        "AccessionNumber",
        "StudyInstanceUID",
        "StudyDescription",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "ReferringPhysicianName",
        "StudyID",
        "NumberOfStudyRelatedSeries",
        "NumberOfStudyRelatedInstances",
    }

    for key, value in query.items():
        if key in _SUPPORTED_KEYS:
            setattr(ds, key, value)
        else:
            logger.debug("Skipping unsupported C-FIND key: %s", key)

    # Always request return keys that aren't already set
    _RETURN_KEYS = [
        "PatientID",
        "PatientName",
        "StudyDate",
        "StudyInstanceUID",
        "Modality",
        "AccessionNumber",
        "StudyDescription",
    ]
    for rk in _RETURN_KEYS:
        if not hasattr(ds, rk):
            setattr(ds, rk, "")

    return ds


# ---------------------------------------------------------------------------
# DicomClient
# ---------------------------------------------------------------------------


class DicomClient:
    """High-level DICOM DIMSE client for C-STORE and C-FIND operations.

    Wraps ``pynetdicom`` to provide a simple interface for sending
    DICOM datasets (ECG waveforms, Structured Reports) to remote
    Application Entities and querying for ECG studies.

    Args:
        local_ae_title: AE title for this client.  Default ``AORTICA``.
        local_port: Local port for this client.  Default ``11113``.
        timeout: Association and DIMSE timeout in seconds.
            Default ``30``.
        additional_sop_classes: Extra SOP classes to negotiate
            for C-STORE beyond the defaults.

    Example::

        client = DicomClient(local_ae_title="AORTICA")
        result = client.c_store(
            dataset=sr_dataset,
            remote_ae="PACS",
            remote_host="10.0.0.1",
            remote_port=104,
        )
    """

    def __init__(
        self,
        local_ae_title: str = "AORTICA",
        local_port: int = 11113,
        timeout: int = 30,
        additional_sop_classes: Optional[Sequence[str]] = None,
    ) -> None:
        self.local_ae_title = local_ae_title
        self.local_port = local_port
        self.timeout = timeout
        self._additional_sop_classes = list(additional_sop_classes or [])

    # ---------------------------------------------------------------
    # C-STORE
    # ---------------------------------------------------------------

    def c_store(
        self,
        dataset: Dataset,
        remote_ae: str,
        remote_host: str,
        remote_port: int,
    ) -> CStoreResult:
        """Send a DICOM dataset to a remote SCP via C-STORE.

        Establishes a DIMSE association with the remote AE,
        negotiates presentation contexts for the dataset's
        SOP Class, and sends the dataset.

        Args:
            dataset: The ``pydicom.Dataset`` to store.  Must have
                ``SOPClassUID`` and ``SOPInstanceUID`` set.
            remote_ae: Called AE title of the remote SCP.
            remote_host: Hostname or IP of the remote SCP.
            remote_port: TCP port of the remote SCP.

        Returns:
            :class:`CStoreResult` with status information.
        """
        from pynetdicom import AE  # type: ignore[import-untyped]

        sop_instance_uid = str(getattr(dataset, "SOPInstanceUID", ""))
        sop_class_uid = str(getattr(dataset, "SOPClassUID", ""))

        if not sop_class_uid:
            return CStoreResult(
                success=False,
                error_message="Dataset has no SOPClassUID",
                sop_instance_uid=sop_instance_uid,
            )

        ae = AE(ae_title=self.local_ae_title)
        ae.acse_timeout = self.timeout
        ae.dimse_timeout = self.timeout
        ae.network_timeout = self.timeout

        # Build presentation contexts for this specific SOP class
        # plus defaults for common ECG/SR classes
        sop_classes = set(_DEFAULT_STORE_SOP_CLASSES)
        sop_classes.update(self._additional_sop_classes)
        sop_classes.add(sop_class_uid)

        for sop_uid in sop_classes:
            ae.add_requested_context(sop_uid, _DEFAULT_TRANSFER_SYNTAXES)

        logger.info(
            "C-STORE: Associating with %s@%s:%d",
            remote_ae,
            remote_host,
            remote_port,
        )

        try:
            assoc = ae.associate(
                remote_host,
                remote_port,
                ae_title=remote_ae,
            )
        except Exception as exc:
            return CStoreResult(
                success=False,
                error_message=f"Association failed: {exc}",
                sop_instance_uid=sop_instance_uid,
            )

        if not assoc.is_established:
            return CStoreResult(
                success=False,
                error_message="Association rejected or aborted by remote AE",
                sop_instance_uid=sop_instance_uid,
            )

        try:
            status = assoc.send_c_store(dataset)

            if status is None:
                return CStoreResult(
                    success=False,
                    error_message="No response from remote SCP (connection dropped?)",
                    sop_instance_uid=sop_instance_uid,
                )

            status_code = int(status.Status)
            category = _status_category(status_code)
            success = status_code == 0x0000

            return CStoreResult(
                success=success,
                status_code=status_code,
                status_category=category,
                status_description=f"0x{status_code:04X} ({category})",
                sop_instance_uid=sop_instance_uid,
            )

        except Exception as exc:
            return CStoreResult(
                success=False,
                error_message=f"C-STORE operation failed: {exc}",
                sop_instance_uid=sop_instance_uid,
            )
        finally:
            assoc.release()

    def c_store_batch(
        self,
        datasets: Sequence[Dataset],
        remote_ae: str,
        remote_host: str,
        remote_port: int,
    ) -> CStoreMultiResult:
        """Send multiple DICOM datasets to a remote SCP via C-STORE.

        Opens a single association for all datasets.

        Args:
            datasets: Sequence of ``pydicom.Dataset`` to store.
            remote_ae: Called AE title of the remote SCP.
            remote_host: Hostname or IP of the remote SCP.
            remote_port: TCP port of the remote SCP.

        Returns:
            :class:`CStoreMultiResult` with per-dataset results.
        """
        from pynetdicom import AE  # type: ignore[import-untyped]

        multi_result = CStoreMultiResult(total=len(datasets))
        if not datasets:
            multi_result.success = True  # type: ignore[attr-defined]
            return multi_result

        ae = AE(ae_title=self.local_ae_title)
        ae.acse_timeout = self.timeout
        ae.dimse_timeout = self.timeout
        ae.network_timeout = self.timeout

        sop_classes = set(_DEFAULT_STORE_SOP_CLASSES)
        sop_classes.update(self._additional_sop_classes)
        for ds in datasets:
            sop_uid = str(getattr(ds, "SOPClassUID", ""))
            if sop_uid:
                sop_classes.add(sop_uid)

        for sop_uid in sop_classes:
            ae.add_requested_context(sop_uid, _DEFAULT_TRANSFER_SYNTAXES)

        try:
            assoc = ae.associate(
                remote_host,
                remote_port,
                ae_title=remote_ae,
            )
        except Exception as exc:
            for ds in datasets:
                multi_result.results.append(
                    CStoreResult(
                        success=False,
                        error_message=f"Association failed: {exc}",
                        sop_instance_uid=str(getattr(ds, "SOPInstanceUID", "")),
                    )
                )
            multi_result.failed = len(datasets)
            return multi_result

        if not assoc.is_established:
            for ds in datasets:
                multi_result.results.append(
                    CStoreResult(
                        success=False,
                        error_message="Association rejected by remote AE",
                        sop_instance_uid=str(getattr(ds, "SOPInstanceUID", "")),
                    )
                )
            multi_result.failed = len(datasets)
            return multi_result

        try:
            for ds in datasets:
                sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
                try:
                    status = assoc.send_c_store(ds)
                    if status is None:
                        result = CStoreResult(
                            success=False,
                            error_message="No response from SCP",
                            sop_instance_uid=sop_uid,
                        )
                    else:
                        code = int(status.Status)
                        cat = _status_category(code)
                        result = CStoreResult(
                            success=code == 0x0000,
                            status_code=code,
                            status_category=cat,
                            status_description=f"0x{code:04X} ({cat})",
                            sop_instance_uid=sop_uid,
                        )
                except Exception as exc:
                    result = CStoreResult(
                        success=False,
                        error_message=f"C-STORE failed: {exc}",
                        sop_instance_uid=sop_uid,
                    )

                multi_result.results.append(result)
                if result.success:
                    multi_result.succeeded += 1
                else:
                    multi_result.failed += 1
        finally:
            assoc.release()

        return multi_result

    # ---------------------------------------------------------------
    # C-FIND
    # ---------------------------------------------------------------

    def c_find(
        self,
        query: Dict[str, str],
        remote_ae: str,
        remote_host: str,
        remote_port: int,
        query_model: str = "study",
    ) -> CFindResult:
        """Query a remote SCP for ECG studies via C-FIND.

        Args:
            query: Dictionary of search criteria.  Supported keys:
                ``PatientID``, ``PatientName``, ``StudyDate``,
                ``Modality``, ``AccessionNumber``,
                ``StudyInstanceUID``, ``StudyDescription``.
                Wildcard matching uses ``*`` (e.g.
                ``{"PatientName": "DOE*"}``).
            remote_ae: Called AE title of the remote SCP.
            remote_host: Hostname or IP of the remote SCP.
            remote_port: TCP port of the remote SCP.
            query_model: Query/Retrieve information model.
                ``"study"`` (Study Root, default) or ``"patient"``
                (Patient Root).

        Returns:
            :class:`CFindResult` with matching studies.
        """
        from pynetdicom import AE  # type: ignore[import-untyped]

        ae = AE(ae_title=self.local_ae_title)
        ae.acse_timeout = self.timeout
        ae.dimse_timeout = self.timeout
        ae.network_timeout = self.timeout

        # Select SOP class for Q/R model
        if query_model == "patient":
            find_sop = _PATIENT_ROOT_QR_FIND
            qr_level = "STUDY"
        else:
            find_sop = _STUDY_ROOT_QR_FIND
            qr_level = "STUDY"

        ae.add_requested_context(find_sop, _DEFAULT_TRANSFER_SYNTAXES)

        identifier = _build_find_dataset(query, qr_level)

        logger.info(
            "C-FIND: Querying %s@%s:%d with %d criteria",
            remote_ae,
            remote_host,
            remote_port,
            len(query),
        )

        try:
            assoc = ae.associate(
                remote_host,
                remote_port,
                ae_title=remote_ae,
            )
        except Exception as exc:
            return CFindResult(
                success=False,
                error_message=f"Association failed: {exc}",
            )

        if not assoc.is_established:
            return CFindResult(
                success=False,
                error_message="Association rejected or aborted by remote AE",
            )

        matches: List[Dict[str, Any]] = []

        try:
            responses = assoc.send_c_find(identifier, find_sop)

            for status, ds in responses:
                if status is None:
                    return CFindResult(
                        success=False,
                        matches=matches,
                        match_count=len(matches),
                        error_message="Connection lost during C-FIND",
                    )

                status_code = int(status.Status)

                # Pending — a match was returned
                if status_code in (0xFF00, 0xFF01):
                    if ds is not None:
                        matches.append(_dataset_to_dict(ds))

                # Success — query complete
                elif status_code == 0x0000:
                    break

                # Error
                else:
                    cat = _status_category(status_code)
                    return CFindResult(
                        success=False,
                        matches=matches,
                        match_count=len(matches),
                        error_message=f"C-FIND error: 0x{status_code:04X} ({cat})",
                    )

        except Exception as exc:
            return CFindResult(
                success=False,
                matches=matches,
                match_count=len(matches),
                error_message=f"C-FIND operation failed: {exc}",
            )
        finally:
            assoc.release()

        return CFindResult(
            success=True,
            matches=matches,
            match_count=len(matches),
        )

    # ---------------------------------------------------------------
    # Echo / Verification
    # ---------------------------------------------------------------

    def verify(
        self,
        remote_ae: str,
        remote_host: str,
        remote_port: int,
    ) -> bool:
        """Send C-ECHO to verify connectivity with a remote AE.

        Args:
            remote_ae: Called AE title.
            remote_host: Hostname or IP.
            remote_port: TCP port.

        Returns:
            ``True`` if the remote AE responds successfully.
        """
        from pynetdicom import AE  # type: ignore[import-untyped]

        ae = AE(ae_title=self.local_ae_title)
        ae.acse_timeout = self.timeout
        ae.dimse_timeout = self.timeout
        ae.network_timeout = self.timeout

        ae.add_requested_context(_VERIFICATION_SOP_CLASS)

        try:
            assoc = ae.associate(
                remote_host,
                remote_port,
                ae_title=remote_ae,
            )
        except Exception:
            return False

        if not assoc.is_established:
            return False

        try:
            status = assoc.send_c_echo()
            if status is None:
                return False
            return int(status.Status) == 0x0000
        except Exception:
            return False
        finally:
            assoc.release()
