"""ECG format readers (WFDB, DICOM, SCP-ECG, CSV, MAT, HL7 aECG, XML)."""

from aortica.io.csv_reader import CSVConfig, read_csv
from aortica.io.dicom_reader import read_dicom
from aortica.io.ecg_record import ECGRecord
from aortica.io.hl7_aecg_reader import read_hl7_aecg
from aortica.io.mat_reader import MATConfig, read_mat
from aortica.io.scp_reader import read_scp
from aortica.io.wfdb_reader import read_wfdb

__all__ = [
    "CSVConfig",
    "ECGRecord",
    "MATConfig",
    "read_csv",
    "read_dicom",
    "read_hl7_aecg",
    "read_mat",
    "read_scp",
    "read_wfdb",
]
