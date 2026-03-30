"""Type stubs for generated protobuf module ``ecg_service_pb2``.

Auto-generated protobuf modules use dynamic attribute creation, which
mypy cannot resolve.  This stub provides the minimal type information
needed for downstream type checking.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

class LeadQuality:
    lead_name: str
    score: float
    classification: str
    flags: List[str]
    def __init__(
        self,
        *,
        lead_name: str = ...,
        score: float = ...,
        classification: str = ...,
        flags: Optional[Sequence[str]] = ...,
    ) -> None: ...

class QualityReport:
    per_lead: List[LeadQuality]
    overall_score: float
    overall_classification: str
    recommendation: str
    def __init__(
        self,
        *,
        per_lead: Optional[Sequence[LeadQuality]] = ...,
        overall_score: float = ...,
        overall_classification: str = ...,
        recommendation: str = ...,
    ) -> None: ...

class TaskPrediction:
    task: str
    class_names: List[str]
    probabilities: List[float]
    def __init__(
        self,
        *,
        task: str = ...,
        class_names: Optional[Sequence[str]] = ...,
        probabilities: Optional[Sequence[float]] = ...,
    ) -> None: ...

class PredictionSet:
    indices: List[int]
    def __init__(self, *, indices: Optional[Sequence[int]] = ...) -> None: ...

class ConfidenceInterval:
    lower: List[float]
    upper: List[float]
    def __init__(
        self,
        *,
        lower: Optional[Sequence[float]] = ...,
        upper: Optional[Sequence[float]] = ...,
    ) -> None: ...

class UncertaintyReport:
    prediction_sets: Dict[str, PredictionSet]
    confidence_intervals: Dict[str, ConfidenceInterval]
    ood_flag: bool
    entropy_score: float
    def __init__(
        self,
        *,
        prediction_sets: Optional[Mapping[str, PredictionSet]] = ...,
        confidence_intervals: Optional[Mapping[str, ConfidenceInterval]] = ...,
        ood_flag: bool = ...,
        entropy_score: float = ...,
    ) -> None: ...

class PredictRequest:
    file_data: bytes
    filename: str
    format_override: str
    def __init__(
        self,
        *,
        file_data: bytes = ...,
        filename: str = ...,
        format_override: str = ...,
    ) -> None: ...
    def SerializeToString(self) -> bytes: ...
    def ParseFromString(self, s: bytes) -> int: ...

class PredictBatchRequest:
    files: List[PredictRequest]
    def __init__(
        self,
        *,
        files: Optional[Sequence[PredictRequest]] = ...,
    ) -> None: ...
    def SerializeToString(self) -> bytes: ...
    @staticmethod
    def FromString(s: bytes) -> "PredictBatchRequest": ...

class PredictReply:
    quality_report: QualityReport
    predictions: List[TaskPrediction]
    uncertainty: UncertaintyReport
    def __init__(
        self,
        *,
        quality_report: Optional[QualityReport] = ...,
        predictions: Optional[Sequence[TaskPrediction]] = ...,
        uncertainty: Optional[UncertaintyReport] = ...,
    ) -> None: ...
    def SerializeToString(self) -> bytes: ...
    @staticmethod
    def FromString(s: bytes) -> "PredictReply": ...

class BatchFileResult:
    filename: str
    status: str
    error: str
    result: PredictReply
    def __init__(
        self,
        *,
        filename: str = ...,
        status: str = ...,
        error: str = ...,
        result: Optional[PredictReply] = ...,
    ) -> None: ...

class PredictBatchReply:
    total: int
    succeeded: int
    failed: int
    results: List[BatchFileResult]
    def __init__(
        self,
        *,
        total: int = ...,
        succeeded: int = ...,
        failed: int = ...,
        results: Optional[Sequence[BatchFileResult]] = ...,
    ) -> None: ...
    def SerializeToString(self) -> bytes: ...
    @staticmethod
    def FromString(s: bytes) -> "PredictBatchReply": ...
