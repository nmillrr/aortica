"""Federated learning SDK: train across sites without pooling raw ECGs.

Raw data stays on site; model updates are exchanged in the clear.  The
:mod:`~aortica.federated.dp` and :mod:`~aortica.federated.secure_agg` modules
are experimental and **not** wired into the client, server or CLI, so the
pipeline as shipped offers neither differential privacy nor secure
aggregation.
"""

from __future__ import annotations

from aortica.federated.data_quality import (
    DQ_POLICIES,
    DataQualityGate,
    DataQualityReport,
    QualityCheck,
    site_admitted,
)
from aortica.federated.dp import (
    DPConfig,
    DPWrapper,
    PrivacyBudgetTracker,
)
from aortica.federated.fl_client import (
    AorticaFlowerClient,
    FLClientConfig,
)
from aortica.federated.fl_metrics_store import (
    CampaignStatus,
    FLMetricsStore,
    RoundRecord,
    SiteRecord,
)
from aortica.federated.fl_server import (
    FLServer,
    FLServerConfig,
    RoundMetrics,
)
from aortica.federated.release_pipeline import (
    FederatedReleaseConfig,
    PipelineStepResult,
    ReleasePipelineResult,
    release_pipeline,
)
from aortica.federated.secure_agg import (
    EncryptedWeights,
    SecureAggConfig,
    SecureAggregator,
)
from aortica.federated.strategies import (
    FedProxStrategy,
    SCAFFOLDStrategy,
)

__all__ = [
    "AorticaFlowerClient",
    "CampaignStatus",
    "DQ_POLICIES",
    "DataQualityGate",
    "DataQualityReport",
    "DPConfig",
    "DPWrapper",
    "EncryptedWeights",
    "FLClientConfig",
    "FLMetricsStore",
    "FLServer",
    "FLServerConfig",
    "FederatedReleaseConfig",
    "FedProxStrategy",
    "PipelineStepResult",
    "PrivacyBudgetTracker",
    "QualityCheck",
    "ReleasePipelineResult",
    "RoundMetrics",
    "RoundRecord",
    "SCAFFOLDStrategy",
    "SecureAggConfig",
    "SecureAggregator",
    "SiteRecord",
    "release_pipeline",
    "site_admitted",
]

