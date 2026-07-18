"""Federated learning SDK for privacy-preserving collaborative model training."""

from __future__ import annotations

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
    "ReleasePipelineResult",
    "RoundMetrics",
    "RoundRecord",
    "SCAFFOLDStrategy",
    "SecureAggConfig",
    "SecureAggregator",
    "SiteRecord",
    "release_pipeline",
]

