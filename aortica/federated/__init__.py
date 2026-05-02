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
from aortica.federated.fl_server import (
    FLServer,
    FLServerConfig,
    RoundMetrics,
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
    "DPConfig",
    "DPWrapper",
    "EncryptedWeights",
    "FLClientConfig",
    "FLServer",
    "FLServerConfig",
    "FedProxStrategy",
    "PrivacyBudgetTracker",
    "RoundMetrics",
    "SCAFFOLDStrategy",
    "SecureAggConfig",
    "SecureAggregator",
]
