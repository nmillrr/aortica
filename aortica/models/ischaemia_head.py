"""Ischaemia & Metabolic Task Head for multi-task ECG analysis.

Provides :class:`IschaemiaHead`, a multi-label classification head producing 15
sigmoid outputs corresponding to ischaemic and metabolic ECG patterns.

The 15 classes are:
    STEMI, posterior_MI, occlusive_NSTEMI, old_MI, hyperkalaemia,
    hypokalaemia, hypercalcaemia, hypothyroidism_pattern,
    digitalis_effect, QTc_prolongation,
    early_repol_vs_STEMI, de_Winter_T_wave, Wellens_syndrome,
    aVR_ST_elevation, Sgarbossa_criteria

The head connects to the backbone + attention output and uses binary
cross-entropy with optional class-weight balancing.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    import types

    HAS_TORCH = False
    torch = types.ModuleType("torch")  # type: ignore[assignment]

    class _DummyModule:
        """Placeholder base when torch is absent."""

        pass

    nn = types.ModuleType("nn")  # type: ignore[assignment]
    nn.Module = _DummyModule  # type: ignore[attr-defined]


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for IschaemiaHead. "
            "Install with: pip install aortica[torch]"
        )


# Canonical class ordering (15 ischaemia & metabolic classes).
ISCHAEMIA_CLASSES: list[str] = [
    "STEMI",
    "posterior_MI",
    "occlusive_NSTEMI",
    "old_MI",
    "hyperkalaemia",
    "hypokalaemia",
    "hypercalcaemia",
    "hypothyroidism_pattern",
    "digitalis_effect",
    "QTc_prolongation",
    # Phase 3 STEMI mimics & subtle ischaemia subtypes (US-073)
    "early_repol_vs_STEMI",
    "de_Winter_T_wave",
    "Wellens_syndrome",
    "aVR_ST_elevation",
    "Sgarbossa_criteria",
]

NUM_ISCHAEMIA_CLASSES: int = len(ISCHAEMIA_CLASSES)


class IschaemiaHead(nn.Module):
    """Multi-label ischaemia & metabolic classification head (15 classes).

    Architecture:

    * Linear(feature_dim, hidden_dim) + ReLU + Dropout
    * Linear(hidden_dim, 15)
    * Sigmoid activation (applied in :meth:`forward`)

    The head outputs raw logits via :meth:`forward_logits` (for use with
    ``BCEWithLogitsLoss``) or probabilities via :meth:`forward`.

    Args:
        feature_dim: Dimension of the input feature vector from the
            backbone + attention module.  Default ``256``.
        hidden_dim: Hidden layer dimension.  Default ``128``.
        dropout: Dropout probability.  Default ``0.3``.

    Example::

        head = IschaemiaHead(feature_dim=256)
        features = torch.randn(4, 256)
        probs = head(features)       # [4, 15], values in (0, 1)
        logits = head.forward_logits(features)  # [4, 15], raw logits
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        _check_torch()
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = NUM_ISCHAEMIA_CLASSES

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_ISCHAEMIA_CLASSES),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (for use with ``BCEWithLogitsLoss``).

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Logit tensor of shape ``[batch, 15]``.
        """
        logits: torch.Tensor = self.classifier(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities.

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Probability tensor of shape ``[batch, 15]``, values in ``(0, 1)``.
        """
        logits = self.forward_logits(x)
        probs: torch.Tensor = torch.sigmoid(logits)
        return probs


def compute_ischaemia_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BCE loss with optional per-class weight balancing.

    Args:
        logits: Raw logits of shape ``[batch, 15]``.
        targets: Binary targets of shape ``[batch, 15]``.
        class_weights: Optional weight tensor of shape ``[15]`` for class
            balancing.  If ``None``, uniform weighting is used.

    Returns:
        Scalar loss tensor.
    """
    _check_torch()
    loss: torch.Tensor = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=class_weights,
    )
    return loss
