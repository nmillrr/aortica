"""Rhythm & Conduction Task Head for multi-task ECG analysis.

Provides :class:`RhythmHead`, a multi-label classification head producing 22
sigmoid outputs corresponding to rhythm and conduction abnormalities.

The 22 classes are:
    AF, AFL, SVT, AVNRT, AVRT, VT, VF, idioventricular, sinus_brady,
    sinus_tachy, PAC, PVC, av_block_1st, av_block_2nd, av_block_3rd,
    LBBB, RBBB, LAFB, LPFB, WPW, pacemaker_rhythm, normal_sinus_rhythm

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
            "PyTorch is required for RhythmHead. "
            "Install with: pip install aortica[torch]"
        )


# Canonical class ordering (22 rhythm & conduction classes).
RHYTHM_CLASSES: list[str] = [
    "AF",
    "AFL",
    "SVT",
    "AVNRT",
    "AVRT",
    "VT",
    "VF",
    "idioventricular",
    "sinus_brady",
    "sinus_tachy",
    "PAC",
    "PVC",
    "av_block_1st",
    "av_block_2nd",
    "av_block_3rd",
    "LBBB",
    "RBBB",
    "LAFB",
    "LPFB",
    "WPW",
    "pacemaker_rhythm",
    "normal_sinus_rhythm",
]

NUM_RHYTHM_CLASSES: int = len(RHYTHM_CLASSES)


class RhythmHead(nn.Module):
    """Multi-label rhythm & conduction classification head (22 classes).

    Architecture:

    * Linear(feature_dim, hidden_dim) + ReLU + Dropout
    * Linear(hidden_dim, 22)
    * Sigmoid activation (applied in :meth:`forward`)

    The head outputs raw logits via :meth:`forward_logits` (for use with
    ``BCEWithLogitsLoss``) or probabilities via :meth:`forward`.

    Args:
        feature_dim: Dimension of the input feature vector from the
            backbone + attention module.  Default ``256``.
        hidden_dim: Hidden layer dimension.  Default ``128``.
        dropout: Dropout probability.  Default ``0.3``.

    Example::

        head = RhythmHead(feature_dim=256)
        features = torch.randn(4, 256)
        probs = head(features)       # [4, 22], values in (0, 1)
        logits = head.forward_logits(features)  # [4, 22], raw logits
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
        self.num_classes = NUM_RHYTHM_CLASSES

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_RHYTHM_CLASSES),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (for use with ``BCEWithLogitsLoss``).

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Logit tensor of shape ``[batch, 22]``.
        """
        logits: torch.Tensor = self.classifier(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities.

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Probability tensor of shape ``[batch, 22]``, values in ``(0, 1)``.
        """
        logits = self.forward_logits(x)
        probs: torch.Tensor = torch.sigmoid(logits)
        return probs


def compute_rhythm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BCE loss with optional per-class weight balancing.

    Args:
        logits: Raw logits of shape ``[batch, 22]``.
        targets: Binary targets of shape ``[batch, 22]``.
        class_weights: Optional weight tensor of shape ``[22]`` for class
            balancing.  If ``None``, uniform weighting is used.

    Returns:
        Scalar loss tensor.
    """
    _check_torch()
    loss: torch.Tensor = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=class_weights,
    )
    return loss
