"""Structural & Functional Task Head for multi-task ECG analysis.

Provides :class:`StructuralHead`, a multi-label classification head producing 19
sigmoid outputs corresponding to structural and functional cardiac abnormalities.

The 19 classes are:
    LVH, RVH, LVSD, HFpEF_risk, DCM, HCM, ARVC, amyloidosis,
    aortic_stenosis, mitral_regurgitation, pulmonary_HTN,
    LA_enlargement, RA_enlargement, pericarditis, myocarditis,
    LV_strain_grade, RV_strain_PE, Takotsubo_pattern,
    infiltrative_cardiomyopathy_strain

The head connects to the backbone + attention output and uses binary
cross-entropy with an optional focal loss modifier for rare classes.
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
            "PyTorch is required for StructuralHead. "
            "Install with: pip install aortica[torch]"
        )


# Canonical class ordering (19 structural & functional classes).
STRUCTURAL_CLASSES: list[str] = [
    "LVH",
    "RVH",
    "LVSD",
    "HFpEF_risk",
    "DCM",
    "HCM",
    "ARVC",
    "amyloidosis",
    "aortic_stenosis",
    "mitral_regurgitation",
    "pulmonary_HTN",
    "LA_enlargement",
    "RA_enlargement",
    "pericarditis",
    "myocarditis",
    # Phase 3 strain pattern sub-classifiers (US-074)
    "LV_strain_grade",
    "RV_strain_PE",
    "Takotsubo_pattern",
    "infiltrative_cardiomyopathy_strain",
]

NUM_STRUCTURAL_CLASSES: int = len(STRUCTURAL_CLASSES)


class StructuralHead(nn.Module):
    """Multi-label structural & functional classification head (19 classes).

    Architecture:

    * Linear(feature_dim, hidden_dim) + ReLU + Dropout
    * Linear(hidden_dim, 19)
    * Sigmoid activation (applied in :meth:`forward`)

    The head outputs raw logits via :meth:`forward_logits` (for use with
    ``BCEWithLogitsLoss`` or focal loss) or probabilities via :meth:`forward`.

    Args:
        feature_dim: Dimension of the input feature vector from the
            backbone + attention module.  Default ``256``.
        hidden_dim: Hidden layer dimension.  Default ``128``.
        dropout: Dropout probability.  Default ``0.3``.

    Example::

        head = StructuralHead(feature_dim=256)
        features = torch.randn(4, 256)
        probs = head(features)       # [4, 19], values in (0, 1)
        logits = head.forward_logits(features)  # [4, 19], raw logits
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
        self.num_classes = NUM_STRUCTURAL_CLASSES

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_STRUCTURAL_CLASSES),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (for use with ``BCEWithLogitsLoss`` or focal loss).

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Logit tensor of shape ``[batch, 19]``.
        """
        logits: torch.Tensor = self.classifier(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities.

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Probability tensor of shape ``[batch, 19]``, values in ``(0, 1)``.
        """
        logits = self.forward_logits(x)
        probs: torch.Tensor = torch.sigmoid(logits)
        return probs


def compute_structural_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute BCE loss with optional focal loss modifier for rare classes.

    When ``focal=False`` (default), this is standard BCE with optional
    per-class weights.  When ``focal=True``, it applies the focal loss
    modulating factor ``(1 - p_t)^gamma`` to down-weight easy examples
    and focus training on hard/rare classes.

    Args:
        logits: Raw logits of shape ``[batch, 19]``.
        targets: Binary targets of shape ``[batch, 19]``.
        class_weights: Optional weight tensor of shape ``[19]`` for class
            balancing (used only when ``focal=False``).
        focal: If ``True``, use focal loss instead of plain BCE.
        focal_gamma: Focusing parameter for focal loss.  Default ``2.0``.
        focal_alpha: Optional per-class balancing factor of shape ``[19]``
            for focal loss.  If ``None``, uniform alpha is used.

    Returns:
        Scalar loss tensor.
    """
    _check_torch()

    if not focal:
        loss: torch.Tensor = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=class_weights,
        )
        return loss

    # Focal loss implementation
    # p = sigmoid(logits)
    p = torch.sigmoid(logits)
    # p_t = p * target + (1 - p) * (1 - target)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)

    # BCE per element (no reduction)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )

    # Focal modulator
    focal_weight = (1.0 - p_t) ** focal_gamma

    loss_elements = focal_weight * bce

    if focal_alpha is not None:
        # alpha_t: alpha for positive, (1 - alpha) for negative
        alpha_t = focal_alpha * targets + (1.0 - focal_alpha) * (1.0 - targets)
        loss_elements = alpha_t * loss_elements

    loss = loss_elements.mean()
    return loss
