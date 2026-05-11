"""Unified Multi-Task Model Assembly for ECG analysis.

Provides :class:`AorticaModel`, which composes the shared backbone encoder,
cross-lead attention module, and all four task heads (rhythm, structural,
ischaemia, risk) into a single ``nn.Module`` with one forward pass.

A forward pass returns a :class:`MultiTaskOutput` dataclass containing the
predictions from each enabled task head.

Any task head can be disabled at construction time or at inference time
via the ``enabled_tasks`` parameter.  The backbone can be frozen
independently of the heads via :meth:`freeze_backbone` /
:meth:`unfreeze_backbone`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
            "PyTorch is required for AorticaModel. "
            "Install with: pip install aortica[torch]"
        )


# All supported task names.
TASK_NAMES: list[str] = ["rhythm", "structural", "ischaemia", "risk"]


@dataclass
class MultiTaskOutput:
    """Container for multi-task model predictions.

    Each field is ``None`` when the corresponding task head is disabled.

    Attributes:
        rhythm: Rhythm head output ``[batch, 28]`` or ``None``.
        structural: Structural head output ``[batch, 15]`` or ``None``.
        ischaemia: Ischaemia head output ``[batch, 10]`` or ``None``.
        risk: Risk head output ``[batch, 3]`` or ``None``.
    """

    rhythm: Optional[torch.Tensor] = field(default=None)
    structural: Optional[torch.Tensor] = field(default=None)
    ischaemia: Optional[torch.Tensor] = field(default=None)
    risk: Optional[torch.Tensor] = field(default=None)

    def as_dict(self) -> dict[str, Optional[torch.Tensor]]:
        """Return predictions as a plain dictionary."""
        return {
            "rhythm": self.rhythm,
            "structural": self.structural,
            "ischaemia": self.ischaemia,
            "risk": self.risk,
        }


class AorticaModel(nn.Module):
    """Unified multi-task ECG analysis model.

    Composes the shared :class:`AorticaBackbone`, :class:`CrossLeadAttention`,
    and four task heads into a single ``nn.Module``.  A single forward pass
    produces predictions from all enabled task heads.

    Args:
        in_channels: Number of ECG leads.  Default ``12``.
        feature_dim: Backbone feature dimension.  Default ``256``.
        num_leads: Number of leads for cross-lead attention.  Default ``12``.
        num_heads: Attention heads.  Default ``4``.
        head_dim: Per-head attention dimension.  Default ``64``.
        attention_dropout: Attention dropout rate.  Default ``0.0``.
        head_hidden_dim: Hidden dimension for task heads.  Default ``128``.
        head_dropout: Dropout rate for task heads.  Default ``0.3``.
        enabled_tasks: Which task heads to include.  Default all four.
            Any subset of ``['rhythm', 'structural', 'ischaemia', 'risk']``.

    Example::

        model = AorticaModel(in_channels=12)
        x = torch.randn(4, 12, 5000)
        output = model(x)
        output.rhythm    # [4, 28]
        output.risk      # [4, 3]

        # Rhythm-only model
        model = AorticaModel(enabled_tasks=['rhythm'])
        output = model(x)
        output.rhythm    # [4, 28]
        output.structural  # None
    """

    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 256,
        num_leads: int = 12,
        num_heads: int = 4,
        head_dim: int = 64,
        attention_dropout: float = 0.0,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.3,
        enabled_tasks: Optional[list[str]] = None,
    ) -> None:
        _check_torch()
        super().__init__()

        if enabled_tasks is None:
            enabled_tasks = list(TASK_NAMES)
        for task in enabled_tasks:
            if task not in TASK_NAMES:
                raise ValueError(
                    f"Unknown task '{task}'. Must be one of {TASK_NAMES}."
                )

        self.enabled_tasks = enabled_tasks
        self.feature_dim = feature_dim

        # --- Backbone ---
        from aortica.models.backbone import AorticaBackbone

        self.backbone = AorticaBackbone(
            in_channels=in_channels,
            feature_dim=feature_dim,
        )

        # --- Cross-Lead Attention ---
        from aortica.models.attention import CrossLeadAttention

        self.attention = CrossLeadAttention(
            feature_dim=feature_dim,
            num_leads=num_leads,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=attention_dropout,
        )

        # --- Task Heads (only create enabled ones) ---
        self.rhythm_head: Optional[nn.Module] = None
        self.structural_head: Optional[nn.Module] = None
        self.ischaemia_head: Optional[nn.Module] = None
        self.risk_head: Optional[nn.Module] = None

        if "rhythm" in enabled_tasks:
            from aortica.models.rhythm_head import RhythmHead

            self.rhythm_head = RhythmHead(
                feature_dim=feature_dim,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )

        if "structural" in enabled_tasks:
            from aortica.models.structural_head import StructuralHead

            self.structural_head = StructuralHead(
                feature_dim=feature_dim,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )

        if "ischaemia" in enabled_tasks:
            from aortica.models.ischaemia_head import IschaemiaHead

            self.ischaemia_head = IschaemiaHead(
                feature_dim=feature_dim,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )

        if "risk" in enabled_tasks:
            from aortica.models.risk_head import RiskHead

            self.risk_head = RiskHead(
                feature_dim=feature_dim,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (stop gradient updates)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters (allow gradient updates)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @property
    def attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from the last forward pass."""
        return self.attention.attention_weights

    def forward(
        self,
        x: torch.Tensor,
        tasks: Optional[list[str]] = None,
    ) -> MultiTaskOutput:
        """Run a full forward pass through backbone, attention, and heads.

        Args:
            x: Input ECG tensor of shape ``[batch, leads, samples]``.
            tasks: Optional subset of enabled tasks to run on this pass.
                If ``None``, runs all enabled task heads.

        Returns:
            :class:`MultiTaskOutput` with predictions for each active head.
        """
        if tasks is not None:
            for task in tasks:
                if task not in self.enabled_tasks:
                    raise ValueError(
                        f"Task '{task}' is not enabled. "
                        f"Enabled tasks: {self.enabled_tasks}"
                    )
            active_tasks = tasks
        else:
            active_tasks = self.enabled_tasks

        # Shared feature extraction
        features = self.backbone(x)
        features = self.attention(features)

        # Task-specific predictions
        rhythm_out: Optional[torch.Tensor] = None
        structural_out: Optional[torch.Tensor] = None
        ischaemia_out: Optional[torch.Tensor] = None
        risk_out: Optional[torch.Tensor] = None

        if "rhythm" in active_tasks and self.rhythm_head is not None:
            rhythm_out = self.rhythm_head(features)

        if "structural" in active_tasks and self.structural_head is not None:
            structural_out = self.structural_head(features)

        if "ischaemia" in active_tasks and self.ischaemia_head is not None:
            ischaemia_out = self.ischaemia_head(features)

        if "risk" in active_tasks and self.risk_head is not None:
            risk_out = self.risk_head(features)

        return MultiTaskOutput(
            rhythm=rhythm_out,
            structural=structural_out,
            ischaemia=ischaemia_out,
            risk=risk_out,
        )
