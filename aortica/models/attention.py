"""Cross-Lead Temporal Attention module for multi-task ECG analysis.

Provides :class:`CrossLeadAttention`, a multi-head attention module that
captures inter-lead relationships in ECG signals.  The module takes backbone
feature output and applies cross-lead attention to produce an enriched
representation.

Attention weights are stored after each forward pass and can be extracted
for XAI (explainability) purposes via the :attr:`attention_weights` property.

Architecture:

* Input: backbone features of shape ``[batch, feature_dim]``
* Reshape to ``[batch, num_leads, feature_dim // num_leads]``
* Multi-head self-attention across the lead dimension
* LayerNorm + residual connection
* Output: enriched feature tensor of shape ``[batch, feature_dim]``
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

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
    F = types.ModuleType("F")  # type: ignore[assignment]


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for CrossLeadAttention. "
            "Install with: pip install aortica[torch]"
        )


class CrossLeadAttention(nn.Module):
    """Multi-head cross-lead attention module.

    Applies multi-head self-attention across ECG leads to capture inter-lead
    relationships such as axis patterns, ischaemia territory, and conduction
    abnormalities.

    The module reshapes the backbone feature vector into per-lead tokens,
    applies multi-head attention, and flattens back.  Attention weights
    from the last forward pass are stored for XAI extraction.

    Args:
        feature_dim: Dimension of the input feature vector from the backbone.
            Must be divisible by ``num_leads``.  Default ``256``.
        num_leads: Number of ECG leads (tokens for attention).  Default ``12``.
        num_heads: Number of attention heads.  Default ``4``.
        head_dim: Dimension per attention head.  Default ``64``.
            If ``None``, computed as ``feature_dim // num_leads // num_heads``
            (requires divisibility).
        dropout: Dropout probability on attention weights.  Default ``0.0``.

    Raises:
        ValueError: If ``feature_dim`` is not divisible by ``num_leads``.

    Example::

        attn = CrossLeadAttention(feature_dim=256, num_leads=12)
        features = torch.randn(4, 256)   # backbone output
        enriched = attn(features)         # [4, 256]
        weights = attn.attention_weights  # [4, num_heads, 12, 12]
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_leads: int = 12,
        num_heads: int = 4,
        head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        _check_torch()
        super().__init__()

        if feature_dim % num_leads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by "
                f"num_leads ({num_leads})."
            )

        self.feature_dim = feature_dim
        self.num_leads = num_leads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        # Per-lead token dimension from backbone
        self.lead_token_dim = feature_dim // num_leads

        # Project each lead token to query/key/value
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(self.lead_token_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(self.lead_token_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(self.lead_token_dim, inner_dim, bias=False)

        # Output projection back to lead_token_dim
        self.out_proj = nn.Linear(inner_dim, self.lead_token_dim, bias=False)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(self.lead_token_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Storage for attention weights (for XAI)
        self._attention_weights: Optional[torch.Tensor] = None

    @property
    def attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from the last forward pass.

        Shape: ``[batch, num_heads, num_leads, num_leads]`` or ``None``
        if no forward pass has been performed.
        """
        return self._attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-lead attention to backbone features.

        Args:
            x: Feature tensor of shape ``[batch, feature_dim]``.

        Returns:
            Enriched feature tensor of shape ``[batch, feature_dim]``.
        """
        batch_size = x.shape[0]

        # Reshape to [batch, num_leads, lead_token_dim]
        tokens = x.view(batch_size, self.num_leads, self.lead_token_dim)

        # Compute Q, K, V  → [batch, num_leads, num_heads * head_dim]
        q = self.q_proj(tokens)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        # Reshape to [batch, num_heads, num_leads, head_dim]
        q = q.view(batch_size, self.num_leads, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.view(batch_size, self.num_leads, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.view(batch_size, self.num_leads, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Store attention weights for XAI (detached)
        self._attention_weights = attn_weights.detach()

        # Apply attention to values → [batch, num_heads, num_leads, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape → [batch, num_leads, num_heads * head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(
            batch_size, self.num_leads, self.num_heads * self.head_dim,
        )

        # Project back to lead_token_dim
        attn_output = self.out_proj(attn_output)

        # Residual connection + LayerNorm
        tokens = self.layer_norm(tokens + attn_output)

        # Flatten back to [batch, feature_dim]
        out: torch.Tensor = tokens.reshape(batch_size, self.feature_dim)
        return out
