"""TensorFlow/Keras implementation of the Cross-Lead Temporal Attention module.

Provides :func:`build_cross_lead_attention_tf`, a Keras functional-API model
that applies multi-head cross-lead self-attention to backbone features.

The model returns both the enriched features and the attention weights so that
attention can be visualised for XAI purposes.

Input shape:  ``(batch, feature_dim)``
Output shapes:
  - enriched features: ``(batch, feature_dim)``
  - attention weights:  ``(batch, num_heads, num_leads, num_leads)``
"""

from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow import keras  # type: ignore[attr-defined]

    HAS_TF = True
except ImportError:
    HAS_TF = False


def _check_tf() -> None:
    if not HAS_TF:
        raise ImportError(
            "TensorFlow is required for build_cross_lead_attention_tf. "
            "Install with: pip install tensorflow"
        )


def build_cross_lead_attention_tf(
    feature_dim: int = 256,
    num_leads: int = 12,
    num_heads: int = 4,
    head_dim: int = 64,
    dropout: float = 0.0,
) -> "keras.Model":
    """Build a Keras functional-API cross-lead attention model.

    Args:
        feature_dim: Input feature vector dimension. Must be divisible by
            ``num_leads``.  Default ``256``.
        num_leads: Number of ECG leads.  Default ``12``.
        num_heads: Number of attention heads.  Default ``4``.
        head_dim: Dimension per attention head.  Default ``64``.
        dropout: Dropout probability on attention weights.  Default ``0.0``.

    Returns:
        A ``keras.Model`` with two outputs: (enriched_features, attn_weights).

    Raises:
        ValueError: If ``feature_dim`` is not divisible by ``num_leads``.
    """
    _check_tf()

    if feature_dim % num_leads != 0:
        raise ValueError(
            f"feature_dim ({feature_dim}) must be divisible by "
            f"num_leads ({num_leads})."
        )

    lead_token_dim = feature_dim // num_leads
    inner_dim = num_heads * head_dim

    inputs = keras.Input(shape=(feature_dim,), name="backbone_features")

    # Reshape to [batch, num_leads, lead_token_dim]
    tokens = keras.layers.Reshape(
        (num_leads, lead_token_dim), name="reshape_to_tokens",
    )(inputs)

    # Q, K, V projections
    q = keras.layers.Dense(inner_dim, use_bias=False, name="q_proj")(tokens)
    k = keras.layers.Dense(inner_dim, use_bias=False, name="k_proj")(tokens)
    v = keras.layers.Dense(inner_dim, use_bias=False, name="v_proj")(tokens)

    # Reshape to [batch, num_leads, num_heads, head_dim]
    # then permute to [batch, num_heads, num_leads, head_dim]
    q = keras.layers.Reshape(
        (num_leads, num_heads, head_dim), name="q_reshape",
    )(q)
    q = keras.layers.Permute((2, 1, 3), name="q_permute")(q)

    k = keras.layers.Reshape(
        (num_leads, num_heads, head_dim), name="k_reshape",
    )(k)
    k = keras.layers.Permute((2, 1, 3), name="k_permute")(k)

    v = keras.layers.Reshape(
        (num_leads, num_heads, head_dim), name="v_reshape",
    )(v)
    v = keras.layers.Permute((2, 1, 3), name="v_permute")(v)

    # Scaled dot-product attention
    scale = head_dim ** 0.5
    # attn_scores: [batch, num_heads, num_leads, num_leads]
    attn_scores = keras.layers.Lambda(
        lambda qk: tf.matmul(qk[0], qk[1], transpose_b=True) / scale,
        name="attn_scores",
    )([q, k])

    attn_weights = keras.layers.Softmax(axis=-1, name="attn_softmax")(
        attn_scores,
    )

    if dropout > 0.0:
        attn_weights_dropped = keras.layers.Dropout(
            dropout, name="attn_dropout",
        )(attn_weights)
    else:
        attn_weights_dropped = attn_weights

    # Apply attention: [batch, num_heads, num_leads, head_dim]
    attn_output = keras.layers.Lambda(
        lambda wv: tf.matmul(wv[0], wv[1]),
        name="attn_apply",
    )([attn_weights_dropped, v])

    # Permute back to [batch, num_leads, num_heads, head_dim]
    attn_output = keras.layers.Permute(
        (2, 1, 3), name="attn_permute_back",
    )(attn_output)

    # Reshape to [batch, num_leads, inner_dim]
    attn_output = keras.layers.Reshape(
        (num_leads, inner_dim), name="attn_reshape_back",
    )(attn_output)

    # Output projection back to lead_token_dim
    attn_output = keras.layers.Dense(
        lead_token_dim, use_bias=False, name="out_proj",
    )(attn_output)

    # Residual + LayerNorm
    residual = keras.layers.Add(name="residual_add")([tokens, attn_output])
    normed = keras.layers.LayerNormalization(
        name="layer_norm",
    )(residual)

    # Flatten back to [batch, feature_dim]
    enriched = keras.layers.Reshape(
        (feature_dim,), name="flatten_features",
    )(normed)

    model = keras.Model(
        inputs=inputs,
        outputs=[enriched, attn_weights],
        name="CrossLeadAttentionTF",
    )
    return model
