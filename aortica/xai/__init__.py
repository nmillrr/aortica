"""Explainability: integrated gradients, VAE latent factor model."""

from aortica.xai.explain import (
    ECG_SEGMENTS,
    FeatureAttribution,
    FeatureContribution,
    SegmentBoundaries,
    delineate_segments,
    explain,
)
from aortica.xai.median_beat_vae import (
    LatentLabel,
    MedianBeatVAE,
    TrainResult,
    VAEOutput,
    extract_median_beat,
    label_latent_dimensions,
    train_vae,
    vae_loss,
)

__all__ = [
    "ECG_SEGMENTS",
    "FeatureAttribution",
    "FeatureContribution",
    "LatentLabel",
    "MedianBeatVAE",
    "SegmentBoundaries",
    "TrainResult",
    "VAEOutput",
    "delineate_segments",
    "explain",
    "extract_median_beat",
    "label_latent_dimensions",
    "train_vae",
    "vae_loss",
]
