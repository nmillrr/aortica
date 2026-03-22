"""Explainability: integrated gradients, VAE latent factor model, VAE reporter."""

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
from aortica.xai.vae_report import (
    LatentActivation,
    VAEReport,
    vae_report,
)

__all__ = [
    "ECG_SEGMENTS",
    "FeatureAttribution",
    "FeatureContribution",
    "LatentActivation",
    "LatentLabel",
    "MedianBeatVAE",
    "SegmentBoundaries",
    "TrainResult",
    "VAEOutput",
    "VAEReport",
    "delineate_segments",
    "explain",
    "extract_median_beat",
    "label_latent_dimensions",
    "train_vae",
    "vae_loss",
    "vae_report",
]
