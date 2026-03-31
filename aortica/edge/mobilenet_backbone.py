"""Lightweight MobileNet-style 1D backbone for edge deployment.

Provides :class:`MobileNetBackbone1D`, a compact encoder that uses
depthwise-separable 1D convolutions at 32, 64, and 128 filter widths.
The architecture is designed to be a drop-in replacement for
:class:`~aortica.models.backbone.AorticaBackbone` with significantly fewer
parameters (≤ 2.5 M), enabling deployment on resource-constrained devices
such as Raspberry Pi or mobile ARM hardware.

The backbone accepts input of shape ``[batch, leads, samples]`` and returns
a feature vector of shape ``[batch, feature_dim]``, identical to
``AorticaBackbone``.
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
    nn.Sequential = _DummyModule  # type: ignore[attr-defined]


def _check_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for MobileNetBackbone1D. "
            "Install with: pip install aortica[torch]"
        )


class DepthwiseSeparableConv1D(nn.Module):
    """Depthwise-separable 1D convolution block.

    Combines a depthwise convolution (one filter per input channel) with a
    pointwise (1×1) convolution to mix channels.  Each is followed by batch
    normalization and ReLU activation.

    This is the fundamental building block of MobileNet architectures and
    reduces parameter count by roughly ``kernel_size`` times compared to a
    standard convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the depthwise convolution. Default ``7``.
        stride: Stride for the depthwise convolution. Default ``1``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
    ) -> None:
        _check_torch()
        super().__init__()
        padding = kernel_size // 2

        # Depthwise: one filter per input channel.
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm1d(in_channels)

        # Pointwise: 1×1 conv to mix channels.
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn_pw = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[batch, in_channels, length]``.

        Returns:
            Output tensor of shape ``[batch, out_channels, length // stride]``.
        """
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu(x)

        return x


class InvertedResidual1D(nn.Module):
    """Inverted residual block (MobileNet-v2 style) adapted for 1D signals.

    Expands channels via a pointwise conv, applies a depthwise conv, then
    projects back down.  Includes a residual connection when input and output
    shapes match (same channels and stride == 1).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the depthwise convolution. Default ``7``.
        stride: Stride for the depthwise convolution. Default ``1``.
        expand_ratio: Channel expansion factor. Default ``2``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        expand_ratio: int = 2,
    ) -> None:
        _check_torch()
        super().__init__()

        mid_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []

        # Expansion (pointwise).
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
            ])

        # Depthwise.
        padding = kernel_size // 2
        layers.extend([
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        ])

        # Projection (pointwise, no activation — linear bottleneck).
        layers.extend([
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection.

        Args:
            x: Input tensor of shape ``[batch, in_channels, length]``.

        Returns:
            Output tensor of shape ``[batch, out_channels, length // stride]``.
        """
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNetBackbone1D(nn.Module):
    """Lightweight MobileNet-style 1D backbone for edge ECG inference.

    Architecture:

    * Initial convolution (kernel=15, stride=2) + BN + ReLU
    * Stage 1: 2 inverted residual blocks at 32 channels (stride=2 entry)
    * Stage 2: 2 inverted residual blocks at 64 channels (stride=2 entry)
    * Stage 3: 2 inverted residual blocks at 128 channels (stride=2 entry)
    * Global adaptive average pooling → feature vector
    * Optional linear projection to ``feature_dim``

    The model is designed as a drop-in replacement for
    :class:`~aortica.models.backbone.AorticaBackbone` accepting the same input
    shape ``[batch, leads, samples]`` and returning ``[batch, feature_dim]``.

    Args:
        in_channels: Number of input channels (leads). Default ``12``.
        feature_dim: Dimension of the output feature vector. Default ``256``.
        kernel_size: Convolution kernel size for residual blocks. Default ``7``.
        expand_ratio: Channel expansion ratio in inverted residuals.
            Default ``2``.

    Example::

        backbone = MobileNetBackbone1D(in_channels=12)
        x = torch.randn(4, 12, 5000)  # 4 samples, 12 leads, 10s @ 500 Hz
        features = backbone(x)          # [4, 256]
    """

    DEFAULT_FEATURE_DIM: int = 256

    # Stage filter widths — deliberately compact for edge deployment.
    STAGE_CHANNELS: tuple[int, ...] = (32, 64, 128)

    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 256,
        kernel_size: int = 7,
        expand_ratio: int = 2,
    ) -> None:
        _check_torch()
        super().__init__()

        self.feature_dim = feature_dim
        first_stage_ch = self.STAGE_CHANNELS[0]

        # Initial feature extraction.
        self.conv1 = nn.Conv1d(
            in_channels, first_stage_ch, kernel_size=15,
            stride=2, padding=7, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(first_stage_ch)
        self.relu = nn.ReLU(inplace=True)

        # Build stages.
        stages: list[nn.Module] = []
        prev_ch = first_stage_ch
        for stage_ch in self.STAGE_CHANNELS:
            # First block of each stage strides by 2 (and may change channels).
            stages.append(InvertedResidual1D(
                prev_ch, stage_ch, kernel_size,
                stride=2, expand_ratio=expand_ratio,
            ))
            # Second block keeps the same channels and no stride.
            stages.append(InvertedResidual1D(
                stage_ch, stage_ch, kernel_size,
                stride=1, expand_ratio=expand_ratio,
            ))
            prev_ch = stage_ch

        self.stages = nn.Sequential(*stages)

        # Global pooling → feature vector.
        last_ch = self.STAGE_CHANNELS[-1]  # 128
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Projection to feature_dim.
        if feature_dim != last_ch:
            self.fc: Optional[nn.Linear] = nn.Linear(last_ch, feature_dim)
        else:
            self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from ECG input.

        Args:
            x: Input tensor of shape ``[batch, leads, samples]``.

        Returns:
            Feature tensor of shape ``[batch, feature_dim]``.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stages(x)

        x = self.avgpool(x)
        x = x.squeeze(-1)

        if self.fc is not None:
            x = self.fc(x)

        return x
