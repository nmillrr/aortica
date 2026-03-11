"""Shared ResNet backbone encoder for multi-task ECG analysis.

Provides :class:`AorticaBackbone`, a 1D ResNet encoder with residual blocks at
64, 128, and 256 filter widths.  The output is a feature tensor suitable for
downstream task heads (rhythm, structural, ischaemia, risk).

The backbone accepts input of shape ``[batch, leads, samples]`` and uses
adaptive pooling so it can handle varying sampling rates (250–1000 Hz) and
window durations (2.5–10 s).
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
            "PyTorch is required for AorticaBackbone. "
            "Install with: pip install aortica[torch]"
        )


class ResidualBlock1D(nn.Module):
    """Basic 1D residual block with two convolutional layers.

    Uses BatchNorm and ReLU.  When *downsample* is provided the
    shortcut connection is projected to match the output dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        _check_torch()
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class AorticaBackbone(nn.Module):
    """Shared 1D ResNet backbone encoder for multi-task ECG analysis.

    Architecture:

    * Initial convolution (kernel=15, stride=2) + BN + ReLU + MaxPool
    * 3 residual stages with ``[2, 2, 2]`` blocks at
      ``[64, 128, 256]`` filter widths
    * Global adaptive average pooling → feature vector of size 256

    The backbone is designed to be composed with downstream task heads
    via :class:`aortica.models.AorticaModel`.

    Args:
        in_channels: Number of input channels (leads). Default ``12``.
        feature_dim: Dimension of the output feature vector. Default ``256``
            (matches the last residual stage width).
        kernel_size: Convolution kernel size for residual blocks. Default ``7``.

    Example::

        backbone = AorticaBackbone(in_channels=12)
        x = torch.randn(4, 12, 5000)  # 4 samples, 12 leads, 10s @ 500 Hz
        features = backbone(x)          # [4, 256]
    """

    # Output feature dimension (useful for task heads to query).
    DEFAULT_FEATURE_DIM: int = 256

    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 256,
        kernel_size: int = 7,
    ) -> None:
        _check_torch()
        super().__init__()

        self.feature_dim = feature_dim
        self._in_channels_current = 64

        # Initial feature extraction
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages at 64, 128, 256
        self.layer1 = self._make_layer(64, 2, kernel_size, stride=1)
        self.layer2 = self._make_layer(128, 2, kernel_size, stride=2)
        self.layer3 = self._make_layer(256, 2, kernel_size, stride=2)

        # Global pooling → feature vector
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Projection to feature_dim if it differs from the last stage width
        if feature_dim != 256:
            self.fc = nn.Linear(256, feature_dim)
        else:
            self.fc = None  # type: ignore[assignment]

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        kernel_size: int,
        stride: int,
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None
        if stride != 1 or self._in_channels_current != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self._in_channels_current, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers: list[nn.Module] = [
            ResidualBlock1D(
                self._in_channels_current, out_channels,
                kernel_size, stride, downsample,
            )
        ]
        self._in_channels_current = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock1D(out_channels, out_channels, kernel_size)
            )

        return nn.Sequential(*layers)

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
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.squeeze(-1)

        if self.fc is not None:
            x = self.fc(x)

        return x
