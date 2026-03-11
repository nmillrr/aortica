"""1D ResNet-18 baseline model adapted for 12-lead ECG classification.

This module provides a 1D adaptation of the standard ResNet-18 architecture
for multi-label rhythm classification on 12-lead ECG data.  The network
accepts inputs of shape ``[batch, leads, samples]`` and produces sigmoid
outputs suitable for multi-label binary classification.
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
    # Provide stubs so the class *definitions* succeed at import time.
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
            "PyTorch is required for ResNet1D. Install with: pip install aortica[torch]"
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
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
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


class ResNet1D(nn.Module):
    """1D ResNet-18 adapted for 12-lead ECG rhythm classification.

    Architecture mirrors the standard ResNet-18 but operates on 1D
    temporal signals rather than 2D images:

    * Initial convolution (kernel=15, stride=2) + BN + ReLU + MaxPool
    * 4 residual stages with ``[2, 2, 2, 2]`` blocks at
      ``[64, 128, 256, 512]`` filter widths
    * Global adaptive average pooling → fully connected classifier

    Args:
        in_channels: Number of input channels (leads). Default ``12``.
        num_classes: Number of output classes. Default ``3``
            (rhythm, structural, ischaemia superclasses for PTB-XL).
        kernel_size: Convolution kernel size for residual blocks.
            Default ``7``.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 3,
        kernel_size: int = 7,
    ) -> None:
        _check_torch()
        super().__init__()

        self.in_channels_current = 64

        # Initial feature extraction
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(64, 2, kernel_size, stride=1)
        self.layer2 = self._make_layer(128, 2, kernel_size, stride=2)
        self.layer3 = self._make_layer(256, 2, kernel_size, stride=2)
        self.layer4 = self._make_layer(512, 2, kernel_size, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        kernel_size: int,
        stride: int,
    ) -> nn.Sequential:
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.in_channels_current != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels_current,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers: list[nn.Module] = [
            ResidualBlock1D(
                self.in_channels_current, out_channels, kernel_size, stride, downsample
            )
        ]
        self.in_channels_current = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock1D(out_channels, out_channels, kernel_size)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``[batch, leads, samples]``.

        Returns:
            Sigmoid-activated output of shape ``[batch, num_classes]``.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return torch.sigmoid(x)
