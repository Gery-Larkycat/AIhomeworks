"""
ResNet-18 implementation from scratch, adapted for CIFAR (32x32) images.
从零实现的 ResNet-18，适配 CIFAR（32x32）图像。

Architecture changes from standard ImageNet ResNet-18:
- Stem: 3x3 conv stride=1 (instead of 7x7 conv stride=2 + maxpool)
- All residual blocks remain standard BasicBlock [2,2,2,2]
- Only FC layer changes for different datasets (num_classes)
"""

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock (two 3x3 conv layers).
    expansion=1 means in_channels == out_channels for each block group.
    """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 with CIFAR-adapted stem.
    Stem: 3x3 conv stride=1, no maxpool (preserves 32x32 for CIFAR images)
    Layers: 4 groups of BasicBlock [2,2,2,2], channels 64→128→256→512
    Head: AdaptiveAvgPool2d → Dropout → Linear(512, num_classes)
    """

    def __init__(
        self, num_classes: int = 10, dropout_rate: float = 0.0, **kwargs
    ) -> None:
        super().__init__()
        self.in_channels = 64

        # Stem: 3x3 conv stride=1 instead of 7x7 stride=2 + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual layer groups: [2,2,2,2] blocks, channels: 64→128→256→512
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def create_model(num_classes: int = 10, dropout_rate: float = 0.0) -> ResNet18:
    """Factory function: create a ResNet-18 instance."""
    return ResNet18(num_classes=num_classes, dropout_rate=dropout_rate)


def get_feature_extractor_state(model: ResNet18) -> OrderedDict[str, Any]:
    """
    Extract state dict without FC layer, for transfer learning.
    提取不含 FC 层的 state dict，供迁移学习使用。
    """
    return OrderedDict(
        (k, v) for k, v in model.state_dict().items() if not k.startswith("fc.")
    )
