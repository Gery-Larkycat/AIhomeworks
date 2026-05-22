"""
ResNet-18 implementation from scratch, adapted for CIFAR (32x32) images.
从零实现的 ResNet-18，适配 CIFAR（32x32）图像。

Architecture changes from standard ImageNet ResNet-18:
- Stem: 3x3 conv stride=1 (instead of 7x7 conv stride=2 + maxpool)
- All residual blocks remain standard BasicBlock [2,2,2,2]
- Only FC layer changes for different datasets (num_classes)

与标准 ImageNet ResNet-18 的架构差异：
- Stem: 3x3 卷积 stride=1（替代 7x7 stride=2 + maxpool）
- 所有残差块完全保持标准 BasicBlock [2,2,2,2]
- 仅 FC 层适配不同数据集的类别数
"""

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock (two 3x3 conv layers).
    标准 ResNet 基础残差块（两个 3x3 卷积层）。

    Each block: conv3x3 → BN → ReLU → conv3x3 → BN + shortcut → ReLU
    expansion=1 means in_channels == out_channels for each block group.
    expansion=1 表示每个块组的输入通道数等于输出通道数。
    """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # First conv may downsample via stride / 第一个卷积可能通过 stride 下采样
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second conv maintains spatial size / 第二个卷积保持空间尺寸
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut / 残差跳跃连接
        # If dimensions don't match, use 1x1 conv to align / 维度不匹配时用 1x1 卷积对齐
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
    适配 CIFAR 的 ResNet-18。

    Stem: 3x3 conv stride=1, no maxpool (preserves 32x32 for CIFAR images)
    Layers: 4 groups of BasicBlock [2,2,2,2], channels 64→128→256→512
    Head: AdaptiveAvgPool2d → Linear(512, num_classes)

    Stem: 3x3 卷积 stride=1，无 maxpool（保持 CIFAR 32x32 分辨率）
    层组：4 组 BasicBlock [2,2,2,2]，通道 64→128→256→512
    头部：AdaptiveAvgPool2d → Linear(512, num_classes)
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.in_channels = 64

        # Stem: 3x3 conv stride=1 instead of 7x7 stride=2 + maxpool
        # Stem: 3x3 卷积 stride=1，替代 7x7 stride=2 + maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool for CIFAR / CIFAR 不使用 maxpool

        # Residual layer groups / 残差层组
        # [2,2,2,2] blocks, channels: 64→128→256→512, strides: 1→2→2→2
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # Classification head / 分类头部
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Weight initialization / 权重初始化
        self._initialize_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Build a group of BasicBlocks / 构建一组 BasicBlock。
        First block may downsample (stride>1), subsequent blocks have stride=1.
        第一个块可能下采样（stride>1），后续块 stride=1。
        """
        layers: list[nn.Module] = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """
        Kaiming initialization for conv layers, constant for BN.
        卷积层使用 Kaiming 初始化，BN 层使用常数初始化。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        out = self.relu(self.bn1(self.conv1(x)))
        # Residual layers / 残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Head
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def create_model(num_classes: int = 100) -> ResNet18:
    """
    Factory function: create a ResNet-18 instance.
    工厂函数：创建 ResNet-18 实例。
    """
    return ResNet18(num_classes=num_classes)


def get_feature_extractor_state(model: ResNet18) -> OrderedDict[str, Any]:
    """
    Extract state dict without FC layer, for transfer learning.
    提取不含 FC 层的 state dict，供迁移学习使用。

    The returned weights can be loaded into a new ResNet18 model with a
    different num_classes, enabling transfer learning to other datasets (e.g. CIFAR-10).
    返回的权重可以加载到具有不同 num_classes 的新 ResNet18 模型中，
    实现到其他数据集（如 CIFAR-10）的迁移学习。
    """
    return OrderedDict(
        (k, v) for k, v in model.state_dict().items() if not k.startswith("fc.")
    )
