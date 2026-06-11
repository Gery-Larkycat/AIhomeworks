"""
VGG-16 implementation from scratch, adapted for CIFAR (32x32) images.
从零实现的 VGG-16，适配 CIFAR（32x32）图像。

Architecture changes from standard ImageNet VGG-16:
- Only 3 MaxPool layers (blocks 1/2/3) instead of 5, preserving spatial resolution
  for 32x32 inputs: 32→16→8→4 (blocks 4/5 keep 4x4)
- FC simplified to 2 layers: 512→512→num_classes (original: 4096→4096→1000)
- BatchNorm after each conv (optional via use_bn parameter)
- Dropout before each FC layer

Standard VGG-16 cfg: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                       512, 512, 512, 'M', 512, 512, 512, 'M']
CIFAR-10 adaptation:  Block 1/2/3 保留 MaxPool，Block 4/5 去掉 MaxPool
"""

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

# VGG-16 标准配置：通道数列表，'M' 表示 MaxPool
# 标准 VGG-16 有 5 个 MaxPool；CIFAR 适配仅保留前 3 个
_VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
              512, 512, 512, 512, 512, 512]


def _make_layers(cfg: list, use_bn: bool = True) -> nn.Sequential:
    """
    根据 VGG 配置列表构建卷积层。
    Build convolutional layers from VGG config list.

    Args:
        cfg:    通道数列表，'M' 表示 MaxPool2d(2,2)
        use_bn: 是否在 Conv 后加 BatchNorm2d
    """
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(
                in_channels, v, kernel_size=3, padding=1, bias=not use_bn,
            ))
            if use_bn:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """
    VGG-16 with CIFAR-adapted architecture.
    适配 CIFAR 的 VGG-16：3 个 MaxPool（32→16→8→4），2 层 FC。

    Args:
        num_classes:  分类数（CIFAR-10=10）
        dropout_rate: FC 前的 Dropout 概率（0=禁用）
        use_bn:       是否使用 BatchNorm（False 时 Conv 后无 BN）
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.0,
        use_bn: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # 卷积特征提取：VGG-16 标准结构，CIFAR 适配（仅 3 个 MaxPool）
        self.features = _make_layers(_VGG16_CFG, use_bn=use_bn)

        # 自适应全局池化 → 统一为 1x1 特征图
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2 层 FC 分类器（原始 VGG 为 3 层 4096/4096/1000）
        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def create_model(
    num_classes: int = 10,
    dropout_rate: float = 0.0,
    use_bn: bool = True,
) -> VGG16:
    """Factory function: create a VGG-16 instance."""
    return VGG16(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_bn=use_bn,
    )


def get_feature_extractor_state(model: VGG16) -> OrderedDict[str, Any]:
    """
    Extract state dict without FC layer keys, for transfer learning.
    提取不含 FC 层的 state dict，供迁移学习使用。
    """
    return OrderedDict(
        (k, v) for k, v in model.state_dict().items()
        if not k.startswith("fc1.") and not k.startswith("fc2.")
    )
