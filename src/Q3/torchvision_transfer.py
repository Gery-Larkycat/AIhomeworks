"""
Transfer learning using PyTorch pretrained ResNet-18 (ImageNet → CIFAR-10).
使用 PyTorch 预训练 ResNet-18 的迁移学习（ImageNet → CIFAR-10）。

Loads torchvision's official ImageNet-pretrained ResNet-18, replaces FC
with CIFAR-10 classification (10 classes), freezes backbone, trains FC only.

加载 torchvision 官方 ImageNet 预训练 ResNet-18，替换 FC 为 CIFAR-10
分类（10 类），冻结 backbone，仅训练 FC 层。

Architecture: torchvision ResNet-18 (224x224 input, 7x7 stem + maxpool).
Input images are resized from CIFAR-10's 32x32 to 224x224.
"""

import dataclasses
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from .checkpoint import save_training_history
from .config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AugmentationConfig,
    TorchvisionTransferConfig,
    TrainConfig,
)
from .train import train


# ---------------------------------------------------------------------------
# Model preparation / 模型准备
# ---------------------------------------------------------------------------


def load_torchvision_pretrained(
    target_num_classes: int = 10,
) -> nn.Module:
    """
    加载 torchvision 预训练 ResNet-18，替换 FC 为目标类别数。

    Load torchvision pretrained ResNet-18 with ImageNet weights,
    replace the final FC layer for target classification.

    预训练权重来自 ImageNet（1000 类），FC 层替换为
    Linear(512, target_num_classes)，其他层保留预训练权重。

    Args:
        target_num_classes: 目标分类数（CIFAR-10 = 10）

    Returns:
        torchvision ResNet-18 模型（backbone 保留 ImageNet 权重，FC 随机初始化）
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, target_num_classes)
    return model


def freeze_backbone_tv(model: nn.Module) -> None:
    """
    冻结除 FC 外的所有参数（conv, bn, avgpool）。

    torchvision ResNet-18 的分类层属性名也是 'fc'，
    与自定义 ResNet18 结构一致，过滤逻辑相同。

    in-place 操作，仅 FC 层保持 requires_grad=True。
    """
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def print_tv_transfer_summary(model: nn.Module) -> None:
    """
    打印 torchvision 迁移模型摘要：冻结/可训练参数数。
    """
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen params: {frozen:,}")
    print(f"  Trainable params: {trainable:,}")
    # FC: 512 * 10 + 10 = 5130
    print(f"  FC params: {sum(p.numel() for p in model.fc.parameters()):,}")


# ---------------------------------------------------------------------------
# Data transforms & loaders / 数据变换与加载
# ---------------------------------------------------------------------------


def _build_tv_transforms(
    image_size: int = 224,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    augment: bool = False,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    构建 torchvision 迁移学习的 train/test 变换管线。

    所有管线都以 Resize(image_size) 开始（CIFAR-10 32x32 → 224x224），
    使用 ImageNet 归一化统计量。augment 时添加 HFlip + Crop + ColorJitter。

    Args:
        image_size: 目标图像尺寸（默认 224）
        mean: 归一化均值（默认 ImageNet 统计量）
        std: 归一化标准差（默认 ImageNet 统计量）
        augment: 是否在 train 管线中添加数据增强

    Returns:
        (train_transform, test_transform)
    """
    # Train pipeline / 训练变换管线
    train_pipeline: list[nn.Module] = [transforms.Resize(image_size)]
    if augment:
        train_pipeline.extend([
            transforms.RandomHorizontalFlip(),
            # ~12.5% padding, 与 CIFAR RandomCrop(32, padding=4) 比例一致
            transforms.RandomCrop(
                image_size,
                padding=int(image_size * 0.125),
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2,
            ),
        ])
    train_pipeline.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Test pipeline / 测试变换管线（仅 Resize + Normalize）
    test_pipeline: list[nn.Module] = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(train_pipeline), transforms.Compose(test_pipeline)


def get_cifar10_224_loaders(
    config: TorchvisionTransferConfig,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 DataLoader（224x224 + ImageNet 归一化）。

    与 get_cifar10_loaders 结构一致，但将图像上采样到 224x224
    并使用 ImageNet 归一化统计量（匹配 torchvision 预训练模型）。

    Args:
        config: TorchvisionTransferConfig（需有 image_size, mean, std,
                batch_size, num_workers, pin_memory, data_root 字段）
        augment: 是否使用训练增强（HFlip + Crop + ColorJitter）

    Returns:
        (train_loader, test_loader)
    """
    train_tf, test_tf = _build_tv_transforms(
        image_size=config.image_size,
        mean=config.mean,
        std=config.std,
        augment=augment,
    )

    train_dataset = datasets.CIFAR10(
        root=str(config.data_root),
        train=True,
        download=True,
        transform=train_tf,
    )
    test_dataset = datasets.CIFAR10(
        root=str(config.data_root),
        train=False,
        download=True,
        transform=test_tf,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Config conversion / 配置转换
# ---------------------------------------------------------------------------


def _to_train_config(
    tv_config: TorchvisionTransferConfig,
) -> TrainConfig:
    """
    将 TorchvisionTransferConfig 转为 TrainConfig。

    字段名一一对应（num_classes, batch_size, epochs, learning_rate 等）。
    TorchvisionTransferConfig 独有的 image_size 在转换中被忽略
    （仅用于数据加载，不影响训练循环）。
    """
    train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    overrides = {
        f: getattr(tv_config, f)
        for f in train_fields
        if hasattr(tv_config, f)
    }
    return TrainConfig(**overrides)


# ---------------------------------------------------------------------------
# Public API / 公共接口
# ---------------------------------------------------------------------------


def run_torchvision_transfer(
    config: TorchvisionTransferConfig,
) -> dict[str, list[float]]:
    """
    torchvision 预训练 ResNet-18 迁移学习主流程。

    Transfer learning pipeline using torchvision pretrained ResNet-18:
    1. 加载 ImageNet 预训练权重（自动下载）
    2. 替换 FC 为 CIFAR-10 分类（10 类）
    3. 冻结 backbone，仅 FC 可训练
    4. 训练 FC 层
    5. 保存训练历史

    Args:
        config: TorchvisionTransferConfig 迁移学习配置

    Returns:
        训练历史 dict（train_loss, train_acc, test_loss, test_acc, lr）
    """
    print("=" * 60)
    print("Torchvision Pretrained ResNet-18 → CIFAR-10")
    print("PyTorch 预训练 ResNet-18 迁移学习")
    print("=" * 60)
    print(f"  Image size: {config.image_size}x{config.image_size}")
    print(f"  Target classes: {config.num_classes}")
    print()

    # ---- 加载预训练模型 / Load pretrained model ----
    print("Loading torchvision pretrained ResNet-18 / 加载预训练模型...")
    model = load_torchvision_pretrained(config.num_classes)
    freeze_backbone_tv(model)
    print_tv_transfer_summary(model)

    # ---- 构造 TrainConfig 并加载数据 ----
    train_config = _to_train_config(config)
    train_loader, test_loader = get_cifar10_224_loaders(
        config, augment=config.augmentation.use_augmentation,
    )
    print(
        f"\nCIFAR-10: Train {len(train_loader.dataset)}"
        f" | Test {len(test_loader.dataset)} samples"
    )

    # ---- 训练 / Train ----
    print(
        f"\nStarting transfer training for {config.epochs} epochs"
        f" / 开始迁移训练 {config.epochs} 轮..."
    )
    history = train(model, train_loader, test_loader, train_config)

    # ---- 保存 / Save ----
    save_training_history(history, train_config)

    print("\nTorchvision transfer training complete. / 迁移训练完成。")
    return history
