"""
CIFAR-100 data loading module.
CIFAR-100 数据加载模块。

Train transforms delegate to augment.py for rich augmentation;
test transforms use only ToTensor + Normalize.
训练变换委托给 augment.py 实现丰富增强;
测试变换仅使用 ToTensor + Normalize。
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .augment import build_test_transforms, build_train_transforms
from .config import TrainConfig


def get_cifar100_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-100 train and test DataLoaders with separate transforms.
    创建 CIFAR-100 训练和测试 DataLoader，分别使用不同变换管线。

    Train: full augmentation pipeline from augment.py.
    Test:  ToTensor + Normalize only (no augmentation).
    训练集: augment.py 的完整增强管线。
    测试集: 仅 ToTensor + Normalize（无增强）。

    Returns:
        (train_loader, test_loader)
    """
    train_transform = build_train_transforms(
        config.augmentation, config
    )
    test_transform = build_test_transforms(config)

    train_dataset = datasets.CIFAR100(
        root=str(config.data_root),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=str(config.data_root),
        train=False,
        download=True,
        transform=test_transform,
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
