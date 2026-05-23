"""
CIFAR-100 data loading module.
CIFAR-100 数据加载模块。

Train transforms delegate to augment.py for rich augmentation;
test transforms use only ToTensor + Normalize.
训练变换委托给 augment.py 实现丰富增强;
测试变换仅使用 ToTensor + Normalize。
"""

import torch
from torch.utils.data import DataLoader, random_split
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


def get_cifar100_search_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    创建搜索专用的训练子集和验证集 DataLoader。
    Create train-subset and validation DataLoaders for hyperparameter search.

    从 CIFAR-100 训练集按 config.val_ratio 划分。
    使用固定种子保证可复现。

    训练子集使用增强 transforms，验证集使用测试 transforms
    （仅 ToTensor + Normalize），保证验证指标无偏。

    Returns:
        (train_subset_loader, val_loader)
    """
    train_transform = build_train_transforms(
        config.augmentation, config
    )
    val_transform = build_test_transforms(config)

    # 加载完整训练集 / Load full training set
    full_train = datasets.CIFAR100(
        root=str(config.data_root),
        train=True,
        download=True,
        transform=train_transform,
    )

    # 计算划分大小 / Calculate split sizes
    total = len(full_train)
    val_size = int(total * config.val_ratio)
    train_size = total - val_size

    # 固定种子划分 / Fixed-seed split for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_train, [train_size, val_size],
        generator=generator,
    )

    # 验证集需要用无增强 transforms，通过 Subset 覆盖 transform
    # val_subset 共享 full_train 的 transform，需替换为 val_transform
    # 方案：创建一个独立的 CIFAR100 实例用于 val
    val_dataset = datasets.CIFAR100(
        root=str(config.data_root),
        train=True,
        download=True,
        transform=val_transform,
    )
    # random_split 返回的 Subset 保存了索引，用相同索引但不同 dataset
    val_indices = val_subset.indices
    val_subset_new = torch.utils.data.Subset(val_dataset, val_indices)

    train_subset_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_subset_new,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_subset_loader, val_loader
