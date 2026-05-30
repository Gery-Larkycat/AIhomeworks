"""
Q1 CIFAR-10 数据加载。
CIFAR-10 data loading for Q1.

- get_cifar10_datasets: 返回 Dataset，由 skorch 训练器负责创建 DataLoader。
- get_cifar10_loaders: 返回 DataLoader，供评估和可视化使用。
"""

from torch.utils.data import DataLoader
from torchvision import datasets

from utils.augment import build_test_transforms, build_train_transforms


def get_cifar10_datasets(config):
    """
    获取 CIFAR-10 训练集和测试集（带增强）。
    Get CIFAR-10 train and test datasets with augmentation.

    Args:
        config: 鸭子类型配置，需有 data_root, mean, std, augmentation 字段

    Returns:
        (train_dataset, test_dataset) — torchvision Datasets with transforms
    """
    train_transform = build_train_transforms(
        config.augmentation, config.mean, config.std
    )
    test_transform = build_test_transforms(config.mean, config.std)

    train_dataset = datasets.CIFAR10(
        root=str(config.data_root),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(config.data_root),
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


def get_cifar10_test_only(config):
    """
    仅获取 CIFAR-10 测试集（用于评估已训练模型）。
    Get CIFAR-10 test dataset only (for evaluating trained models).
    """
    test_transform = build_test_transforms(config.mean, config.std)
    return datasets.CIFAR10(
        root=str(config.data_root),
        train=False,
        download=True,
        transform=test_transform,
    )


def get_cifar10_loaders(config) -> tuple[DataLoader, DataLoader]:
    """
    创建 CIFAR-10 训练和测试 DataLoader。
    Create CIFAR-10 train and test DataLoaders.

    包装 get_cifar10_datasets + DataLoader 创建，供评估和可视化使用。

    Args:
        config: 鸭子类型配置，需有 data_root, mean, std, augmentation,
                batch_size, num_workers, pin_memory 字段

    Returns:
        (train_loader, test_loader)
    """
    train_ds, test_ds = get_cifar10_datasets(config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return train_loader, test_loader
