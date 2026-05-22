"""
CIFAR-100 data loading module.
CIFAR-100 数据加载模块。

No additional augmentation — only Normalize with CIFAR-100 statistics.
不做额外增强——仅使用 CIFAR-100 统计量进行归一化。
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainConfig


def get_transforms(config: TrainConfig) -> transforms.Compose:
    """
    Build transform pipeline with normalization only.
    构建仅含归一化的变换管线。
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def get_cifar100_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-100 train and test DataLoaders.
    创建 CIFAR-100 训练和测试数据加载器。

    Returns:
        (train_loader, test_loader)
    """
    transform = get_transforms(config)

    train_dataset = datasets.CIFAR100(
        root=str(config.data_root),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR100(
        root=str(config.data_root),
        train=False,
        download=True,
        transform=transform,
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
