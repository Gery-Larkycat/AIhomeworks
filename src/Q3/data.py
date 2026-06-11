"""
Q3 CIFAR-100/10 数据加载。
Q3 CIFAR-100/10 data loading.

- get_cifar100_datasets / get_cifar100_loaders: 委托到 utils.data
- get_cifar10_loaders: Q3 迁移学习专用，保留（处理可选 augmentation）
"""

from torch.utils.data import DataLoader
from torchvision import datasets

from utils.augment import build_test_transforms, build_train_transforms

# CIFAR-100 标准函数重导出 / Re-export standard CIFAR-100 functions
from utils.data import (  # noqa: F401
    get_datasets as get_cifar100_datasets,
    get_loaders as get_cifar100_loaders,
)


def get_cifar10_loaders(
    config,
) -> tuple[DataLoader, DataLoader]:
    """
    CIFAR-10 train/test DataLoaders，用于迁移学习。
    CIFAR-10 train/test DataLoaders for transfer learning.

    接受 TransferConfig 或 TrainConfig（鸭子类型）。
    TransferConfig 有 augmentation 字段，TrainConfig 也有；
    若无 augmentation 属性则不使用增强（仅 Normalize）。

    Args:
        config: TransferConfig 或 TrainConfig（鸭子类型，需有
                data_root, mean, std, batch_size,
                num_workers, pin_memory 字段;
                可选 augmentation 字段）

    Returns:
        (train_loader, test_loader)
    """
    test_transform = build_test_transforms(config.mean, config.std)

    # 若配置有 augmentation 属性则使用，否则仅 Normalize
    aug_config = getattr(config, "augmentation", None)
    if aug_config is not None:
        train_transform = build_train_transforms(
            aug_config, config.mean, config.std,
        )
    else:
        train_transform = test_transform

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
