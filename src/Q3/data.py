"""
CIFAR-100/10 data loading module.
CIFAR-100/10 数据加载模块。

Train transforms delegate to utils.augment for rich augmentation;
test transforms use only ToTensor + Normalize.
训练变换委托给 utils.augment 实现丰富增强;
测试变换仅使用 ToTensor + Normalize。

设计动机：utils.augment 使用 (mean, std) 参数签名，
而非旧版 (config) 签名，使增强管线独立于特定配置类。
"""

from torch.utils.data import DataLoader
from torchvision import datasets

from utils.augment import build_test_transforms, build_train_transforms

from .config import TrainConfig


def get_cifar100_datasets(
    config: TrainConfig,
) -> tuple[datasets.CIFAR100, datasets.CIFAR100]:
    """
    返回 (train_dataset, test_dataset) 用于 skorch 训练。
    Return (train_dataset, test_dataset) for skorch training.

    skorch 的 .fit(dataset, y=None) 直接接收 Dataset，
    无需在此创建 DataLoader（skorch 内部创建）。

    Train: full augmentation pipeline from utils.augment.
    Test:  ToTensor + Normalize only (no augmentation).
    训练集: utils.augment 的完整增强管线。
    测试集: 仅 ToTensor + Normalize（无增强）。

    Args:
        config: TrainConfig（需有 augmentation, mean, std, data_root 字段）

    Returns:
        (train_dataset, test_dataset)
    """
    # utils.augment 使用 (aug_config, mean, std) 签名
    train_transform = build_train_transforms(
        config.augmentation, config.mean, config.std,
    )
    test_transform = build_test_transforms(config.mean, config.std)

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

    return train_dataset, test_dataset


def get_cifar100_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-100 train and test DataLoaders.
    创建 CIFAR-100 训练和测试 DataLoader。

    向后兼容包装：get_cifar100_datasets + DataLoader 创建。
    Backward compat wrapper: get_cifar100_datasets + DataLoader creation.
    供未迁移到 skorch 的代码继续使用。

    Returns:
        (train_loader, test_loader)
    """
    train_ds, test_ds = get_cifar100_datasets(config)

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
