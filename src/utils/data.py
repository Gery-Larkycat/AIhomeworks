"""
统一数据加载模块：CIFAR-10 / CIFAR-100。
Unified data loading for CIFAR-10 and CIFAR-100.

设计动机：
- Q1/Q2 都加载 CIFAR-10，代码 100% 相同（93 行）。
- Q3 加载 CIFAR-100，逻辑相同仅数据集类不同。
- 统一为三个函数 + 数据集注册表，通过 config 自动选择数据集。

开闭原则：新增数据集只需在 _DATASET_MAP 注册。
"""

from torch.utils.data import DataLoader
from torchvision import datasets

from .augment import build_train_transforms, build_test_transforms


# ---------------------------------------------------------------------------
# Dataset registry / 数据集注册表
# ---------------------------------------------------------------------------

_DATASET_MAP = {
    "CIFAR-10": datasets.CIFAR10,
    "CIFAR-100": datasets.CIFAR100,
}


def _resolve_dataset_cls(config) -> type:
    """
    根据 config 解析数据集类。
    Resolve dataset class from config.

    优先使用 config.dataset_name（如有），
    否则按 num_classes 推断：>10 → CIFAR-100，其他 → CIFAR-10。

    Args:
        config: 鸭子类型配置，需有 num_classes 字段，
                可选 dataset_name 字段

    Returns:
        torchvision 数据集类（datasets.CIFAR10 或 datasets.CIFAR100）
    """
    # TaskSpec 注册了 dataset_name，通过 make_config 创建的配置会有
    if hasattr(config, "dataset_name") and config.dataset_name:
        ds_cls = _DATASET_MAP.get(config.dataset_name)
        if ds_cls is not None:
            return ds_cls
    # 回退：按类别数推断
    if getattr(config, "num_classes", 10) > 10:
        return datasets.CIFAR100
    return datasets.CIFAR10


# ---------------------------------------------------------------------------
# Public API / 公共接口
# ---------------------------------------------------------------------------


def get_datasets(config) -> tuple:
    """
    获取训练集和测试集（带增强）。
    Get (train_dataset, test_dataset) with augmentation.

    根据 config 自动选择 CIFAR-10 或 CIFAR-100。
    训练集使用完整增强管线，测试集仅 ToTensor + Normalize。

    Args:
        config: 鸭子类型配置，需有 augmentation, mean, std, data_root 字段

    Returns:
        (train_dataset, test_dataset) — torchvision Datasets with transforms
    """
    ds_cls = _resolve_dataset_cls(config)
    train_transform = build_train_transforms(
        config.augmentation, config.mean, config.std
    )
    test_transform = build_test_transforms(config.mean, config.std)

    train_dataset = ds_cls(
        root=str(config.data_root),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = ds_cls(
        root=str(config.data_root),
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


def get_test_only(config):
    """
    仅获取测试集（用于评估已训练模型）。
    Get test dataset only (for evaluating trained models).

    Args:
        config: 鸭子类型配置

    Returns:
        test_dataset — torchvision Dataset with test transform
    """
    ds_cls = _resolve_dataset_cls(config)
    test_transform = build_test_transforms(config.mean, config.std)
    return ds_cls(
        root=str(config.data_root),
        train=False,
        download=True,
        transform=test_transform,
    )


def get_loaders(config) -> tuple[DataLoader, DataLoader]:
    """
    创建训练和测试 DataLoader。
    Create (train_loader, test_loader) for evaluation.

    skorch 训练使用 Dataset 即可，但评估（per_class_accuracy、confusion_matrix）
    需要 DataLoader。

    Args:
        config: 鸭子类型配置，需有 augmentation, mean, std, data_root,
                batch_size, num_workers, pin_memory 字段

    Returns:
        (train_loader, test_loader)
    """
    train_ds, test_ds = get_datasets(config)

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
