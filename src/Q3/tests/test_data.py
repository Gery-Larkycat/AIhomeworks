"""
Tests for CIFAR-100 data loading.
CIFAR-100 数据加载的测试。

These tests require network access to download CIFAR-100 on first run.
这些测试首次运行时需要网络下载 CIFAR-100。
"""

import torch

from src.Q3.config import TrainConfig
from src.Q3.data import get_cifar100_loaders


def _small_config() -> TrainConfig:
    """Config with small batch for fast tests / 使用小批量的快速测试配置。"""
    return TrainConfig(
        batch_size=4,
        num_workers=0,  # Use 0 for test stability / 测试稳定性用 0
        pin_memory=False,
    )


def test_dataset_shape() -> None:
    """Images should be (B, 3, 32, 32), labels should be (B,)."""
    config = _small_config()
    train_loader, _ = get_cifar100_loaders(config)
    images, labels = next(iter(train_loader))

    assert images.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {images.shape}"
    assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"


def test_label_range() -> None:
    """Labels should be in [0, 99] for CIFAR-100."""
    config = _small_config()
    train_loader, _ = get_cifar100_loaders(config)
    _, labels = next(iter(train_loader))

    assert labels.min() >= 0, f"Min label {labels.min()} < 0"
    assert labels.max() <= 99, f"Max label {labels.max()} > 99"


def test_normalization_range() -> None:
    """After normalization, values should be roughly in [-3, 3]."""
    config = _small_config()
    train_loader, _ = get_cifar100_loaders(config)
    images, _ = next(iter(train_loader))

    assert images.min() > -5, f"Min value {images.min():.2f} suspiciously low"
    assert images.max() < 5, f"Max value {images.max():.2f} suspiciously high"


def test_test_set_size() -> None:
    """CIFAR-100 test set should have 10,000 samples."""
    config = _small_config()
    _, test_loader = get_cifar100_loaders(config)
    assert len(test_loader.dataset) == 10_000, f"Expected 10000, got {len(test_loader.dataset)}"
