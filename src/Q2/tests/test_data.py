"""
Tests for CIFAR-10 data loading.
CIFAR-10 数据加载测试。

These tests require network access to download CIFAR-10 on first run.
这些测试首次运行时需要网络下载 CIFAR-10。
"""

import torch

from src.Q2.config import Q2TrainConfig
from src.Q2.data import get_cifar10_datasets, get_cifar10_loaders, get_cifar10_test_only


def _small_config() -> Q2TrainConfig:
    """Config with small batch for fast tests / 使用小批量的快速测试配置。"""
    return Q2TrainConfig(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )


class TestDatasetShapes:
    """Test dataset sizes / 数据集大小测试。"""

    def test_train_size(self):
        config = _small_config()
        train_ds, _ = get_cifar10_datasets(config)
        assert len(train_ds) == 50_000, f"Expected 50000, got {len(train_ds)}"

    def test_test_size(self):
        config = _small_config()
        _, test_ds = get_cifar10_datasets(config)
        assert len(test_ds) == 10_000, f"Expected 10000, got {len(test_ds)}"


class TestLoaderShapes:
    """Test DataLoader batch shapes / DataLoader 批次形状测试。"""

    def test_train_batch_shape(self):
        config = _small_config()
        train_loader, _ = get_cifar10_loaders(config)
        images, labels = next(iter(train_loader))
        assert images.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {images.shape}"
        assert labels.shape == (4,), f"Expected (4,), got {labels.shape}"

    def test_test_batch_shape(self):
        config = _small_config()
        _, test_loader = get_cifar10_loaders(config)
        images, labels = next(iter(test_loader))
        assert images.shape[1:] == (3, 32, 32)
        assert labels.dim() == 1


class TestLabelRange:
    """Labels should be in [0, 9] for CIFAR-10."""

    def test_train_labels(self):
        config = _small_config()
        train_loader, _ = get_cifar10_loaders(config)
        _, labels = next(iter(train_loader))
        assert labels.min() >= 0, f"Min label {labels.min()} < 0"
        assert labels.max() <= 9, f"Max label {labels.max()} > 9"


class TestNormalization:
    """After normalization, values should be roughly in [-5, 5]."""

    def test_value_range(self):
        config = _small_config()
        train_loader, _ = get_cifar10_loaders(config)
        images, _ = next(iter(train_loader))
        assert images.min() > -5, f"Min value {images.min():.2f} suspiciously low"
        assert images.max() < 5, f"Max value {images.max():.2f} suspiciously high"


class TestTestOnly:
    """get_cifar10_test_only returns correct dataset."""

    def test_returns_test_dataset(self):
        config = _small_config()
        ds = get_cifar10_test_only(config)
        assert len(ds) == 10_000, f"Expected 10000, got {len(ds)}"
