"""
Tests for CIFAR-10 data loading (Q1).
CIFAR-10 数据加载测试。
"""

import torch

from src.Q1.config import Q1TrainConfig
from src.Q1.data import get_cifar10_datasets, get_cifar10_loaders, get_cifar10_test_only


def _small_config() -> Q1TrainConfig:
    return Q1TrainConfig(batch_size=4, num_workers=0, pin_memory=False)


class TestDatasetShapes:

    def test_train_size(self):
        train_ds, _ = get_cifar10_datasets(_small_config())
        assert len(train_ds) == 50_000

    def test_test_size(self):
        _, test_ds = get_cifar10_datasets(_small_config())
        assert len(test_ds) == 10_000


class TestLoaderShapes:

    def test_batch_shape(self):
        train_loader, _ = get_cifar10_loaders(_small_config())
        images, labels = next(iter(train_loader))
        assert images.shape == (4, 3, 32, 32)
        assert labels.shape == (4,)


class TestLabelRange:

    def test_labels_in_range(self):
        _, test_loader = get_cifar10_loaders(_small_config())
        _, labels = next(iter(test_loader))
        assert labels.min() >= 0
        assert labels.max() <= 9


class TestNormalization:

    def test_value_range(self):
        train_loader, _ = get_cifar10_loaders(_small_config())
        images, _ = next(iter(train_loader))
        assert images.min() > -5
        assert images.max() < 5


class TestTestOnly:

    def test_returns_test_dataset(self):
        ds = get_cifar10_test_only(_small_config())
        assert len(ds) == 10_000
