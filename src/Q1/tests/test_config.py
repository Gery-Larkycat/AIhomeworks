"""
Tests for Q1TrainConfig.
Q1TrainConfig 配置测试。
"""

import dataclasses
from pathlib import Path

import pytest

from src.Q1.config import Q1TrainConfig
from utils.config import AugmentationConfig, CIFAR10_MEAN, CIFAR10_STD


class TestDefaultValues:
    """Verify Q1TrainConfig defaults."""

    def test_num_classes(self):
        assert Q1TrainConfig().num_classes == 10

    def test_dropout_rate(self):
        assert Q1TrainConfig().dropout_rate == 0.5

    def test_batch_size(self):
        assert Q1TrainConfig().batch_size == 256

    def test_epochs(self):
        assert Q1TrainConfig().epochs == 200

    def test_learning_rate(self):
        assert Q1TrainConfig().learning_rate == 0.1

    def test_momentum(self):
        assert Q1TrainConfig().momentum == 0.9

    def test_weight_decay(self):
        assert Q1TrainConfig().weight_decay == 5e-4

    def test_label_smoothing(self):
        assert Q1TrainConfig().label_smoothing == 0.1

    def test_optimizer_type(self):
        assert Q1TrainConfig().optimizer_type == "sgd"

    def test_use_bn(self):
        assert Q1TrainConfig().use_bn is True

    def test_normalization(self):
        config = Q1TrainConfig()
        assert config.mean == CIFAR10_MEAN
        assert config.std == CIFAR10_STD


class TestFrozen:

    def test_cannot_set_attribute(self):
        config = Q1TrainConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.epochs = 50  # type: ignore[misc]


class TestCustomOverride:

    def test_override_epochs(self):
        config = Q1TrainConfig(epochs=50, batch_size=64)
        assert config.epochs == 50
        assert config.batch_size == 64

    def test_override_use_bn(self):
        config = Q1TrainConfig(use_bn=False)
        assert config.use_bn is False


class TestDefaults:

    def test_checkpoint_dir(self):
        assert Q1TrainConfig().checkpoint_dir == Path("outputs/Q1/checkpoints")

    def test_augmentation_instance(self):
        assert isinstance(Q1TrainConfig().augmentation, AugmentationConfig)


class TestReplace:

    def test_creates_new_instance(self):
        config = Q1TrainConfig()
        new_config = dataclasses.replace(config, epochs=50)
        assert new_config.epochs == 50
        assert config.epochs == 200
