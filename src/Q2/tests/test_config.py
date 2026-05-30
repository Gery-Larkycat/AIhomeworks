"""
Tests for Q2TrainConfig.
Q2TrainConfig 配置测试。

Validates: default values, immutability, field access, replace.
验证：默认值、不可变性、字段访问、替换。
"""

import dataclasses
from pathlib import Path

import pytest

from src.Q2.config import Q2TrainConfig
from utils.config import AugmentationConfig, CIFAR10_MEAN, CIFAR10_STD


class TestDefaultValues:
    """Verify all Q2TrainConfig defaults match CIFAR-10 spec."""

    def test_num_classes(self):
        config = Q2TrainConfig()
        assert config.num_classes == 10

    def test_dropout_rate(self):
        config = Q2TrainConfig()
        assert config.dropout_rate == 0.5

    def test_batch_size(self):
        config = Q2TrainConfig()
        assert config.batch_size == 128

    def test_epochs(self):
        config = Q2TrainConfig()
        assert config.epochs == 200

    def test_learning_rate(self):
        config = Q2TrainConfig()
        assert config.learning_rate == 0.1

    def test_momentum(self):
        config = Q2TrainConfig()
        assert config.momentum == 0.9

    def test_weight_decay(self):
        config = Q2TrainConfig()
        assert config.weight_decay == 5e-4

    def test_label_smoothing(self):
        config = Q2TrainConfig()
        assert config.label_smoothing == 0.1

    def test_optimizer_type(self):
        config = Q2TrainConfig()
        assert config.optimizer_type == "sgd"

    def test_scheduler_type(self):
        config = Q2TrainConfig()
        assert config.scheduler_type == "cosine"

    def test_use_amp(self):
        config = Q2TrainConfig()
        assert config.use_amp is False

    def test_patience(self):
        config = Q2TrainConfig()
        assert config.patience == 10

    def test_min_delta(self):
        config = Q2TrainConfig()
        assert config.min_delta == 1e-4

    def test_scheduler_t_max(self):
        config = Q2TrainConfig()
        assert config.scheduler_t_max == 200

    def test_num_workers(self):
        config = Q2TrainConfig()
        assert config.num_workers == 0

    def test_pin_memory(self):
        config = Q2TrainConfig()
        assert config.pin_memory is True

    def test_normalization(self):
        config = Q2TrainConfig()
        assert config.mean == CIFAR10_MEAN
        assert config.std == CIFAR10_STD


class TestFrozen:
    """Q2TrainConfig is frozen (immutable)."""

    def test_cannot_set_attribute(self):
        config = Q2TrainConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.epochs = 50  # type: ignore[misc]

    def test_cannot_set_path(self):
        config = Q2TrainConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.checkpoint_dir = Path("other")  # type: ignore[misc]


class TestCustomOverride:
    """Custom values via constructor work correctly."""

    def test_override_epochs_and_batch_size(self):
        config = Q2TrainConfig(epochs=50, batch_size=64)
        assert config.epochs == 50
        assert config.batch_size == 64
        # Other fields keep defaults / 其他字段保持默认值
        assert config.num_classes == 10
        assert config.learning_rate == 0.1

    def test_override_learning_rate(self):
        config = Q2TrainConfig(learning_rate=0.01)
        assert config.learning_rate == 0.01

    def test_override_checkpoint_dir(self):
        config = Q2TrainConfig(checkpoint_dir=Path("custom/dir"))
        assert config.checkpoint_dir == Path("custom/dir")


class TestDefaults:
    """Specific default path and augmentation checks."""

    def test_checkpoint_dir_default(self):
        config = Q2TrainConfig()
        assert config.checkpoint_dir == Path("outputs/Q2/checkpoints")

    def test_data_root_default(self):
        config = Q2TrainConfig()
        assert config.data_root == Path("data")

    def test_augmentation_default(self):
        config = Q2TrainConfig()
        assert isinstance(config.augmentation, AugmentationConfig)


class TestReplace:
    """dataclasses.replace creates new instance, original unchanged."""

    def test_replace_creates_new_instance(self):
        config = Q2TrainConfig()
        new_config = dataclasses.replace(config, epochs=50)
        assert new_config.epochs == 50
        assert config.epochs == 200  # Original unchanged / 原始不变

    def test_replace_multiple_fields(self):
        config = Q2TrainConfig()
        new_config = dataclasses.replace(
            config, epochs=30, batch_size=256, learning_rate=0.05,
        )
        assert new_config.epochs == 30
        assert new_config.batch_size == 256
        assert new_config.learning_rate == 0.05
