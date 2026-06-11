"""
统一架构验证测试：注册表、配置工厂、数据加载、CLI。
Unified architecture tests: registry, config factory, data loading, CLI.

使用参数化测试覆盖所有任务，替代 Q1/Q2 重复的测试文件。
"""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from models.registry import get_spec, make_config, list_tasks  # noqa: E402
from utils.config import TrainConfig  # noqa: E402
from utils.data import get_datasets  # noqa: E402
from utils.cli import apply_cli_overrides  # noqa: E402
import argparse  # noqa: E402


# ---------------------------------------------------------------------------
# Registry tests / 注册表测试
# ---------------------------------------------------------------------------


class TestRegistry:
    """TaskSpec 注册表基本功能测试。"""

    def test_all_tasks_registered(self):
        """Q1, Q2, Q3 都已注册。"""
        tasks = list_tasks()
        assert "Q1" in tasks
        assert "Q2" in tasks
        assert "Q3" in tasks

    def test_get_spec_raises_on_unknown(self):
        """未注册的任务名应报 KeyError。"""
        with pytest.raises(KeyError):
            get_spec("Q99")


# ---------------------------------------------------------------------------
# Unified TrainConfig tests / 统一配置测试
# ---------------------------------------------------------------------------


class TestUnifiedTrainConfig:
    """参数化测试：验证 make_config 为每个任务返回正确的默认值。"""

    @pytest.mark.parametrize("task,expected", [
        ("Q1", {
            "num_classes": 10, "model_name": "vgg16",
            "batch_size": 256,
        }),
        ("Q2", {
            "num_classes": 10, "model_name": "resnet18",
            "batch_size": 128,
        }),
        ("Q3", {
            "num_classes": 100, "model_name": "resnet18",
            "batch_size": 1024, "epochs": 150, "patience": 8,
        }),
    ])
    def test_task_defaults(self, task, expected):
        """make_config(task) 返回正确的任务默认值。"""
        cfg = make_config(task)
        for field, value in expected.items():
            assert getattr(cfg, field) == value, (
                f"{task}: {field} = {getattr(cfg, field)}, expected {value}"
            )

    def test_make_config_with_overrides(self):
        """make_config 支持用户覆盖。"""
        cfg = make_config("Q1", epochs=50, batch_size=64)
        assert cfg.epochs == 50
        assert cfg.batch_size == 64
        # 未覆盖的保持 Q1 默认值
        assert cfg.model_name == "vgg16"

    def test_frozen(self):
        """TrainConfig 是 frozen dataclass，禁止直接赋值。"""
        cfg = TrainConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.epochs = 99

    def test_replace_works(self):
        """dataclasses.replace 正常工作。"""
        cfg = make_config("Q1")
        modified = dataclasses.replace(cfg, use_bn=False, epochs=10)
        assert modified.use_bn is False
        assert modified.epochs == 10
        assert cfg.use_bn is True  # 原始不变

    def test_common_fields_present(self):
        """所有任务配置都有必要的技术开关字段。"""
        for task in ("Q1", "Q2", "Q3"):
            cfg = make_config(task)
            assert hasattr(cfg, "use_scheduler")
            assert hasattr(cfg, "use_early_stopping")
            assert hasattr(cfg, "use_bn")
            assert hasattr(cfg, "augmentation")


# ---------------------------------------------------------------------------
# Unified data loading tests / 统一数据加载测试
# ---------------------------------------------------------------------------


_DUMMY_MEAN = (0.5, 0.5, 0.5)
_DUMMY_STD = (0.5, 0.5, 0.5)


class TestUnifiedData:
    """参数化测试：验证统一数据加载对每个任务正确工作。"""

    @pytest.mark.parametrize("task,expected_train_size", [
        ("Q1", 50_000),
        ("Q2", 50_000),
        ("Q3", 50_000),
    ])
    def test_dataset_sizes(self, task, expected_train_size):
        """get_datasets 返回正确的数据集大小。"""
        cfg = make_config(task, batch_size=4, num_workers=0,
                          pin_memory=False)
        train_ds, test_ds = get_datasets(cfg)
        assert len(train_ds) == expected_train_size
        assert len(test_ds) == 10_000


# ---------------------------------------------------------------------------
# CLI override tests / CLI 覆盖测试
# ---------------------------------------------------------------------------


class TestCLIOverrides:
    """apply_cli_overrides 的统一测试。"""

    def test_empty_args(self):
        """无覆盖时返回空 dict。"""
        args = argparse.Namespace(
            epochs=None, batch_size=None, lr=None, dropout=None,
            no_bn=False, amp=False, no_augmentation=False,
            data_root=None, no_scheduler=False, no_weight_decay=False,
            no_label_smoothing=False, no_dropout=False,
            no_early_stopping=False, no_cutmix=False, no_mixup=False,
            no_geom_aug=False, no_color_aug=False, no_noise_aug=False,
            no_weather_aug=False, no_mixing_aug=False,
        )
        overrides = apply_cli_overrides(args)
        assert overrides == {}

    def test_full_override(self):
        """所有开关同时启用时全部覆盖。"""
        args = argparse.Namespace(
            epochs=50, batch_size=64, lr=0.01, dropout=0.3,
            no_bn=True, amp=True, no_augmentation=False,
            data_root="/tmp/data", no_scheduler=True,
            no_weight_decay=True, no_label_smoothing=True,
            no_dropout=True, no_early_stopping=True,
            no_cutmix=True, no_mixup=True,
            no_geom_aug=True, no_color_aug=True, no_noise_aug=True,
            no_weather_aug=True, no_mixing_aug=True,
        )
        overrides = apply_cli_overrides(args)
        assert overrides["epochs"] == 50
        assert overrides["batch_size"] == 64
        assert overrides["learning_rate"] == 0.01
        assert overrides["dropout_rate"] == 0.0  # no_dropout wins
        assert overrides["use_bn"] is False
        assert overrides["use_amp"] is True
        assert overrides["use_scheduler"] is False
        assert overrides["weight_decay"] == 0.0
        assert overrides["label_smoothing"] == 0.0
        assert overrides["use_early_stopping"] is False
        assert overrides["data_root"] == Path("/tmp/data")
        # Augmentation overrides
        aug = overrides["augmentation"]
        assert aug.use_cutmix is False
        assert aug.use_mixup is False
        assert aug.use_geom_aug is False
        assert aug.use_color_aug is False
        assert aug.use_noise_aug is False
        assert aug.use_weather_aug is False
        assert aug.use_mixing_aug is False


# ---------------------------------------------------------------------------
# Backward compatibility tests / 向后兼容测试
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """验证旧的 import 路径仍然有效。"""

    def test_q1_config_import(self):
        """from Q1.config import Q1TrainConfig 仍然有效。"""
        from Q1.config import Q1TrainConfig
        cfg = Q1TrainConfig()
        assert cfg.model_name == "vgg16"
        assert cfg.batch_size == 256

    def test_q2_config_import(self):
        """from Q2.config import Q2TrainConfig 仍然有效。"""
        from Q2.config import Q2TrainConfig
        cfg = Q2TrainConfig()
        assert cfg.model_name == "resnet18"
        assert cfg.batch_size == 128

    def test_q3_config_import(self):
        """from Q3.config import TrainConfig 仍然有效。"""
        from Q3.config import TrainConfig as Q3TrainConfig
        # TrainConfig 是通用类，Q3 差异化通过 make_config("Q3") 获取
        assert Q3TrainConfig is TrainConfig
        # 用 make_config 获取 Q3 默认值
        cfg = make_config("Q3")
        assert cfg.num_classes == 100

    def test_q1_model_import(self):
        """from Q1.model import VGG16 仍然有效。"""
        from Q1.model import VGG16
        model = VGG16(num_classes=10)
        assert isinstance(model, torch.nn.Module)

    def test_q2_model_import(self):
        """from Q2.model import ResNet18 仍然有效。"""
        from Q2.model import ResNet18
        model = ResNet18(num_classes=10)
        assert isinstance(model, torch.nn.Module)

    def test_q1_data_import(self):
        """from Q1.data import get_cifar10_datasets 仍然有效。"""
        from Q1.data import get_cifar10_datasets
        assert callable(get_cifar10_datasets)

    def test_q2_data_import(self):
        """from Q2.data import get_cifar10_loaders 仍然有效。"""
        from Q2.data import get_cifar10_loaders
        assert callable(get_cifar10_loaders)

    def test_q1_training_import(self):
        """from Q1.training import train_vgg 仍然有效。"""
        from Q1.training import train_vgg
        assert callable(train_vgg)

    def test_q2_training_import(self):
        """from Q2.training import train_resnet 仍然有效。"""
        from Q2.training import train_resnet
        assert callable(train_resnet)
