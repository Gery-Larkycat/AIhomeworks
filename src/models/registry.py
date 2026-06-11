"""
任务注册表：每道作业题目的差异化信息 + 统一配置工厂。
Task registry: per-task差异化信息 + unified config factory.

设计动机：
- 三道题仅在模型、数据集、默认超参上有差异，其余代码完全相同。
- TaskSpec 将这些差异封装为一个值对象，避免重复代码。
- 注册表模式使得添加新题目只需一行 register() 调用。

开闭原则（OCP）实现：
- 新增模型/数据集 → 注册新 TaskSpec，无需修改已有代码。
- 所有共享管线（训练、评估、搜索、消融）通过 TaskSpec 获取差异化参数。
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from utils.config import (
    AugmentationConfig,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)


# ---------------------------------------------------------------------------
# TaskSpec / 任务规格
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """
    一道训练题目的全部差异化信息。
    Everything unique to one training task.

    设计思路：将「模型 + 数据集 + 默认超参 + 搜索后缀」封装为一个
    不可变值对象，供共享管线通过 get_spec(name) 获取。
    新题目只需注册一个新的 TaskSpec，零改动共享代码。

    Attributes:
        name:                  任务名（"Q1", "Q2", "Q3"）
        model_class:           模型类（VGG16, ResNet18）
        model_name:            模型名，用于检查点文件名前缀
        dataset_name:          数据集名（"CIFAR-10", "CIFAR-100"）
        num_classes:           分类数
        default_overrides:     TrainConfig 构造时的覆盖参数 dict
        save_feature_extractor: 是否额外保存特征提取器（Q3 迁移学习需要）
        search_suffix:         超参搜索结果文件名后缀
    """
    name: str
    model_class: type
    model_name: str
    dataset_name: str
    num_classes: int
    default_overrides: dict
    save_feature_extractor: bool = False
    search_suffix: str = "hp_search"


# ---------------------------------------------------------------------------
# Registry / 注册表
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> None:
    """
    注册一个任务规格。同名注册会覆盖旧值。
    Register a TaskSpec. Overwrites if name already exists.

    Args:
        spec: TaskSpec 实例

    Raises:
        TypeError: spec 不是 TaskSpec
    """
    if not isinstance(spec, TaskSpec):
        raise TypeError(f"Expected TaskSpec, got {type(spec)}")
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> TaskSpec:
    """
    按名称获取已注册的任务规格。
    Get registered TaskSpec by name.

    Args:
        name: 任务名（"Q1", "Q2", "Q3"）

    Returns:
        TaskSpec 实例

    Raises:
        KeyError: 未注册的任务名
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown task '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def list_tasks() -> list[str]:
    """返回所有已注册的任务名 / Return all registered task names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 内置注册 / Built-in registrations
# ---------------------------------------------------------------------------
# 模块加载时自动注册 Q1/Q2/Q3。
# 延迟导入避免循环依赖（models/__init__.py 导入本模块，
# 本模块通过 dataclass 字段引用模型类而非直接 import）。

def _register_builtins() -> None:
    """注册内置任务。延迟调用以避免循环导入。"""
    if _REGISTRY:  # 已注册则跳过
        return

    # 延迟导入：避免 models/__init__.py → registry.py → models/vgg16.py 循环
    from .vgg16 import VGG16
    from .resnet18 import ResNet18

    register(TaskSpec(
        name="Q1",
        model_class=VGG16,
        model_name="vgg16",
        dataset_name="CIFAR-10",
        num_classes=10,
        default_overrides={
            "batch_size": 256,
            "model_name": "vgg16",
            "num_classes": 10,
            "checkpoint_dir": Path("outputs/Q1/checkpoints"),
            "mean": CIFAR10_MEAN,
            "std": CIFAR10_STD,
        },
        search_suffix="vgg16_cifar10_hp_search",
    ))

    register(TaskSpec(
        name="Q2",
        model_class=ResNet18,
        model_name="resnet18",
        dataset_name="CIFAR-10",
        num_classes=10,
        default_overrides={
            "batch_size": 128,
            "model_name": "resnet18",
            "num_classes": 10,
            "checkpoint_dir": Path("outputs/Q2/checkpoints"),
            "mean": CIFAR10_MEAN,
            "std": CIFAR10_STD,
        },
        search_suffix="cifar10_hp_search",
    ))

    register(TaskSpec(
        name="Q3",
        model_class=ResNet18,
        model_name="resnet18",
        dataset_name="CIFAR-100",
        num_classes=100,
        default_overrides={
            "batch_size": 1024,
            "epochs": 150,
            "num_classes": 100,
            "model_name": "resnet18",
            "scheduler_t_max": 150,
            "patience": 8,
            "mean": CIFAR100_MEAN,
            "std": CIFAR100_STD,
            "checkpoint_dir": Path("outputs/Q3/checkpoints"),
        },
        save_feature_extractor=True,
        search_suffix="cifar100_hp_search",
    ))


_register_builtins()


# ---------------------------------------------------------------------------
# Unified config factory / 统一配置工厂
# ---------------------------------------------------------------------------


def make_config(task_name: str, **overrides):
    """
    按任务名创建统一训练配置。
    Create unified TrainConfig with per-task defaults, then apply user overrides.

    创建顺序：TrainConfig() 默认值 → TaskSpec.default_overrides → 用户 overrides。
    后者覆盖前者，确保用户参数优先级最高。

    Args:
        task_name: 任务名（"Q1", "Q2", "Q3"）
        **overrides: 用户覆盖参数（如 epochs=50, batch_size=64）

    Returns:
        TrainConfig 实例

    Raises:
        KeyError: 未注册的任务名
    """
    # 延迟导入避免循环：utils.config 本模块被 models/registry 引用
    from utils.config import TrainConfig

    spec = get_spec(task_name)
    config = TrainConfig(**spec.default_overrides)
    if overrides:
        config = dataclasses.replace(config, **overrides)
    return config
