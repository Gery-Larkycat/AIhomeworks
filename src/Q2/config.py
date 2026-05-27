"""
Q2 训练配置：ResNet-18 CIFAR-10。
Training configuration for ResNet-18 on CIFAR-10.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from utils.config import (
    AugmentationConfig,
    CIFAR10_MEAN,
    CIFAR10_STD,
    SearchConfig,
    generate_timestamp,
    make_run_dir,
)


@dataclass(frozen=True)
class Q2TrainConfig:
    """
    Q2 CIFAR-10 训练配置。不可变的冻结 dataclass。
    Q2 CIFAR-10 training config. Immutable frozen dataclass.
    """

    # -- Paths / 路径 --
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")

    # -- Model / 模型 --
    num_classes: int = 10
    dropout_rate: float = 0.5

    # -- Training / 训练超参数 --
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1

    # -- Optimizer & Scheduler / 优化器与调度器 --
    optimizer_type: str = "sgd"
    scheduler_type: str = "cosine"
    use_amp: bool = False

    # -- Early stopping / 早停 --
    patience: int = 10
    min_delta: float = 1e-4

    # -- Scheduler / 学习率调度 --
    scheduler_t_max: int = 200

    # -- Data loading / 数据加载 --
    num_workers: int = 0
    pin_memory: bool = True

    # -- Augmentation / 数据增强 --
    augmentation: AugmentationConfig = AugmentationConfig()

    # -- CIFAR-10 normalization / CIFAR-10 归一化 --
    mean: tuple[float, ...] = CIFAR10_MEAN
    std: tuple[float, ...] = CIFAR10_STD
