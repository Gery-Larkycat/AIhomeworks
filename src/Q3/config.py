"""
Configuration module for ResNet-18 CIFAR-100 training.
ResNet-18 CIFAR-100 训练配置模块。

All hyperparameters live here as a single source of truth.
所有超参数集中在此作为唯一真相来源。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """
    Immutable training configuration / 不可变的训练配置。
    Frozen dataclass prevents accidental mutation during training.
    冻结 dataclass 防止训练过程中意外修改。
    """

    # -- Paths / 路径配置 --
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")

    # -- Model / 模型配置 --
    num_classes: int = 100  # CIFAR-100 has 100 classes / CIFAR-100 有 100 个类别

    # -- Training / 训练超参数 --
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1

    # -- Early stopping / 早停 --
    patience: int = 20  # Epochs to wait without improvement / 无改善等待轮数
    min_delta: float = 1e-4  # Minimum accuracy improvement to qualify / 视为改善的最小准确率增量

    # -- Scheduler / 学习率调度 --
    scheduler_t_max: int = 200  # Cosine annealing period / 余弦退火周期

    # -- Data loading / 数据加载 --
    num_workers: int = 4  # Windows requires if __name__ guard when > 0 / Windows 下需 if __name__ 守卫
    pin_memory: bool = True  # Faster CPU→GPU transfer / 加速 CPU→GPU 数据传输

    # -- CIFAR-100 normalization stats / CIFAR-100 归一化统计量 --
    mean: tuple[float, ...] = (0.5071, 0.4867, 0.4408)
    std: tuple[float, ...] = (0.2675, 0.2565, 0.2761)
