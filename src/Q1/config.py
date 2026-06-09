"""
Q1 训练配置：VGG-16 CIFAR-10。
Training configuration for VGG-16 on CIFAR-10.
"""

from dataclasses import dataclass
from pathlib import Path

from utils.config import (
    AugmentationConfig,
    CIFAR10_MEAN,
    CIFAR10_STD,
)


@dataclass(frozen=True)
class Q1TrainConfig:
    """
    Q1 CIFAR-10 训练配置。不可变的冻结 dataclass。
    Q1 CIFAR-10 training config. Immutable frozen dataclass.
    """

    # -- Paths / 路径 --
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("outputs/Q1/checkpoints")

    # -- Model / 模型 --
    num_classes: int = 10
    dropout_rate: float = 0.5
    use_bn: bool = True
    model_name: str = "vgg16"

    # -- Training / 训练超参数 --
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1

    # -- Optimizer & Scheduler / 优化器与调度器 --
    optimizer_type: str = "sgd"
    scheduler_type: str = "cosine"
    use_amp: bool = False
    use_scheduler: bool = True       # CosineAnneLR on/off / 余弦退火开关
    use_early_stopping: bool = True  # EarlyStopping on/off / 早停开关

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
