"""
Configuration module for ResNet-18 CIFAR-100 training and hyperparameter search.
ResNet-18 CIFAR-100 训练和超参数搜索配置模块。

All hyperparameters live here as a single source.
所有超参数集中在此作为唯一来源。
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
    batch_size: int = 1024
    epochs: int = 100
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1

    # -- Optimizer & Scheduler / 优化器与调度器 --
    optimizer_type: str = "sgd"  # "sgd", "adam", "adamw", "rmsprop", "nadam"
    scheduler_type: str = "cosine"  # "cosine", "constant", "step"
    use_amp: bool = False  # Mixed precision (FP16) training / 混合精度训练

    # -- Early stopping / 早停 --
    patience: int = 6  # Epochs to wait without improvement / 无改善等待轮数
    min_delta: float = (
        1e-4  # Minimum accuracy improvement to qualify / 视为改善的最小准确率增量
    )

    # -- Scheduler / 学习率调度 --
    scheduler_t_max: int = 100  # Cosine annealing period / 余弦退火周期

    # -- Data loading / 数据加载 --
    num_workers: int = (
        4  # Windows requires if __name__ guard when > 0 / Windows 下需 if __name__ 守卫
    )
    pin_memory: bool = True  # Faster CPU→GPU transfer / 加速 CPU→GPU 数据传输

    # -- CIFAR-100 normalization stats / CIFAR-100 归一化统计量 --
    mean: tuple[float, ...] = (0.5071, 0.4867, 0.4408)
    std: tuple[float, ...] = (0.2675, 0.2565, 0.2761)


@dataclass(frozen=True)
class HyperparamRange:
    """
    Range definition for a searchable continuous hyperparameter.
    可搜索连续超参数的范围定义。

    distribution: "uniform" (线性均匀) or "log_uniform" (对数均匀).
    log_uniform 适用于跨越多个数量级的参数（如 learning_rate, weight_decay）。
    """

    low: float
    high: float
    distribution: str = "uniform"  # "uniform" or "log_uniform"


@dataclass(frozen=True)
class SearchConfig:
    """
    Configuration for evolutionary hyperparameter search.
    进化超参数搜索配置。

    All search-related settings live here. Set any parameter range to None to
    exclude it from the search. search.py reads from this config exclusively.
    所有搜索相关设置集中在此。将任意参数范围设为 None 即可跳过该参数的搜索。
    search.py 仅从此配置读取搜索空间。
    """

    # -- Evolutionary algorithm / 演化算法参数 --
    search_epochs: int = 5  # Epochs per individual evaluation / 每个个体的训练轮数
    population_size: int = 8  # μ: number of parents / 种群大小
    offspring_per_gen: int = 4  # λ: offspring per generation / 每代后代数
    num_generations: int = 3  # G: number of generations / 演化代数
    tournament_size: int = 3  # Tournament selection size / 锦标赛选择大小
    mutation_rate: float = 0.25  # Per-gene mutation probability / 逐基因变异概率

    # -- Continuous params (set None to skip) / 连续参数（设 None 跳过）--
    learning_rate: HyperparamRange | None = HyperparamRange(1e-4, 1.0, "log_uniform")
    weight_decay: HyperparamRange | None = HyperparamRange(1e-6, 1e-2, "log_uniform")
    momentum: HyperparamRange | None = HyperparamRange(0.8, 0.99, "uniform")

    # -- Discrete params (set None to skip) / 离散参数（设 None 跳过）--
    batch_size: tuple[int, ...] | None = (128, 256, 512, 1024)
    optimizer_type: tuple[str, ...] | None = (
        "sgd",
        "adam",
        "adamw",
        "rmsprop",
        "nadam",
    )
    scheduler_type: tuple[str, ...] | None = ("cosine", "constant", "step")
