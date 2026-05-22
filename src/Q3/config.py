"""
Configuration module for ResNet-18 CIFAR-100 training and hyperparameter search.
ResNet-18 CIFAR-18 CIFAR-100 训练和超参数搜索配置模块。

All hyperparameters live here as a single source.
所有超参数集中在此作为唯一来源。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AugmentationConfig:
    """
    数据增强配置，独立于训练配置管理。
    Data augmentation configuration, independent from training config.

    Design: frozen dataclass with all fields having sensible defaults.
    Each field can be overridden via dataclasses.replace() for search.
    设计：冻结 dataclass，所有字段有合理默认值。
    每个字段可通过 dataclasses.replace() 覆盖，便于超参数搜索。

    19 种增强技术分 5 大类:
      A. Geometric (几何变换): RandomCrop, HFlip, Affine, Perspective
      B. Color (颜色变换): ColorJitter, Grayscale, AutoContrast,
         Equalize, Posterize, Solarize
      C. Noise & Degradation (噪声与降质): GaussianNoise, SaltPepper,
         GaussianBlur, RandomErasing
      D. Weather & Compression (天气与压缩): JPEGCompression, Fog, Rain
      E. Batch Mixing (批次级混合): CutMix, Mixup
    """

    # -- Master switch / 总开关 --
    use_augmentation: bool = True

    # -- A. Geometric / 几何变换 --
    random_crop_padding: int = 4
    hflip_prob: float = 0.5
    affine_degrees: float = 15.0
    affine_translate: float = 0.1
    affine_scale: tuple[float, float] = (0.9, 1.1)
    affine_shear: float = 5.0
    perspective_distortion: float = 0.2
    perspective_prob: float = 0.3

    # -- B. Color / 颜色变换 --
    cj_brightness: float = 0.3
    cj_contrast: float = 0.3
    cj_saturation: float = 0.3
    cj_hue: float = 0.15
    grayscale_prob: float = 0.1
    auto_contrast_prob: float = 0.2
    equalize_prob: float = 0.1
    posterize_bits: int = 4
    posterize_prob: float = 0.1
    solarize_threshold: float = 128.0
    solarize_prob: float = 0.1

    # -- C. Noise & Degradation / 噪声与降质 --
    gaussian_noise_std: float = 0.02
    gaussian_noise_prob: float = 0.5
    salt_pepper_amount: float = 0.01
    salt_pepper_prob: float = 0.2
    gaussian_blur_kernel: int = 3
    gaussian_blur_prob: float = 0.2
    erasing_prob: float = 0.25
    erasing_scale: tuple[float, float] = (0.02, 0.2)

    # -- D. Weather & Compression / 天气与压缩 --
    jpeg_quality: tuple[int, int] = (30, 70)
    jpeg_prob: float = 0.2
    fog_intensity: tuple[float, float] = (0.05, 0.2)
    fog_prob: float = 0.15
    rain_drops: tuple[int, int] = (3, 10)
    rain_angle: tuple[float, float] = (-30.0, 30.0)
    rain_prob: float = 0.15

    # -- E. Batch mixing / 批次级混合 --
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    # P(applying either cutmix or mixup per batch);
    # within that, cutmix 4:3 mixup ratio (matches 40%/30% plan)
    # 每个批次应用 CutMix 或 Mixup 的概率;
    # 其中 CutMix:Mixup = 4:3（对应 40%/30% 方案）
    mix_prob: float = 0.7


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
        0  # Windows spawn overhead > parallel benefit; set 2-4 on Linux
        # Windows 下 spawn 开销大于并行收益；Linux 可设 2-4
    )
    pin_memory: bool = True  # Faster CPU→GPU transfer / 加速 CPU→GPU 数据传输

    # -- Data augmentation / 数据增强 --
    augmentation: AugmentationConfig = AugmentationConfig()

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
    Configuration for hyperparameter search.
    超参数搜索配置。

    Supports three strategies: evolutionary, random, grid.
    支持三种策略：演化搜索、随机搜索、网格搜索。

    All search-related settings live here. Set any parameter range to None to
    exclude it from the search. search.py reads from this config exclusively.
    所有搜索相关设置集中在此。将任意参数范围设为 None 即可跳过该参数的搜索。
    search.py 仅从此配置读取搜索空间。
    """

    # -- Strategy / 搜索策略 --
    strategy: str = "random"  # "evolutionary", "random", "grid"

    # -- Shared / 共享参数 --
    search_epochs: int = 5  # Epochs per individual evaluation / 每个个体的训练轮数

    # -- Evolutionary algorithm / 演化算法参数 --
    population_size: int = 8  # μ: number of parents / 种群大小
    offspring_per_gen: int = 4  # λ: offspring per generation / 每代后代数
    num_generations: int = 3  # G: number of generations / 演化代数
    tournament_size: int = 3  # Tournament selection size / 锦标赛选择大小
    mutation_rate: float = 0.25  # Per-gene mutation probability / 逐基因变异概率

    # -- Random search / 随机搜索参数 --
    num_trials: int = 10  # Number of random evaluations / 随机评估次数

    # -- Grid search / 网格搜索参数 --
    grid_num_points: int = 5  # Points per continuous dimension / 每个连续维度的采样点数

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
