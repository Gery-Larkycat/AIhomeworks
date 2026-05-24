"""
Configuration module for ResNet-18 CIFAR-100 training and hyperparameter search.
ResNet-18 CIFAR-18 CIFAR-100 训练和超参数搜索配置模块。

All hyperparameters live here as a single source.
所有超参数集中在此作为唯一来源。
"""

from dataclasses import dataclass
from datetime import datetime
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
    dropout_rate: float = (
        0.5  # Dropout after global avg pool; 0 = disabled / 全局池化后 Dropout；0 = 禁用
    )

    # -- Training / 训练超参数 --
    batch_size: int = 1024
    epochs: int = 150
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1

    # -- Optimizer & Scheduler / 优化器与调度器 --
    optimizer_type: str = "sgd"  # "sgd", "adam", "adamw", "rmsprop", "nadam"
    scheduler_type: str = "cosine"  # "cosine", "constant", "step"
    use_amp: bool = False  # Mixed precision (FP16) training / 混合精度训练

    # -- Early stopping / 早停 --
    patience: int = 8  # Epochs to wait without improvement / 无改善等待轮数
    min_delta: float = (
        1e-4  # Minimum accuracy improvement to qualify / 视为改善的最小准确率增量
    )

    # -- Scheduler / 学习率调度 --
    scheduler_t_max: int = 150  # Cosine annealing period / 余弦退火周期

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
class SearchConfig:
    """
    超参数搜索配置，用于 skorch + sklearn 搜索管线。
    Search config for skorch + sklearn hyperparameter search pipeline.

    搜索空间（lr, momentum, weight_decay）以 scipy.stats 分布形式
    硬编码在 search.py 中；此处仅控制搜索策略和行为参数。

    Supports: halving-random (default), random, grid.
    支持：halving-random（默认）、random、grid。
    """

    # -- Strategy / 搜索策略 --
    strategy: str = "halving-random"  # "halving-random", "random", "grid"

    # -- Successive halving / 逐步减半参数 --
    search_epochs_min: int = 2  # 初始最少训练轮数 / minimum epochs at start
    search_epochs_max: int = 20  # 最终最多训练轮数 / maximum epochs at end
    halving_factor: int = 3  # 每轮保留 1/factor / keep top 1/factor per round

    # -- Candidate sampling / 候选采样 --
    num_trials: int = 50  # 随机采样候选数 / number of random candidates

    # -- Cross-validation / 交叉验证 --
    cv: int = 3  # CV 折数 / number of CV folds

    # -- Discrete params / 离散参数 --
    batch_size_choices: tuple[int, ...] = (128, 256, 512)

    # -- Scoring / 评分指标 --
    scoring: str = "accuracy"


# ---------------------------------------------------------------------------
# CIFAR-10 constants / CIFAR-10 常量
# ---------------------------------------------------------------------------

CIFAR10_MEAN: tuple[float, ...] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, ...] = (0.2470, 0.2435, 0.2616)

# ---------------------------------------------------------------------------
# ImageNet constants (torchvision pretrained model normalization) / ImageNet 常量
# ---------------------------------------------------------------------------

IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransferConfig:
    """
    迁移学习配置：CIFAR-100 预训练 → CIFAR-10 微调。
    Transfer learning config: CIFAR-100 pretrained → CIFAR-10 fine-tune.

    加载预训练权重 → 冻结 backbone → 替换 FC → 仅训练 FC。
    同时包含训练配置和搜索配置（搜索空间/默认值与 CIFAR-100 不同）。

    设计思路：与 TrainConfig 平行但独立，因为迁移学习的超参数
    （lr、epochs、batch_size 等）与全量训练差异较大，不宜共用默认值。
    """

    # -- Source / 源模型 --
    source_checkpoint: Path = Path("checkpoints/resnet18_cifar100_best.pth")
    source_num_classes: int = 100  # 源模型类别数 / source model class count

    # -- Target / 目标数据集 --
    num_classes: int = 10  # CIFAR-10
    mean: tuple[float, ...] = CIFAR10_MEAN
    std: tuple[float, ...] = CIFAR10_STD

    # -- Training / 训练超参数 --
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 0.01  # FC-only 训练用较低学习率
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    optimizer_type: str = "sgd"
    scheduler_type: str = "cosine"
    use_amp: bool = False
    patience: int = 8
    min_delta: float = 1e-4
    scheduler_t_max: int = 30

    # -- Data loading / 数据加载 --
    num_workers: int = 0
    pin_memory: bool = True
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")

    # -- Augmentation / 数据增强 --
    augmentation: AugmentationConfig = AugmentationConfig()

    # -- Transfer search / 迁移超参搜索 --
    search_epochs_min: int = 2
    search_epochs_max: int = 10  # FC-only 收敛快，不需要太长
    halving_factor: int = 3
    num_trials: int = 30
    cv: int = 3
    batch_size_choices: tuple[int, ...] = (64, 128, 256)


@dataclass(frozen=True)
class TorchvisionTransferConfig:
    """
    PyTorch 预训练 ResNet-18 → CIFAR-10 迁移学习配置。
    Torchvision pretrained ResNet-18 → CIFAR-10 transfer learning config.

    加载 torchvision 官方 ImageNet 预训练权重 → 替换 FC 为 10 类
    → 冻结 backbone → 仅训练 FC 层。

    与 TransferConfig 的区别：使用 ImageNet 归一化统计量、224x224 输入尺寸
    （torchvision ResNet-18 的标准输入），无需 source_checkpoint（权重来自 torchvision）。
    """

    # -- Image / 图像尺寸 --
    image_size: int = 224  # torchvision ResNet-18 标准输入尺寸

    # -- Target / 目标数据集 --
    num_classes: int = 10  # CIFAR-10
    mean: tuple[float, ...] = IMAGENET_MEAN
    std: tuple[float, ...] = IMAGENET_STD

    # -- Training / 训练参数 --
    batch_size: int = 64  # 224x224 显存占用高，需要较小 batch
    epochs: int = 30
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4  # 比全量训练小，防止微调时过度约束预训练特征
    label_smoothing: float = 0.0  # 迁移学习样本少，避免过度正则化
    optimizer_type: str = "sgd"
    scheduler_type: str = "cosine"
    scheduler_t_max: int = 30
    use_amp: bool = False
    patience: int = 8
    min_delta: float = 1e-4

    # -- Data loading / 数据加载 --
    num_workers: int = 0
    pin_memory: bool = True
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")

    # -- Augmentation / 数据增强 --
    augmentation: AugmentationConfig = AugmentationConfig()


# ---------------------------------------------------------------------------
# Run directory helpers / 运行目录辅助函数
# ---------------------------------------------------------------------------


def generate_timestamp() -> str:
    """
    生成时间戳字符串，用于训练运行目录命名。
    Generate timestamp string for run directory naming.

    格式 YYYY-MM-DD_HHMMSS：可排序、可读、Windows 安全（无冒号）。
    """
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def make_run_dir(
    base: Path = Path("checkpoints"),
    timestamp: str | None = None,
) -> Path:
    """
    构造带时间戳的运行目录路径。
    Construct timestamped run directory path.

    Args:
        base: 基础目录，默认 checkpoints
        timestamp: 时间戳字符串，None 时自动生成

    Returns:
        完整的运行目录路径（目录尚未创建，由各 save 函数按需创建）
    """
    if timestamp is None:
        timestamp = generate_timestamp()
    return base / timestamp


def dataset_prefix(num_classes: int) -> str:
    """
    根据 num_classes 返回检查点文件名前缀。
    Return checkpoint filename prefix based on num_classes.

    100 → resnet18_cifar100, 10 → resnet18_cifar10, 其他 → resnet18_Ncls。
    """
    if num_classes == 100:
        return "resnet18_cifar100"
    if num_classes == 10:
        return "resnet18_cifar10"
    return f"resnet18_{num_classes}cls"
