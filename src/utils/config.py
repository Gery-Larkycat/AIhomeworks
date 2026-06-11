"""
Shared configuration components for all homework assignments.
所有作业共享的配置组件。

Contains augmentation settings, search configuration, dataset normalization
constants, and filesystem helper functions used across Q1/Q2/Q3.

包含增强设置、搜索配置、数据集归一化常量，
以及 Q1/Q2/Q3 共用的文件系统辅助函数。
"""

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset normalization constants / 数据集归一化常量
# ---------------------------------------------------------------------------

CIFAR10_MEAN: tuple[float, ...] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, ...] = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN: tuple[float, ...] = (0.5071, 0.4867, 0.4408)
CIFAR100_STD: tuple[float, ...] = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# AugmentationConfig / 增强配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationConfig:
    """
    19 种增强技术，5 大类。
    19 augmentation techniques across 5 categories.

    Design: frozen dataclass with all fields having sensible defaults.
    Each field can be overridden via dataclasses.replace() for search.

    Categories / 大类:
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

    # -- 分类主开关 / Category master switches --
    # 层级：use_augmentation（全局）> use_xxx_aug（分类）> 各单技术参数
    # Hierarchy: use_augmentation (global) > use_xxx_aug (category) > per-technique params
    use_geom_aug: bool = True       # A. Geometric / 几何变换
    use_color_aug: bool = True      # B. Color / 颜色变换
    use_noise_aug: bool = True      # C. Noise & Degradation / 噪声与降质
    use_weather_aug: bool = True    # D. Weather & Compression / 天气与压缩
    use_mixing_aug: bool = True     # E. Batch Mixing / 批次级混合

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
    # within that, cutmix 4:3 mixup ratio
    # 每个批次应用 CutMix 或 Mixup 的概率;
    # 其中 CutMix:Mixup = 4:3
    mix_prob: float = 0.7


# ---------------------------------------------------------------------------
# SearchConfig / 搜索配置
# ---------------------------------------------------------------------------


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
    strategy: str = "halving-random"

    # -- Successive halving / 逐步减半参数 --
    search_epochs_min: int = 2
    search_epochs_max: int = 20
    halving_factor: int = 3

    # -- Candidate sampling / 候选采样 --
    num_trials: int = 50

    # -- Cross-validation / 交叉验证 --
    cv: int = 3

    # -- Discrete params / 离散参数 --
    batch_size_choices: tuple[int, ...] = (128, 256, 512)

    # -- Scoring / 评分指标 --
    scoring: str = "accuracy"


# ---------------------------------------------------------------------------
# TrainConfig / 统一训练配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """
    统一训练配置，覆盖所有题目（Q1/Q2/Q3）。
    Unified training configuration for all tasks.

    设计动机：
    - Q1 (VGG-16/CIFAR-10)、Q2 (ResNet-18/CIFAR-10)、Q3 (ResNet-18/CIFAR-100)
      的训练配置字段完全相同，仅默认值有差异。
    - 通过 models.registry.TaskSpec.default_overrides 覆盖差异项，
      避免三份近乎相同的 frozen dataclass。

    创建方式：
    - 直接构造 TrainConfig() 使用通用默认值
    - 使用 models.registry.make_config("Q1") 获取 Q1 专属默认配置
    - 使用 dataclasses.replace(config, epochs=50) 修改任意字段

    预设条件：需要 AugmentationConfig 提供增强参数。
    """
    # -- Paths / 路径 --
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("outputs/checkpoints")

    # -- Model / 模型 --
    num_classes: int = 10
    dropout_rate: float = 0.5
    use_bn: bool = True
    model_name: str = "resnet18"
    task_tag: str = ""

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
    use_scheduler: bool = True       # CosineAnnealingLR on/off / 余弦退火开关
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

    # -- Normalization / 归一化 --
    mean: tuple[float, ...] = CIFAR10_MEAN
    std: tuple[float, ...] = CIFAR10_STD


# ---------------------------------------------------------------------------
# Run directory helpers / 运行目录辅助函数
# ---------------------------------------------------------------------------


def generate_timestamp() -> str:
    """
    生成时间戳字符串，用于训练运行目录命名。
    Format: YYYY-MM-DD_HHMMSS (sortable, readable, Windows-safe).
    """
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def make_run_dir(
    question: str = "Q3",
    timestamp: str | None = None,
) -> Path:
    """
    构造带时间戳的运行目录路径。
    Construct: outputs/<question>/checkpoints/<timestamp>.
    Directory is NOT created here — save functions create it on demand.
    目录不在此创建——由各 save 函数按需创建。
    """
    if timestamp is None:
        timestamp = generate_timestamp()
    return Path("outputs") / question / "checkpoints" / timestamp


def make_search_dir(question: str = "Q3") -> Path:
    """
    构造超参搜索结果目录路径。
    Construct search results directory: outputs/<question>/search_results/.
    Directory is NOT created here — save functions create it on demand.
    目录不在此创建——由各 save 函数按需创建。
    """
    return Path("outputs") / question / "search_results"


def find_best_search_result(
    search_dir: Path,
    pattern: str = "*_hp_search.json",
) -> Path | None:
    """
    扫描搜索结果目录，找到 mean_test_score 最高的文件。
    Scan search_dir for files matching pattern, pick highest mean_test_score.

    同分数时按文件名倒序（ISO 时间戳前缀 → 最新优先）。
    文件格式需含 best.mean_test_score 字段。
    文件损坏或格式不符时静默跳过。
    """
    if not search_dir.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    # 按名称倒序遍历，同分时最新的排在前面
    paths = sorted(
        search_dir.glob(pattern),
        key=lambda p: p.name, reverse=True,
    )
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            score = float(data["best"]["mean_test_score"])
            candidates.append((score, path))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    if not candidates:
        return None

    # 按分数降序，同分时倒序遍历已保证最新在前
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def dataset_prefix(
    num_classes: int, task: str = "", model_name: str = "resnet18",
) -> str:
    """
    根据 num_classes 和任务类型返回检查点文件名前缀。
    Return checkpoint filename prefix based on num_classes and task.

    100 → resnet18_cifar100, 10 → resnet18_cifar10, 其他 → resnet18_Ncls。
    task 非空时追加下划线后缀：resnet18_cifar10_transfer。
    model_name 替换前缀中的 resnet18：vgg16_cifar10。
    """
    if num_classes == 100:
        base = f"{model_name}_cifar100"
    elif num_classes == 10:
        base = f"{model_name}_cifar10"
    else:
        base = f"{model_name}_{num_classes}cls"
    return f"{base}_{task}" if task else base


def config_to_dict(config) -> dict:
    """
    将 frozen dataclass 配置递归转为 JSON-safe 字典。
    Convert a frozen dataclass config to a JSON-safe dict.

    处理：Path → str, tuple → list, 嵌套 dataclass 递归展开。
    """
    raw = asdict(config)
    return _make_json_safe(raw)


def _make_json_safe(obj):
    """递归转换不可 JSON 序列化的类型。"""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    return obj
