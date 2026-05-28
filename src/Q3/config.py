"""
Q3 专用配置：CIFAR-100 训练、迁移学习、超参搜索。
Q3-specific configs: CIFAR-100 training, transfer learning, hyperparameter search.

共享类型（AugmentationConfig, SearchConfig, 归一化常量, 辅助函数）
从 utils.config 导入，Q3 仅保留 Q3 特有的配置类。
Shared types imported from utils.config; Q3 only keeps Q3-specific classes.
"""

from dataclasses import dataclass
from pathlib import Path

# 共享类型导入 / Shared type imports
from utils.config import (
    AugmentationConfig,
    SearchConfig,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    generate_timestamp,
    make_run_dir,
    make_search_dir,
    find_best_search_result,
    dataset_prefix,
)


# ---------------------------------------------------------------------------
# Q3-specific configs / Q3 专用配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """
    CIFAR-100 训练配置 / CIFAR-100 training configuration.
    不可变的冻结 dataclass 防止训练过程中意外修改。

    设计动机：CIFAR-100 超参数与 Q2/CIFAR-10 差异较大
    （更多类别、更大 batch、更多 epochs），因此独立配置。
    预设条件：需要 utils.config.AugmentationConfig 提供增强参数。
    """

    # -- Paths / 路径配置 --
    data_root: Path = Path("data")
    checkpoint_dir: Path = Path("outputs/Q3/checkpoints")

    # -- Model / 模型配置 --
    num_classes: int = 100  # CIFAR-100 has 100 classes / CIFAR-100 有 100 个类别
    # 任务标签，用于区分检查点文件名：
    # "" = CIFAR-100 训练, "transfer" = CIFAR-100→10 迁移,
    # "tvtransfer" = torchvision→10 迁移
    task_tag: str = ""
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
    mean: tuple[float, ...] = CIFAR100_MEAN
    std: tuple[float, ...] = CIFAR100_STD


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
    source_checkpoint: Path = Path(
        "outputs/Q3/checkpoints"
    )  # 目录，运行时自动发现最优模型
    source_num_classes: int = 100  # 源模型类别数 / source model class count

    # -- Target / 目标数据集 --
    num_classes: int = 10  # CIFAR-10
    mean: tuple[float, ...] = CIFAR10_MEAN
    std: tuple[float, ...] = CIFAR10_STD

    # -- Training / 训练超参数 --
    batch_size: int = 256
    epochs: int = 50
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
    checkpoint_dir: Path = Path("outputs/Q3/checkpoints")

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
    checkpoint_dir: Path = Path("outputs/Q3/checkpoints")

    # -- Augmentation / 数据增强 --
    augmentation: AugmentationConfig = AugmentationConfig()


# ---------------------------------------------------------------------------
# Backward compatibility re-exports / 向后兼容重导出
# ---------------------------------------------------------------------------
# 允许 from Q3.config import AugmentationConfig 仍然有效
# Allows from Q3.config import AugmentationConfig to still work

__all__ = [
    # Q3-specific / Q3 专用
    "TrainConfig",
    "TransferConfig",
    "TorchvisionTransferConfig",
    # Shared from utils.config / 从 utils.config 共享
    "AugmentationConfig",
    "SearchConfig",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "generate_timestamp",
    "make_run_dir",
    "make_search_dir",
    "find_best_search_result",
    "dataset_prefix",
]
