"""
共享 CLI 参数定义与配置覆盖。
Shared CLI argument definitions and config override logic.

设计动机：
- Q1/Q2/Q3 的 parse_args() 有 25+ 个相同的 argparse 参数定义。
- build_config() / _apply_technique_overrides() 中的覆盖逻辑 ×3 处完全相同。
- 统一为两个函数：add_common_train_args() + apply_cli_overrides()。

开闭原则：新增优化技术只需在此文件的列表中添加一项。
"""

import argparse
import dataclasses
from pathlib import Path

from .config import AugmentationConfig


# ---------------------------------------------------------------------------
# CLI argument builder / CLI 参数构建器
# ---------------------------------------------------------------------------


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    """
    添加所有题目共享的训练参数。
    Add 25+ shared training arguments to an ArgumentParser.

    所有 Q 的 parse_args() 都应调用此函数，然后按需添加额外参数。
    """
    # Training overrides / 训练参数覆盖
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num epochs / 覆盖训练轮数",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size / 覆盖批大小",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate / 覆盖学习率",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data root / 覆盖数据根目录",
    )
    parser.add_argument(
        "--dropout", type=float, default=None,
        help=(
            "Dropout rate before FC layer (0 = disabled)"
            " / FC 前的 Dropout 比率（0 = 禁用）"
        ),
    )
    parser.add_argument(
        "--no-bn", action="store_true",
        help=(
            "Disable BatchNorm in conv layers"
            " / 禁用卷积层中的 BatchNorm"
        ),
    )
    # Evaluation / 评估
    parser.add_argument(
        "--eval-only", action="store_true",
        help=(
            "Only run evaluation"
            " (requires existing checkpoint)"
            " / 仅评估（需要已有检查点）"
        ),
    )
    # Search / 超参数搜索
    parser.add_argument(
        "--search", action="store_true",
        help=(
            "Run hyperparameter search,"
            " then train with best params"
            " / 先搜索再用最优配置训练"
        ),
    )
    parser.add_argument(
        "--search-only", action="store_true",
        help=(
            "Only run search, save results, and exit"
            " / 仅运行搜索并报告结果"
        ),
    )
    parser.add_argument(
        "--search-strategy",
        choices=["halving-random", "random", "grid"],
        default=None,
        help=(
            "Search strategy: halving-random (default), random, grid"
            " / 搜索策略"
        ),
    )
    parser.add_argument(
        "--ignore-search", action="store_true",
        help=(
            "Ignore existing search results,"
            " use default/CLI params"
            " / 忽略已有搜索结果"
        ),
    )
    parser.add_argument(
        "--search-results", type=str, default=None,
        help=(
            "Specify search results file"
            " / 指定超参搜索结果文件路径"
        ),
    )
    # Other / 其他
    parser.add_argument(
        "--amp", action="store_true",
        help=(
            "Enable mixed precision (FP16) training"
            " / 启用混合精度训练"
        ),
    )
    parser.add_argument(
        "--no-augmentation", action="store_true",
        help=(
            "Disable data augmentation"
            " / 禁用数据增强"
        ),
    )

    # -- Technique toggles / 优化技术开关 --
    # 新增优化技术只需在此列表添加一项 / Just add to this list for new techniques
    _TECHNIQUE_TOGGLES = [
        ("--no-scheduler", "Disable LR scheduler / 禁用学习率调度"),
        ("--no-weight-decay", "Set weight_decay=0 / 禁用权重衰减"),
        ("--no-label-smoothing", "Set label_smoothing=0 / 禁用标签平滑"),
        ("--no-dropout", "Set dropout_rate=0 / 禁用 Dropout"),
        ("--no-early-stopping", "Disable early stopping / 禁用早停"),
        ("--no-cutmix", "Disable CutMix / 禁用 CutMix"),
        ("--no-mixup", "Disable Mixup / 禁用 Mixup"),
        ("--no-geom-aug", "Disable geometric augmentation / 禁用几何变换增强"),
        ("--no-color-aug", "Disable color augmentation / 禁用颜色变换增强"),
        ("--no-noise-aug", "Disable noise augmentation / 禁用噪声增强"),
        ("--no-weather-aug", "Disable weather augmentation / 禁用天气增强"),
        ("--no-mixing-aug", "Disable batch mixing augmentation / 禁用批次混合增强"),
    ]
    for flag, help_text in _TECHNIQUE_TOGGLES:
        parser.add_argument(flag, action="store_true", help=help_text)


def add_transfer_args(parser: argparse.ArgumentParser) -> None:
    """
    添加 Q3 迁移学习专用参数。
    Add Q3-specific transfer learning arguments.
    """
    parser.add_argument(
        "--transfer", action="store_true",
        help=(
            "Transfer learn from CIFAR-100 to CIFAR-10"
            " / 迁移学习 CIFAR-100 → CIFAR-10"
        ),
    )
    parser.add_argument(
        "--transfer-checkpoint", type=str, default=None,
        help=(
            "Override source checkpoint path for transfer"
            " / 覆盖迁移学习源检查点路径"
        ),
    )
    parser.add_argument(
        "--tv-transfer", action="store_true",
        help=(
            "Transfer learn using PyTorch pretrained ResNet-18"
            " / 使用 PyTorch 预训练模型迁移学习"
        ),
    )


# ---------------------------------------------------------------------------
# Config override builder / 配置覆盖构建器
# ---------------------------------------------------------------------------


def apply_cli_overrides(args: argparse.Namespace) -> dict:
    """
    将 CLI 参数转为 TrainConfig override 字典。
    Convert parsed CLI args to config override dict.

    单一实现替代 Q1/Q2/Q3 main.py 中重复的 build_config() 和
    _apply_technique_overrides()。

    返回值可直接用于 dataclasses.replace(config, **overrides)。

    Args:
        args: parse_args() 返回的 Namespace

    Returns:
        overrides dict，包含所有非默认值的 CLI 覆盖
    """
    overrides: dict = {}

    # 直接映射：arg_name → config_field
    _DIRECT_MAP = {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "learning_rate",
        "dropout": "dropout_rate",
    }
    for arg_name, config_field in _DIRECT_MAP.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            overrides[config_field] = val

    # 布尔开关
    if getattr(args, "no_bn", False):
        overrides["use_bn"] = False
    if getattr(args, "amp", False):
        overrides["use_amp"] = True
    if getattr(args, "no_augmentation", False):
        overrides["augmentation"] = dataclasses.replace(
            AugmentationConfig(), use_augmentation=False,
        )
    if getattr(args, "data_root", None):
        overrides["data_root"] = Path(args.data_root)

    # 技术开关：(arg_attr, config_field, disable_value)
    _TECH_FLAGS = [
        ("no_scheduler", "use_scheduler", False),
        ("no_weight_decay", "weight_decay", 0.0),
        ("no_label_smoothing", "label_smoothing", 0.0),
        ("no_dropout", "dropout_rate", 0.0),
        ("no_early_stopping", "use_early_stopping", False),
    ]
    for arg_attr, config_field, disable_value in _TECH_FLAGS:
        if getattr(args, arg_attr, False):
            overrides[config_field] = disable_value

    # 增强分类覆盖：(arg_attr, aug_field)
    _AUG_FLAGS = [
        ("no_cutmix", "use_cutmix"),
        ("no_mixup", "use_mixup"),
        ("no_geom_aug", "use_geom_aug"),
        ("no_color_aug", "use_color_aug"),
        ("no_noise_aug", "use_noise_aug"),
        ("no_weather_aug", "use_weather_aug"),
        ("no_mixing_aug", "use_mixing_aug"),
    ]
    aug_overrides = {
        field: False
        for arg_attr, field in _AUG_FLAGS
        if getattr(args, arg_attr, False)
    }
    if aug_overrides:
        current = overrides.get("augmentation", AugmentationConfig())
        overrides["augmentation"] = dataclasses.replace(
            current, **aug_overrides,
        )

    return overrides
