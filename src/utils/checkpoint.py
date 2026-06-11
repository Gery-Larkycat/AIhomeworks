"""
通用检查点保存/加载模块。
Generic checkpoint saving/loading for model training and transfer learning.

保存三种产物：
1. 完整模型检查点（用于恢复训练）
2. 特征提取器权重（用于迁移学习）
3. 训练历史 JSON

设计动机：从 Q3/checkpoint.py 泛化而来，移除对特定模型类
（Q2.model.ResNet18）的硬编码依赖，通过参数化支持任意模型。
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import dataset_prefix


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist / 目录不存在则创建。"""
    path.mkdir(parents=True, exist_ok=True)


def save_full_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    accuracy: float,
    checkpoint_dir: Path,
    filename: str = "model_full.pth",
    num_classes: int = 10,
    config_dict: dict | None = None,
) -> Path:
    """
    Save full training checkpoint for resuming.
    保存完整训练检查点，用于恢复训练。

    Args:
        model:          PyTorch 模型
        optimizer:      优化器
        epoch:          当前 epoch
        accuracy:       当前准确率
        checkpoint_dir: 检查点保存目录
        filename:       文件名
        num_classes:    分类数（存入检查点供加载时验证）
        config_dict:    可选的训练配置字典（保存为独立 JSON）

    Returns:
        检查点文件路径
    """
    _ensure_dir(checkpoint_dir)
    path = checkpoint_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
            "num_classes": num_classes,
        },
        path,
    )
    # 保存训练配置为独立 JSON / Save config as separate JSON
    if config_dict is not None:
        prefix = filename.replace("_best.pth", "").replace("_full.pth", "")
        config_path = checkpoint_dir / f"{prefix}_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    return path


def save_best_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    accuracy: float,
    checkpoint_dir: Path,
    num_classes: int = 10,
    task_tag: str = "",
    model_name: str = "resnet18",
    config_dict: dict | None = None,
) -> Path:
    """
    Save best model checkpoint / 保存最佳模型检查点。

    文件名自动通过 dataset_prefix() 生成。
    """
    prefix = dataset_prefix(num_classes, task_tag, model_name)
    filename = f"{prefix}_best.pth"
    return save_full_checkpoint(
        model, optimizer, epoch, accuracy, checkpoint_dir, filename,
        num_classes=num_classes, config_dict=config_dict,
    )


def save_feature_extractor(
    model: nn.Module,
    checkpoint_dir: Path,
    exclude_prefix: str = "fc.",
    filename: str | None = None,
    num_classes: int = 10,
    task_tag: str = "",
    model_name: str = "resnet18",
) -> Path:
    """
    Save feature extractor (all layers except excluded) for transfer learning.
    保存特征提取器（排除指定前缀的层），供迁移学习使用。

    Args:
        model:           PyTorch 模型
        checkpoint_dir:  保存目录
        exclude_prefix:  排除的键前缀（默认 "fc."，VGG 用 "fc1.,fc2."）
        filename:        自定义文件名；None 时自动生成
        num_classes:     分类数（用于自动文件名）
        task_tag:        任务标签（用于自动文件名）
        model_name:      模型名（用于自动文件名）
    """
    if filename is None:
        prefix = dataset_prefix(num_classes, task_tag, model_name)
        filename = f"{prefix}_feature_extractor.pth"
    _ensure_dir(checkpoint_dir)
    path = checkpoint_dir / filename
    # 按 exclude_prefix 过滤，支持逗号分隔的多个前缀
    prefixes = tuple(p.strip() for p in exclude_prefix.split(","))
    feature_state = OrderedDict(
        (k, v) for k, v in model.state_dict().items()
        if not any(k.startswith(p) for p in prefixes)
    )
    torch.save(feature_state, path)
    return path


def save_training_history(
    history: dict[str, list[float]],
    checkpoint_dir: Path,
    filename: str = "training_history.json",
) -> Path:
    """
    Save training history as JSON / 将训练历史保存为 JSON。
    """
    _ensure_dir(checkpoint_dir)
    path = checkpoint_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return path


def load_full_checkpoint(
    path: Path, model: nn.Module, optimizer=None,
) -> dict[str, Any]:
    """
    Load full checkpoint, restoring model and optionally optimizer state.
    加载完整检查点，恢复模型和可选的优化器状态。

    Args:
        path:      检查点文件路径
        model:     PyTorch 模型
        optimizer: 可选的优化器（传入则恢复其状态）

    Returns:
        检查点字典（含 epoch, accuracy, num_classes 等）
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_feature_extractor(path: Path) -> OrderedDict[str, Any]:
    """
    Load feature extractor state dict for transfer learning.
    加载特征提取器 state dict，用于迁移学习。
    """
    return torch.load(path, weights_only=True)
