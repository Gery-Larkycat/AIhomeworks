"""
Checkpoint saving/loading for ResNet-18 training and transfer learning.
ResNet-18 训练检查点保存/加载，以及迁移学习特征提取器导出。

Saves three artifacts:
1. Full model checkpoint (for resuming training)
2. Feature extractor weights (for transfer learning to CIFAR-10)
3. Training history JSON

保存三种产物：
1. 完整模型检查点（用于恢复训练）
2. 特征提取器权重（用于迁移到 CIFAR-10）
3. 训练历史 JSON
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD

from .config import TrainConfig
from .model import ResNet18, get_feature_extractor_state


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist / 目录不存在则创建。"""
    path.mkdir(parents=True, exist_ok=True)


def save_full_checkpoint(
    model: nn.Module,
    optimizer: SGD,
    epoch: int,
    accuracy: float,
    config: TrainConfig,
    filename: str = "resnet18_cifar100_full.pth",
) -> Path:
    """
    Save full training checkpoint for resuming.
    保存完整训练检查点，用于恢复训练。
    """
    _ensure_dir(config.checkpoint_dir)
    path = config.checkpoint_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
            "num_classes": config.num_classes,
        },
        path,
    )
    return path


def save_best_checkpoint(
    model: nn.Module,
    optimizer: SGD,
    epoch: int,
    accuracy: float,
    config: TrainConfig,
) -> Path:
    """
    Save best model checkpoint / 保存最佳模型检查点。
    """
    return save_full_checkpoint(model, optimizer, epoch, accuracy, config, "resnet18_cifar100_best.pth")


def save_feature_extractor(
    model: ResNet18,
    config: TrainConfig,
    filename: str = "resnet18_cifar100_feature_extractor.pth",
) -> Path:
    """
    Save feature extractor (all layers except FC) for transfer learning.
    保存特征提取器（除 FC 外的所有层），供迁移学习使用。

    Usage in transfer learning / 迁移学习用法:
        state = torch.load("resnet18_cifar100_feature_extractor.pth")
        model = ResNet18(num_classes=10)  # CIFAR-10 has 10 classes
        model.load_state_dict(state, strict=False)  # FC layer is randomly initialized
    """
    _ensure_dir(config.checkpoint_dir)
    path = config.checkpoint_dir / filename
    feature_state = get_feature_extractor_state(model)
    torch.save(feature_state, path)
    return path


def save_training_history(
    history: dict[str, list[float]],
    config: TrainConfig,
    filename: str = "training_history.json",
) -> Path:
    """
    Save training history as JSON / 将训练历史保存为 JSON。
    """
    _ensure_dir(config.checkpoint_dir)
    path = config.checkpoint_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return path


def load_full_checkpoint(
    path: Path, model: nn.Module, optimizer: SGD | None = None
) -> dict[str, Any]:
    """
    Load full checkpoint, restoring model and optionally optimizer state.
    加载完整检查点，恢复模型和可选的优化器状态。
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
