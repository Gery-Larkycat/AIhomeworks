"""
Q3 检查点管理（向后兼容重导出）。
Backward-compatible re-exports for Q3 checkpoint management.

实际实现已移至 utils.checkpoint。此处保留 Q3 旧接口的便捷封装。
"""

from utils.checkpoint import (  # noqa: F401
    save_full_checkpoint,
    save_training_history,
    load_full_checkpoint,
    load_feature_extractor,
)

from utils.config import dataset_prefix
from pathlib import Path
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


def save_best_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    accuracy: float,
    config,
    config_dict: dict | None = None,
) -> Path:
    """
    Q3 兼容的 save_best_checkpoint 封装。
    接受 config 对象（TrainConfig duck type），自动提取参数。
    """
    from utils.checkpoint import save_best_checkpoint as _save
    return _save(
        model, optimizer, epoch, accuracy,
        checkpoint_dir=config.checkpoint_dir,
        num_classes=config.num_classes,
        task_tag=getattr(config, "task_tag", ""),
        model_name=getattr(config, "model_name", "resnet18"),
        config_dict=config_dict,
    )


def save_feature_extractor(
    model: nn.Module,
    config,
    filename: str | None = None,
) -> Path:
    """
    Q3 兼容的 save_feature_extractor 封装。
    默认使用 "fc." 前缀（ResNet-18）。
    """
    from utils.checkpoint import save_feature_extractor as _save
    return _save(
        model,
        checkpoint_dir=config.checkpoint_dir,
        exclude_prefix="fc.",
        filename=filename,
        num_classes=config.num_classes,
        task_tag=getattr(config, "task_tag", ""),
        model_name=getattr(config, "model_name", "resnet18"),
    )
