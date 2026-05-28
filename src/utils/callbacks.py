"""
Custom skorch callbacks for training checkpoint and feature extractor saving.
自定义 skorch 回调：训练检查点保存和特征提取器导出。

skorch 内置 Checkpoint 不包含 accuracy/epoch/num_classes 元数据，
迁移学习需要这些字段，因此用自定义回调替代。
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from skorch.callbacks import Callback

from .config import dataset_prefix


class CustomCheckpoint(Callback):
    """
    每当监控指标达到新最优时，保存自定义格式检查点。
    Save custom-format checkpoint when monitored metric reaches new best.

    保存格式: {epoch, model_state_dict, optimizer_state_dict, accuracy, num_classes}
    与 Q3 迁移学习的 load_full_checkpoint() 兼容。
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        num_classes: int,
        monitor: str = "valid_acc_best",
        task_tag: str = "",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_classes = num_classes
        self.monitor = monitor
        self.task_tag = task_tag

    def on_epoch_end(self, net, **kwargs):
        # 仅在新最优时保存 / Only save on new best
        if not net.history[-1].get(self.monitor, False):
            return

        acc = net.history[-1]["valid_acc"]
        epoch = len(net.history)
        prefix = dataset_prefix(self.num_classes, self.task_tag)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{prefix}_best.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.module_.state_dict(),
                "optimizer_state_dict": net.optimizer_.state_dict(),
                "accuracy": acc,
                "num_classes": self.num_classes,
            },
            path,
        )
        print(f"  ** Saved best checkpoint: {acc:.4f} -> {path.name} **")


class FeatureExtractorCheckpoint(Callback):
    """
    每当验证准确率达到新最优时，额外保存特征提取器权重（去掉 FC 层）。
    Save feature extractor weights (minus FC) on new best validation accuracy.

    用于迁移学习：CIFAR-100 训练 → CIFAR-10 微调。
    默认去掉所有 `fc.` 前缀的键（适用于 ResNet-18）。
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        num_classes: int,
        monitor: str = "valid_acc_best",
        exclude_prefix: str = "fc.",
        task_tag: str = "",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_classes = num_classes
        self.monitor = monitor
        self.exclude_prefix = exclude_prefix
        self.task_tag = task_tag

    def on_epoch_end(self, net, **kwargs):
        if not net.history[-1].get(self.monitor, False):
            return

        prefix = dataset_prefix(self.num_classes, self.task_tag)
        path = self.checkpoint_dir / f"{prefix}_feature_extractor.pth"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = OrderedDict(
            (k, v)
            for k, v in net.module_.state_dict().items()
            if not k.startswith(self.exclude_prefix)
        )
        torch.save(state, path)


class LRRecorder(Callback):
    """
    每个 epoch 结束时记录当前学习率到 history。
    Record current learning rate to history at end of each epoch.

    skorch 的 history 额外增加 'lr' 字段，便于可视化学习率调度。
    """

    def on_epoch_end(self, net, **kwargs):
        lr = net.optimizer_.param_groups[0]["lr"]
        net.history.record("lr", lr)


class TrainingHistory(Callback):
    """
    训练结束后将 skorch history 导出为标准 dict 并保存 JSON。
    Export skorch history to standard dict and save as JSON after training.

    输出格式与 Q3 原有 training_history.json 兼容：
    {train_loss, train_acc, test_loss, test_acc, lr, dur}
    """

    def __init__(self, checkpoint_dir: Path | str):
        self.checkpoint_dir = Path(checkpoint_dir)

    def on_train_end(self, net, **kwargs):
        import json

        history = extract_history(net)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"  Training history saved to {path}")


def extract_history(net) -> dict[str, list[float]]:
    """
    将 skorch 的 history_ 转换为标准 dict 格式。
    Convert skorch history_ to standard dict format.

    skorch 字段映射:
      train_loss → train_loss
      valid_loss → test_loss
      train_acc  → train_acc
      valid_acc  → test_acc
      dur        → dur
      lr         → lr（需 LRRecorder 回调）
    """
    h = net.history
    result: dict[str, list[float]] = {}

    field_map = {
        "train_loss": "train_loss",
        "valid_loss": "test_loss",
        "train_acc": "train_acc",
        "valid_acc": "test_acc",
        "dur": "dur",
        "lr": "lr",
    }

    for skorch_key, our_key in field_map.items():
        try:
            result[our_key] = list(h[:, skorch_key])
        except KeyError:
            # Field not present in history (e.g., lr without LRRecorder)
            result[our_key] = []

    return result
