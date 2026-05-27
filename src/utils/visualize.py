"""
Visualization utilities for training metrics and model analysis.
训练指标和模型分析的可视化工具。

Generates: training curves (loss & accuracy), confusion matrix heatmap,
learning rate schedule.
生成：训练曲线（损失和准确率）、混淆矩阵热力图、学习率调度。
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Use CJK-compatible font on Windows to render Chinese labels
mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

import torch


def plot_training_curves(
    history: dict[str, list[float]],
    save_dir: Path,
) -> Path:
    """
    Plot training/test loss and accuracy curves.
    绘制训练/测试损失和准确率曲线。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "training_curves.png"

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(epochs, history["test_loss"], label="Test Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Test Loss / 训练与测试损失")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="blue")
    ax2.plot(epochs, history["test_acc"], label="Test Acc", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Test Accuracy / 训练与测试准确率")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_confusion_matrix(
    cm: torch.Tensor,
    save_dir: Path,
    title: str = "Confusion Matrix / 混淆矩阵",
    max_labels: int = 100,
) -> Path:
    """
    Plot confusion matrix as a heatmap.
    绘制混淆矩阵热力图。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "confusion_matrix.png"

    cm_np = cm.numpy()
    if max_labels > 0 and cm_np.shape[0] > max_labels:
        cm_np = cm_np[:max_labels, :max_labels]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_np, cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Predicted / 预测")
    ax.set_ylabel("True / 真实")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_lr_schedule(
    history: dict[str, list[float]],
    save_dir: Path,
) -> Path:
    """
    Plot learning rate schedule / 绘制学习率变化曲线。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "lr_schedule.png"

    epochs = range(1, len(history["lr"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["lr"], color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule / 学习率调度")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
