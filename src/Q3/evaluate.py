"""
Evaluation metrics for ResNet-18 on CIFAR-100.
ResNet-18 在 CIFAR-100 上的评估指标。

Provides top-1 accuracy, per-class accuracy, and confusion matrix.
提供 top-1 准确率、每类准确率和混淆矩阵。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """
    Evaluate model, returning (loss, accuracy).
    评估模型，返回 (损失, 准确率)。

    Accumulates as GPU tensors to avoid per-batch sync.
    以 GPU 张量累积，避免每 batch 同步。
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum()
        total_samples += images.size(0)

    avg_loss = (total_loss / total_samples).item()
    accuracy = (total_correct / total_samples).item()
    return avg_loss, accuracy


@torch.no_grad()
def per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[int, float]:
    """
    Compute per-class accuracy / 计算每个类别的准确率。
    """
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for cls in range(num_classes):
            mask = labels == cls
            total[cls] += mask.sum().item()
            correct[cls] += (preds[mask] == cls).sum().item()

    return {cls: (correct[cls] / total[cls]).item() if total[cls] > 0 else 0.0
            for cls in range(num_classes)}


@torch.no_grad()
def confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute confusion matrix (rows=true, cols=predicted).
    计算混淆矩阵（行=真实标签，列=预测标签）。
    """
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for t, p in zip(labels, preds):
            cm[t.item(), p.item()] += 1

    return cm
