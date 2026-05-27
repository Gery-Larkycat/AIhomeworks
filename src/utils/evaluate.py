"""
Evaluation metrics: loss/accuracy, per-class accuracy and confusion matrix.
评估指标：损失/准确率、每类准确率和混淆矩阵。

Top-1 accuracy and loss evaluation is handled by skorch's built-in scoring
for the skorch pipeline. evaluate() is kept for legacy training loops
(train.py) that still use manual training.
skorch 管线的 top-1 准确率和损失评估由内置评分处理。
evaluate() 保留给仍使用手写训练循环（train.py）的旧代码。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float]:
    """
    评估模型，返回 (损失, 准确率)。
    Evaluate model, returning (loss, accuracy).

    以 GPU 张量累积，避免每 batch 同步。
    Accumulates as GPU tensors to avoid per-batch sync.

    Args:
        model:  PyTorch 模型
        loader: DataLoader
        device: 计算设备
        use_amp: 是否启用混合精度 / whether to use mixed precision

    Returns:
        (avg_loss, accuracy)
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
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
