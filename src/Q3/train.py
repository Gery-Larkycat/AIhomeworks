"""
Training loop for ResNet-18 on CIFAR-100.
ResNet-18 在 CIFAR-100 上的训练循环。

Uses SGD + Cosine Annealing LR + Label Smoothing CrossEntropy.
使用 SGD + 余弦退火学习率 + 标签平滑交叉熵。
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import TrainConfig
from .evaluate import evaluate


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: SGD,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Train for one epoch, returning (loss, accuracy).
    训练一个 epoch，返回 (损失, 准确率)。
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

        # Progress every 100 batches / 每 100 个 batch 打印进度
        if (batch_idx + 1) % 100 == 0:
            batch_acc = total_correct / total_samples
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} "
                f"| Loss: {total_loss / total_samples:.4f} | Acc: {batch_acc:.4f}"
            )

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainConfig,
) -> dict[str, list[float]]:
    """
    Full training loop with scheduler, returning training history.
    包含调度器的完整训练循环，返回训练历史。

    History contains per-epoch: train_loss, train_acc, test_loss, test_acc, lr.
    历史包含每轮：train_loss, train_acc, test_loss, test_acc, lr。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and scheduler / 优化器和学习率调度器
    optimizer = SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.scheduler_t_max)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }
    best_acc = 0.0

    # Import here to avoid circular dependency / 延迟导入避免循环依赖
    from .checkpoint import save_best_checkpoint, save_feature_extractor

    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config.epochs} | LR: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        # Record history / 记录训练历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        print(
            f"  Summary | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        # Save best model / 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_best_checkpoint(model, optimizer, epoch, test_acc, config)
            save_feature_extractor(model, config)
            print(f"  ** New best accuracy: {best_acc:.4f} **")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")
    return history
