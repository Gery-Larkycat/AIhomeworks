"""
Training loop for ResNet-18 on CIFAR-100.
ResNet-18 在 CIFAR-100 上的训练循环。

Uses configurable optimizer, scheduler, and loss via factory functions.
通过工厂函数使用可配置的优化器、调度器和损失函数。
"""

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, NAdam, RMSprop, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LRScheduler
from torch.utils.data import DataLoader

from .config import TrainConfig
from .evaluate import evaluate


def create_optimizer(
    model: nn.Module, config: TrainConfig
) -> Optimizer:
    """
    Create optimizer from config / 根据配置创建优化器。

    Supports: sgd, adam, adamw, rmsprop, nadam.
    支持：sgd, adam, adamw, rmsprop, nadam。
    """
    params = model.parameters()
    opt = config.optimizer_type.lower()
    if opt == "sgd":
        return SGD(
            params, lr=config.learning_rate, momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif opt == "adam":
        return Adam(
            params, lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif opt == "adamw":
        return AdamW(
            params, lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif opt == "rmsprop":
        return RMSprop(
            params, lr=config.learning_rate, momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif opt == "nadam":
        return NAdam(
            params, lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {config.optimizer_type}"
        )


def create_scheduler(
    optimizer: Optimizer, config: TrainConfig
) -> LRScheduler | None:
    """
    Create scheduler from config / 根据配置创建学习率调度器。

    Returns None for "constant" (no scheduling).
    "constant" 返回 None（不使用调度器）。
    """
    sch = config.scheduler_type.lower()
    if sch == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.scheduler_t_max)
    elif sch == "step":
        step_size = max(1, config.epochs // 3)
        return StepLR(optimizer, step_size=step_size, gamma=0.5)
    elif sch == "constant":
        return None
    else:
        raise ValueError(
            f"Unknown scheduler: {config.scheduler_type}"
        )


def create_criterion(config: TrainConfig) -> nn.Module:
    """
    Create loss function with label smoothing.
    创建带标签平滑的损失函数。
    """
    return nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
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
        total_correct += (
            (outputs.argmax(dim=1) == labels).sum().item()
        )
        total_samples += images.size(0)

        # Progress every 100 batches / 每 100 个 batch 打印进度
        if (batch_idx + 1) % 100 == 0:
            batch_acc = total_correct / total_samples
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)}"
                f" | Loss: {total_loss / total_samples:.4f}"
                f" | Acc: {batch_acc:.4f}"
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

    History contains per-epoch:
    train_loss, train_acc, test_loss, test_acc, lr.
    历史包含每轮：
    train_loss, train_acc, test_loss, test_acc, lr。
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # Create optimizer, scheduler, criterion via factories
    # 通过工厂函数创建优化器、调度器、损失函数
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_criterion(config)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }
    best_acc = 0.0
    # Early stopping counter / 早停计数器
    epochs_without_improvement = 0

    # Import here to avoid circular dependency
    # 延迟导入避免循环依赖
    from .checkpoint import (
        save_best_checkpoint,
        save_feature_extractor,
    )

    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config.epochs} | LR: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch,
        )
        test_loss, test_acc = evaluate(
            model, test_loader, device
        )
        if scheduler is not None:
            scheduler.step()

        # Record history / 记录训练历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        print(
            f"  Summary | Train Loss: {train_loss:.4f}"
            f" | Train Acc: {train_acc:.4f}"
            f" | Test Loss: {test_loss:.4f}"
            f" | Test Acc: {test_acc:.4f}"
        )

        # Save best model + early stopping check
        # 保存最佳模型 + 早停判断
        if test_acc > best_acc + config.min_delta:
            best_acc = test_acc
            epochs_without_improvement = 0
            save_best_checkpoint(
                model, optimizer, epoch, test_acc, config
            )
            save_feature_extractor(model, config)
            print(f"  ** New best accuracy: {best_acc:.4f} **")
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for "
                f"{epochs_without_improvement}/{config.patience}"
                f" epochs"
            )
            if epochs_without_improvement >= config.patience:
                print(
                    f"\nEarly stopping triggered after "
                    f"{epoch} epochs / 触发早停"
                )
                break

    print(
        f"\nTraining complete. "
        f"Best test accuracy: {best_acc:.4f}"
    )
    return history
