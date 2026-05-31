"""
Training loop for ResNet-18 on CIFAR-100.
ResNet-18 在 CIFAR-100 上的训练循环。

Uses configurable optimizer, scheduler, and loss via factory functions.
通过工厂函数使用可配置的优化器、调度器和损失函数。
"""

import dataclasses

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, NAdam, RMSprop, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LRScheduler
from torch.utils.data import DataLoader

from utils.augment import apply_batch_augmentation
from utils.config import config_to_dict
from utils.evaluate import evaluate
from .config import AugmentationConfig, TrainConfig


def create_optimizer(
    model: nn.Module, config: TrainConfig
) -> Optimizer:
    """
    Create optimizer from config / 根据配置创建优化器。

    Supports: sgd, adam, adamw, rmsprop, nadam.
    支持：sgd, adam, adamw, rmsprop, nadam。
    """
    params = [p for p in model.parameters() if p.requires_grad]
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
    scaler: torch.amp.GradScaler | None = None,
    aug_config: AugmentationConfig | None = None,
    num_classes: int = 100,
) -> tuple[float, float]:
    """
    Train for one epoch, returning (loss, accuracy).
    训练一个 epoch，返回 (损失, 准确率)。

    Uses AMP (mixed precision) when scaler is provided.
    提供 scaler 时启用 AMP（混合精度）训练。

    When aug_config is provided, applies CutMix/Mixup batch augmentation
    before forward pass. Soft labels (float tensor) from CutMix/Mixup are
    handled by CrossEntropyLoss natively.
    提供 aug_config 时，在前向传播前应用 CutMix/Mixup 批次级增强。
    CutMix/Mixup 产生的 soft labels（浮点张量）由 CrossEntropyLoss 原生支持。

    Accumulates metrics as GPU tensors to avoid per-batch GPU→CPU sync.
    以 GPU 张量累积指标，避免每 batch 的 GPU→CPU 同步。
    """
    model.train()
    use_amp = scaler is not None
    # Whether batch augmentation is active
    # 批次级增强是否激活
    use_batch_aug = aug_config is not None and aug_config.use_augmentation
    # Accumulate on GPU, sync once at epoch end
    # 在 GPU 上累积，epoch 结束时一次性同步
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # non_blocking=True overlaps H2D transfer with computation
        # non_blocking=True 使数据传输与前向计算重叠
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Batch-level augmentation (CutMix/Mixup)
        # 批次级增强（CutMix/Mixup）
        if use_batch_aug:
            images, labels = apply_batch_augmentation(
                images, labels, aug_config, num_classes
            )

        # set_to_none=True avoids memset overhead
        # set_to_none=True 避免 memset 开销
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Accumulate as tensors (no .item() → no sync)
        # 以张量累积（不用 .item() → 不触发同步）
        batch_size = images.size(0)
        total_loss += loss.detach() * batch_size
        # Accuracy: use dominant label for soft labels
        # 准确率: soft labels 时用主导标签计算
        if labels.is_floating_point():
            # labels is (B, num_classes) soft labels → use original dominant
            # labels 是 (B, num_classes) soft labels → 用主导标签
            dominant = labels.argmax(dim=1)
            total_correct += (outputs.argmax(dim=1) == dominant).sum()
        else:
            total_correct += (outputs.argmax(dim=1) == labels).sum()
        total_samples += batch_size

        # Progress every 100 batches / 每 100 个 batch 打印进度
        if (batch_idx + 1) % 100 == 0:
            # Single sync for progress print (acceptable overhead)
            # 仅打印时同步一次（可接受的开销）
            batch_loss = (total_loss / total_samples).item()
            batch_acc = (total_correct / total_samples).item()
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)}"
                f" | Loss: {batch_loss:.4f}"
                f" | Acc: {batch_acc:.4f}"
            )

    # Single sync at epoch end / epoch 结束时一次同步
    avg_loss = (total_loss / total_samples).item()
    accuracy = (total_correct / total_samples).item()
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

    # Enable cuDNN auto-tuner for fixed input sizes (CIFAR: 32x32)
    # 启用 cuDNN 自动调优（CIFAR 输入尺寸固定为 32x32）
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Create optimizer, scheduler, criterion via factories
    # 通过工厂函数创建优化器、调度器、损失函数
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Disable label_smoothing when CutMix/Mixup active:
    # soft labels already provide similar regularization.
    # CutMix/Mixup 激活时禁用 label_smoothing:
    # soft labels 已提供类似的正则化效果。
    aug_config = config.augmentation
    batch_mix_active = (
        aug_config.use_augmentation
        and (aug_config.use_cutmix or aug_config.use_mixup)
        and aug_config.mix_prob > 0
    )
    if batch_mix_active:
        mix_config = dataclasses.replace(config, label_smoothing=0.0)
        criterion = create_criterion(mix_config)
    else:
        criterion = create_criterion(config)

    # AMP GradScaler for mixed precision on CUDA
    # CUDA 上混合精度训练的 GradScaler
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
            device, epoch, scaler=scaler,
            aug_config=aug_config, num_classes=config.num_classes,
        )
        test_loss, test_acc = evaluate(
            model, test_loader, device, use_amp=use_amp
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
                model, optimizer, epoch, test_acc, config,
                config_dict=config_to_dict(config),
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
