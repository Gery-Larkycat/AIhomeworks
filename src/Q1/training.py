"""
VGG-16 训练管线。
VGG-16 training pipeline using skorch.

用 skorch NeuralNetClassifier 替代手写训练循环，
获得内置计时（history['dur']）、早停、检查点管理。
"""

import torch
from utils.callbacks import extract_history
from utils.net import ClassifierNet, create_classifier_net

from .model import VGG16


def train_vgg(
    config,
    train_dataset,
    test_dataset,
) -> tuple[ClassifierNet, dict[str, list[float]]]:
    """
    完整的 VGG-16 skorch 训练管线。
    Full VGG-16 skorch training pipeline.

    使用 skorch NeuralNetClassifier 包装 VGG-16，自动配置:
    - 计时 (EpochTimer → history['dur'])
    - 早停 (EarlyStopping on valid_acc)
    - 检查点 (CustomCheckpoint)
    - 学习率调度 (CosineAnnealingLR)
    - CutMix/Mixup 批次增强 (via ClassifierNet.train_step_single)

    Args:
        config:               训练配置（鸭子类型，需有 utils/net.py 要求的字段）
        train_dataset:        训练集 Dataset
        test_dataset:         测试集 Dataset（作为验证集）

    Returns:
        (net, history_dict):
          net:          训练后的 ClassifierNet 实例
          history_dict: 标准历史字典 {train_loss, test_loss, train_acc,
                        test_acc, lr, dur}
    """
    # 启用 cuDNN 自动调优（CIFAR 输入尺寸固定为 32x32）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 创建配置好的 skorch 训练器
    net = create_classifier_net(
        model_class=VGG16,
        config=config,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    # 开始训练（skorch 自动处理 epoch 循环、早停、检查点）
    print(f"\nStarting training: {config.epochs} epochs, "
          f"batch_size={config.batch_size}")
    print(f"  Train: {len(train_dataset)} samples, "
          f"Test: {len(test_dataset)} samples")
    net.fit(train_dataset, y=None)

    # 提取标准格式历史（含 dur 计时）
    history = extract_history(net)

    # 打印最终摘要
    if history["test_acc"]:
        best_acc = max(history["test_acc"])
        total_dur = sum(history.get("dur", []))
        print(f"\nTraining complete. "
              f"Best test accuracy: {best_acc:.4f}, "
              f"Total time: {total_dur:.1f}s")

    return net, history
