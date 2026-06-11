"""
共享训练与评估管线。
Shared training and evaluation pipeline.

设计动机：
- Q1/train_vgg 和 Q2/train_resnet 90% 相同，仅 model_class 不同。
- Q1/Q2/Q3 main.py 中的评估块（loss/acc + per_class + cm + plots）复制了 5 次。
- 统一为 train_skorch() + evaluate_and_report() 消除所有重复。

实现思路：
- train_skorch(): 统一 skorch 训练，自动从 TaskSpec 推断 model_class
- evaluate_and_report(): 统一评估 + 报告 + 可视化
- 两者可独立使用，也可通过 run_train_eval_pipeline() 串联
"""

import torch

from .callbacks import extract_history
from .evaluate import evaluate, per_class_accuracy, confusion_matrix
from .net import create_classifier_net
from .visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_lr_schedule,
)


# ---------------------------------------------------------------------------
# Training / 训练
# ---------------------------------------------------------------------------


def train_skorch(
    config,
    train_dataset,
    test_dataset,
    model_class=None,
    save_feature_extractor=False,
):
    """
    统一 skorch 训练管线。
    Unified skorch training pipeline replacing Q1/train_vgg + Q2/train_resnet.

    若 model_class=None，从 TaskSpec 自动推断模型类和
    save_feature_extractor 标志（通过 task_name 或 model_name 匹配）。

    Args:
        config:                训练配置（鸭子类型，需有 utils/net.py 要求的字段）
        train_dataset:         训练集 Dataset
        test_dataset:          测试集 Dataset（作为验证集）
        model_class:           模型类（VGG16, ResNet18）；None 时自动推断
        save_feature_extractor: 是否额外保存特征提取器权重（Q3 迁移学习需要）

    Returns:
        (net, history_dict):
          net:          训练后的 ClassifierNet 实例
          history_dict: 标准历史字典 {train_loss, test_loss, train_acc,
                        test_acc, lr, dur}
    """
    # 自动推断 model_class（若未指定）
    if model_class is None:
        from models.registry import get_spec
        # 按 model_name + num_classes 推断任务名
        model_name = getattr(config, "model_name", "resnet18")
        num_classes = getattr(config, "num_classes", 10)
        # 尝试匹配已注册的 task
        for task_name in ("Q1", "Q2", "Q3"):
            try:
                spec = get_spec(task_name)
                if (spec.model_name == model_name
                        and spec.num_classes == num_classes):
                    model_class = spec.model_class
                    save_feature_extractor = spec.save_feature_extractor
                    break
            except KeyError:
                continue
        # 回退：默认 ResNet18
        if model_class is None:
            from models.resnet18 import ResNet18
            model_class = ResNet18

    # 启用 cuDNN 自动调优（CIFAR 输入尺寸固定为 32x32）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 创建配置好的 skorch 训练器
    net = create_classifier_net(
        model_class=model_class,
        config=config,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        save_feature_extractor=save_feature_extractor,
    )

    # 开始训练（skorch 自动处理 epoch 循环、早停、检查点）
    print(
        f"\nStarting training: {config.epochs} epochs, "
        f"batch_size={config.batch_size}"
    )
    print(
        f"  Train: {len(train_dataset)} samples, "
        f"Test: {len(test_dataset)} samples"
    )
    net.fit(train_dataset, y=None)

    # 提取标准格式历史（含 dur 计时）
    history = extract_history(net)

    # 打印最终摘要
    if history["test_acc"]:
        best_acc = max(history["test_acc"])
        total_dur = sum(history.get("dur", []))
        print(
            f"\nTraining complete. "
            f"Best test accuracy: {best_acc:.4f}, "
            f"Total time: {total_dur:.1f}s"
        )

    return net, history


# ---------------------------------------------------------------------------
# Evaluation / 评估
# ---------------------------------------------------------------------------


def evaluate_and_report(
    model,
    test_loader,
    device,
    num_classes,
    history,
    save_dir,
):
    """
    统一评估 + 报告 + 可视化管线。
    Unified evaluation + reporting + visualization pipeline.

    替代 main.py 中重复 5 次的评估块：
    test loss/acc → per-class accuracy → confusion matrix → plots。

    Args:
        model:        训练后的 PyTorch 模型（已在正确 device 上）
        test_loader:  测试集 DataLoader
        device:       torch.device
        num_classes:  分类数
        history:      训练历史 dict（来自 train_skorch 或 extract_history）
        save_dir:     可视化图表保存目录（checkpoint_dir 或 run_dir）
    """
    # Test loss/acc（使用 utils.evaluate.evaluate 的 GPU tensor 累积版本）
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(
        f"Test Loss: {test_loss:.4f}"
        f" | Test Accuracy: {test_acc:.4f}"
    )

    # Per-class accuracy / 每类准确率
    print(
        "\nComputing per-class accuracy"
        " / 计算每类准确率..."
    )
    class_acc = per_class_accuracy(
        model, test_loader, device, num_classes
    )
    top5 = sorted(
        class_acc.items(), key=lambda x: x[1], reverse=True
    )[:5]
    bottom5 = sorted(
        class_acc.items(), key=lambda x: x[1]
    )[:5]
    print("Top 5 classes / 最高 5 类:")
    for cls, acc in top5:
        print(f"  Class {cls}: {acc:.4f}")
    print("Bottom 5 classes / 最低 5 类:")
    for cls, acc in bottom5:
        print(f"  Class {cls}: {acc:.4f}")

    # Confusion matrix / 混淆矩阵
    print(
        "\nComputing confusion matrix"
        " / 计算混淆矩阵..."
    )
    cm = confusion_matrix(
        model, test_loader, device, num_classes
    )

    # Visualizations / 可视化
    print(
        "\nGenerating visualizations"
        " / 生成可视化..."
    )
    vis_dir = save_dir / "plots"
    plot_training_curves(history, vis_dir)
    plot_confusion_matrix(cm, vis_dir)
    plot_lr_schedule(history, vis_dir)
    print(f"Plots saved to {vis_dir}")
