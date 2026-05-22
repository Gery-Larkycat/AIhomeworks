"""
Main entry point for ResNet-18 CIFAR-100 training and evaluation.
ResNet-18 CIFAR-100 训练和评估的主入口。

Orchestrates: model creation → data loading → training → evaluation → visualization.
编排：模型创建 → 数据加载 → 训练 → 评估 → 可视化。

IMPORTANT on Windows: This file uses if __name__ == "__main__" guard
because DataLoader with num_workers > 0 requires it on Windows.
Windows 重要提示：此文件使用 if __name__ == "__main__" 守卫，
因为 Windows 下 DataLoader 的 num_workers > 0 时需要此守卫。
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path for imports / 将项目根目录添加到路径以便导入
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.Q3.checkpoint import save_training_history
from src.Q3.config import TrainConfig
from src.Q3.data import get_cifar100_loaders
from src.Q3.evaluate import confusion_matrix, evaluate, per_class_accuracy
from src.Q3.model import create_model, get_feature_extractor_state
from src.Q3.train import train
from src.Q3.visualize import plot_confusion_matrix, plot_lr_schedule, plot_training_curves


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments / 解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="ResNet-18 CIFAR-100 Training / ResNet-18 CIFAR-100 训练"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs / 覆盖训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size / 覆盖批大小")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate / 覆盖学习率")
    parser.add_argument("--data-root", type=str, default=None, help="Override data root / 覆盖数据根目录")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation (requires existing checkpoint) / 仅评估（需要已有检查点）",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """
    Build config from defaults + CLI overrides.
    从默认值和命令行覆盖构建配置。
    """
    overrides: dict = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.data_root is not None:
        overrides["data_root"] = Path(args.data_root)
    return TrainConfig(**overrides)


def main() -> None:
    """Main entry / 主入口。"""
    args = parse_args()
    config = build_config(args)

    print("=" * 60)
    print("ResNet-18 CIFAR-100 Training / ResNet-18 CIFAR-100 训练")
    print("=" * 60)
    print(f"Config: {config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Create model / 创建模型
    model = create_model(num_classes=config.num_classes)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet-18 | Parameters: {num_params:,} | Classes: {config.num_classes}")

    # Load data / 加载数据
    print("\nLoading CIFAR-100 dataset / 加载 CIFAR-100 数据集...")
    train_loader, test_loader = get_cifar100_loaders(config)
    print(f"Train: {len(train_loader.dataset)} samples | Test: {len(test_loader.dataset)} samples")

    # Train / 训练
    print(f"\nStarting training for {config.epochs} epochs / 开始训练 {config.epochs} 轮...")
    history = train(model, train_loader, test_loader, config)

    # Save training history / 保存训练历史
    save_training_history(history, config)

    # Final evaluation / 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation / 最终评估")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Per-class accuracy / 每类准确率
    print("\nComputing per-class accuracy / 计算每类准确率...")
    class_acc = per_class_accuracy(model, test_loader, device, config.num_classes)
    top5_classes = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)[:5]
    bottom5_classes = sorted(class_acc.items(), key=lambda x: x[1])[:5]
    print("Top 5 classes / 最高 5 类:")
    for cls, acc in top5_classes:
        print(f"  Class {cls}: {acc:.4f}")
    print("Bottom 5 classes / 最低 5 类:")
    for cls, acc in bottom5_classes:
        print(f"  Class {cls}: {acc:.4f}")

    # Confusion matrix / 混淆矩阵
    print("\nComputing confusion matrix / 计算混淆矩阵...")
    cm = confusion_matrix(model, test_loader, device, config.num_classes)

    # Visualizations / 可视化
    print("\nGenerating visualizations / 生成可视化...")
    vis_dir = config.checkpoint_dir / "plots"
    plot_training_curves(history, vis_dir)
    plot_confusion_matrix(cm, vis_dir)
    plot_lr_schedule(history, vis_dir)
    print(f"Plots saved to {vis_dir}")

    # Verify transfer learning readiness / 验证迁移学习就绪
    print("\nVerifying transfer learning readiness / 验证迁移学习就绪...")
    feature_state = get_feature_extractor_state(model)
    print(f"Feature extractor: {len(feature_state)} parameter tensors (FC excluded)")
    print("Feature extractor saved. Ready for CIFAR-10 transfer learning.")

    print("\nDone! / 完成！")


if __name__ == "__main__":
    main()
