"""
Main entry point for VGG-16 CIFAR-10 training and evaluation.
VGG-16 CIFAR-10 训练和评估的主入口。

Orchestrates: data loading → hyperparameter search (optional) → training → evaluation → visualization.
编排：数据加载 → 超参数搜索（可选）→ 训练 → 评估 → 可视化。

Uses skorch NeuralNetClassifier for training.
使用 skorch NeuralNetClassifier 训练。

IMPORTANT on Windows: This file uses if __name__ == "__main__" guard
because DataLoader with num_workers > 0 requires it on Windows.
Windows 重要提示：此文件使用 if __name__ == "__main__" 守卫，
因为 Windows 下 DataLoader 的 num_workers > 0 时需要此守卫。
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

# Add src/ to path for imports (Q1, utils packages)
# 将 src/ 添加到路径以便导入 Q1、utils 包
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
# Also add project root for src.Q1 style imports
# 同时添加项目根目录以支持 src.Q1 风格的导入
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Q1.config import Q1TrainConfig  # noqa: E402
from Q1.data import get_cifar10_datasets, get_cifar10_loaders  # noqa: E402
from Q1.search import run_q1_search, load_q1_best_params  # noqa: E402
from Q1.training import train_vgg  # noqa: E402
from utils.config import (  # noqa: E402
    AugmentationConfig, SearchConfig,
    generate_timestamp, make_run_dir,
)
from utils.evaluate import per_class_accuracy, confusion_matrix  # noqa: E402
from utils.visualize import (  # noqa: E402
    plot_training_curves, plot_confusion_matrix, plot_lr_schedule,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments / 解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "VGG-16 CIFAR-10 Training"
            " / VGG-16 CIFAR-10 训练"
        ),
    )
    # Training overrides / 训练参数覆盖
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num epochs / 覆盖训练轮数",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size / 覆盖批大小",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate / 覆盖学习率",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data root / 覆盖数据根目录",
    )
    parser.add_argument(
        "--dropout", type=float, default=None,
        help=(
            "Dropout rate before FC layer (0 = disabled)"
            " / FC 前的 Dropout 比率（0 = 禁用）"
        ),
    )
    parser.add_argument(
        "--no-bn", action="store_true",
        help=(
            "Disable BatchNorm in conv layers"
            " / 禁用卷积层中的 BatchNorm"
        ),
    )
    # Evaluation / 评估
    parser.add_argument(
        "--eval-only", action="store_true",
        help=(
            "Only run evaluation"
            " (requires existing checkpoint)"
            " / 仅评估（需要已有检查点）"
        ),
    )
    # Search / 超参数搜索
    parser.add_argument(
        "--search", action="store_true",
        help=(
            "Run hyperparameter search,"
            " then train with best params"
            " / 先搜索再用最优配置训练"
        ),
    )
    parser.add_argument(
        "--search-only", action="store_true",
        help=(
            "Only run search, save results, and exit"
            " / 仅运行搜索并报告结果"
        ),
    )
    parser.add_argument(
        "--search-strategy",
        choices=["halving-random", "random", "grid"],
        default=None,
        help=(
            "Search strategy: halving-random (default), random, grid"
            " / 搜索策略：halving-random（默认）、随机、网格"
        ),
    )
    parser.add_argument(
        "--ignore-search", action="store_true",
        help=(
            "Ignore existing search results,"
            " use default/CLI params"
            " / 忽略已有搜索结果"
        ),
    )
    parser.add_argument(
        "--search-results", type=str, default=None,
        help=(
            "Specify search results file"
            " / 指定超参搜索结果文件路径"
        ),
    )
    # Other / 其他
    parser.add_argument(
        "--amp", action="store_true",
        help=(
            "Enable mixed precision (FP16) training"
            " / 启用混合精度训练"
        ),
    )
    parser.add_argument(
        "--no-augmentation", action="store_true",
        help=(
            "Disable data augmentation"
            " / 禁用数据增强"
        ),
    )
    return parser.parse_args()


def build_config(
    args: argparse.Namespace, checkpoint_dir: Path,
) -> Q1TrainConfig:
    """
    Build config from defaults + CLI overrides + timestamped dir.
    从默认值、命令行覆盖和时间戳目录构建配置。
    """
    overrides: dict = {"checkpoint_dir": checkpoint_dir}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.dropout is not None:
        overrides["dropout_rate"] = args.dropout
    if args.no_bn:
        overrides["use_bn"] = False
    if args.amp:
        overrides["use_amp"] = True
    if args.no_augmentation:
        overrides["augmentation"] = dataclasses.replace(
            AugmentationConfig(), use_augmentation=False,
        )
    if args.data_root is not None:
        overrides["data_root"] = Path(args.data_root)
    return Q1TrainConfig(**overrides)


def main() -> None:
    """Main entry / 主入口。"""
    args = parse_args()

    # Generate timestamped run dir / 生成时间戳运行目录
    timestamp = generate_timestamp()
    run_dir = make_run_dir("Q1", timestamp=timestamp)
    config = build_config(args, checkpoint_dir=run_dir)

    # Print header / 打印训练信息头
    print("=" * 60)
    print("VGG-16 CIFAR-10 Training")
    print("VGG-16 CIFAR-10 训练")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Config: {config}")
    device_name = (
        "CUDA" if torch.cuda.is_available() else "CPU"
    )
    print(f"Device: {device_name}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load CIFAR-10 datasets / 加载 CIFAR-10 数据集
    print(
        "\nLoading CIFAR-10 dataset"
        " / 加载 CIFAR-10 数据集..."
    )
    train_ds, test_ds = get_cifar10_datasets(config)
    print(
        f"Train: {len(train_ds)} samples"
        f" | Test: {len(test_ds)} samples"
    )

    # ---- Hyperparameter search / 超参数搜索 ----
    if args.search or args.search_only:
        search_overrides = {}
        if args.search_strategy is not None:
            search_overrides["strategy"] = args.search_strategy
        search_cfg = SearchConfig(**search_overrides)

        best_params = run_q1_search(config, search_cfg=search_cfg)

        if args.search_only:
            print(
                "\nSearch-only mode. Exiting."
                "\n仅搜索模式，退出。"
            )
            return

        # Apply best params / 应用最优参数
        config = dataclasses.replace(config, **best_params)
        print(
            "\nUsing best params from search"
            " / 使用搜索得到的最佳配置:"
        )
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # batch_size may have changed → rebuild datasets
        # batch_size 可能改变 → 重建数据集
        train_ds, test_ds = get_cifar10_datasets(config)

    elif not args.ignore_search:
        # Auto-load existing search results if available
        # 自动加载已有搜索结果（如果存在）
        specific = (
            Path(args.search_results)
            if args.search_results else None
        )
        best_params = load_q1_best_params(specific_file=specific)
        if best_params is not None:
            config = dataclasses.replace(config, **best_params)
            print(
                "\nLoaded best params from search results"
                " / 从搜索结果加载最优参数:"
            )
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            # Rebuild with new batch_size
            # 用新 batch_size 重建
            train_ds, test_ds = get_cifar10_datasets(config)

    # Train / 训练
    print(
        f"\nStarting training for {config.epochs} epochs"
        f" / 开始训练 {config.epochs} 轮..."
    )
    net, history = train_vgg(config, train_ds, test_ds)

    # Final evaluation / 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation / 最终评估")
    print("=" * 60)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # skorch net.module_ 是训练后的 PyTorch 模型
    model = net.module_
    model = model.to(device)

    # 测试 DataLoader / Test DataLoader for evaluation
    _, test_loader = get_cifar10_loaders(config)

    # Inline test loss/acc computation / 内联计算 test loss/acc
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
    test_loss = total_loss / total_samples
    test_acc = total_correct / total_samples
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
        model, test_loader, device, config.num_classes
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
        model, test_loader, device, config.num_classes
    )

    # Visualizations / 可视化
    print(
        "\nGenerating visualizations"
        " / 生成可视化..."
    )
    vis_dir = config.checkpoint_dir / "plots"
    plot_training_curves(history, vis_dir)
    plot_confusion_matrix(cm, vis_dir)
    plot_lr_schedule(history, vis_dir)
    print(f"Plots saved to {vis_dir}")

    print("\nDone! / 完成！")


if __name__ == "__main__":
    main()
