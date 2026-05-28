"""
Main entry point for ResNet-18 CIFAR-100 training and evaluation.
ResNet-18 CIFAR-100 训练和评估的主入口。

Orchestrates: data loading → training → evaluation → visualization.
编排：数据加载 → 训练 → 评估 → 可视化。

CIFAR-100 训练使用 Q2/training.train_resnet (skorch)，
迁移学习分支使用 Q3/transfer.py 和 Q3/torchvision_transfer.py（保留旧式 train 循环）。

Supports hyperparameter search via --search / --search-only (skorch + sklearn).
通过 --search / --search-only 支持超参数搜索（skorch + sklearn）。

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

# Add src/ to path for imports (Q2, utils packages)
# 将 src/ 添加到路径以便导入 Q2、utils 包
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
# Also add project root for src.Q3 style imports
# 同时添加项目根目录以支持 src.Q3 风格的导入
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.Q3.config import TrainConfig  # noqa: E402
from src.Q3.data import (  # noqa: E402
    get_cifar100_datasets, get_cifar100_loaders,
)
from src.Q3.search import run_search  # noqa: E402

# 共享模块导入 / Shared module imports
# noqa: E402 — sys.path.insert above makes these importable
from Q2.model import get_feature_extractor_state  # noqa: E402
from Q2.training import train_resnet  # noqa: E402
from utils.evaluate import (  # noqa: E402
    per_class_accuracy, confusion_matrix,
)
from utils.visualize import (  # noqa: E402
    plot_training_curves, plot_confusion_matrix,
    plot_lr_schedule,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments / 解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "ResNet-18 CIFAR-100 Training"
            " / ResNet-18 CIFAR-100 训练"
        ),
    )
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
        "--eval-only", action="store_true",
        help=(
            "Only run evaluation"
            " (requires existing checkpoint)"
            " / 仅评估（需要已有检查点）"
        ),
    )
    # Search-related arguments / 搜索相关参数
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
    # Transfer learning arguments / 迁移学习参数
    parser.add_argument(
        "--transfer", action="store_true",
        help=(
            "Transfer learn from CIFAR-100 to CIFAR-10"
            " / 迁移学习 CIFAR-100 → CIFAR-10"
        ),
    )
    parser.add_argument(
        "--transfer-checkpoint", type=str, default=None,
        help=(
            "Override source checkpoint path for transfer"
            " / 覆盖迁移学习源检查点路径"
        ),
    )
    parser.add_argument(
        "--dropout", type=float, default=None,
        help=(
            "Dropout rate before FC layer (0 = disabled)"
            " / FC 前的 Dropout 比率（0 = 禁用）"
        ),
    )
    parser.add_argument(
        "--tv-transfer", action="store_true",
        help=(
            "Transfer learn using PyTorch pretrained ResNet-18"
            " / 使用 PyTorch 预训练模型迁移学习"
        ),
    )
    parser.add_argument(
        "--search-results", type=str, default=None,
        help=(
            "Specify search results file"
            " / 指定超参搜索结果文件路径"
        ),
    )
    return parser.parse_args()


def build_config(
    args: argparse.Namespace, checkpoint_dir: Path,
) -> TrainConfig:
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
    if args.amp:
        overrides["use_amp"] = True
    if args.no_augmentation:
        from src.Q3.config import AugmentationConfig
        overrides["augmentation"] = dataclasses.replace(
            AugmentationConfig(), use_augmentation=False,
        )
    if args.data_root is not None:
        overrides["data_root"] = Path(args.data_root)
    if args.dropout is not None:
        overrides["dropout_rate"] = args.dropout
    return TrainConfig(**overrides)


def _run_torchvision_transfer(
    args: argparse.Namespace,
) -> None:
    """
    PyTorch 预训练 ResNet-18 迁移学习入口。
    加载 torchvision ImageNet 预训练权重 → 冻结 backbone → 训练 FC。
    """
    import dataclasses

    from src.Q3.config import (
        TorchvisionTransferConfig,
        generate_timestamp,
        make_run_dir,
        dataset_prefix,
    )
    from src.Q3.torchvision_transfer import (
        load_torchvision_pretrained,
        get_cifar10_224_loaders,
        run_torchvision_transfer,
    )
    # 使用 utils 中的评估和可视化 / Use utils for eval & vis
    from utils.evaluate import per_class_accuracy, confusion_matrix
    from utils.visualize import (
        plot_training_curves, plot_confusion_matrix,
        plot_lr_schedule,
    )
    from src.Q3.checkpoint import load_full_checkpoint

    # 时间戳运行目录 / Timestamped run dir
    timestamp = generate_timestamp()
    run_dir = make_run_dir("Q3", timestamp=timestamp)
    print(f"Run directory: {run_dir}")

    # 构建配置 / Build config
    config = TorchvisionTransferConfig()
    overrides: dict = {"checkpoint_dir": run_dir}

    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.amp:
        overrides["use_amp"] = True
    if args.no_augmentation:
        from src.Q3.config import AugmentationConfig
        overrides["augmentation"] = dataclasses.replace(
            AugmentationConfig(), use_augmentation=False,
        )
    if args.data_root is not None:
        overrides["data_root"] = Path(args.data_root)
    if overrides:
        config = dataclasses.replace(config, **overrides)

    # 运行迁移学习 / Run transfer learning
    history = run_torchvision_transfer(config)

    # 最终评估 / Final evaluation
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 重新加载最佳模型 / Reload best model
    model = load_torchvision_pretrained(config.num_classes)
    model = model.to(device)

    prefix = dataset_prefix(config.num_classes, "tvtransfer")
    best_path = run_dir / f"{prefix}_best.pth"
    if best_path.exists():
        load_full_checkpoint(best_path, model)
        print(f"  Loaded best checkpoint: {best_path}")

    _, test_loader = get_cifar10_224_loaders(
        config, augment=False,
    )

    # 使用 skorch 的 net 评估 / Evaluate using skorch's net
    # 需要重构: evaluate() 已移除，改用 per_class_accuracy + confusion_matrix
    # 使用 net.score() 获取 accuracy，或手动计算 loss/acc
    # TODO: 恢复 evaluate() 或用 skorch 内置评分
    # 暂时直接计算 loss/acc
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
        f"\nTest Loss: {test_loss:.4f}"
        f" | Test Accuracy: {test_acc:.4f}"
    )

    # 每类准确率 / Per-class accuracy
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

    # 混淆矩阵 / Confusion matrix
    cm = confusion_matrix(
        model, test_loader, device, config.num_classes
    )

    # 可视化 / Visualizations
    vis_dir = run_dir / "plots"
    plot_training_curves(history, vis_dir)
    plot_confusion_matrix(cm, vis_dir)
    plot_lr_schedule(history, vis_dir)
    print(f"Plots saved to {vis_dir}")

    print(
        "\nTorchvision transfer learning done!"
        " / PyTorch 预训练迁移学习完成！"
    )


def _run_transfer(args: argparse.Namespace) -> None:
    """
    迁移学习流程：CIFAR-100 预训练 → CIFAR-10 微调。
    支持 --search / --search-only 进行迁移超参搜索。
    自动发现准确率最高的 CIFAR-100 基础模型。
    """
    import dataclasses

    from src.Q3.config import (
        TransferConfig,
        generate_timestamp,
        make_run_dir,
        dataset_prefix,
    )
    from src.Q3.data import get_cifar10_loaders
    # 使用 utils 中的评估和可视化 / Use utils for eval & vis
    from utils.evaluate import per_class_accuracy, confusion_matrix
    from utils.visualize import (
        plot_training_curves, plot_confusion_matrix,
        plot_lr_schedule,
    )
    from src.Q3.transfer import (
        find_best_cifar100_checkpoint,
        load_pretrained_model,
        freeze_backbone,
        run_transfer,
        run_transfer_search,
        load_transfer_search_params,
        _to_train_config,
    )

    # 生成时间戳运行目录 / Generate timestamped run dir
    timestamp = generate_timestamp()
    run_dir = make_run_dir("Q3", timestamp=timestamp)
    print(f"Run directory: {run_dir}")

    # 构建 TransferConfig / Build TransferConfig
    config = TransferConfig()
    overrides: dict = {}
    overrides["checkpoint_dir"] = run_dir

    # 自动发现或手动指定源检查点
    # Auto-discover or manually specify source checkpoint
    if args.transfer_checkpoint is not None:
        overrides["source_checkpoint"] = Path(
            args.transfer_checkpoint
        )
    else:
        auto_ckpt = find_best_cifar100_checkpoint()
        if auto_ckpt is None:
            print(
                "ERROR: No CIFAR-100 checkpoint found.\n"
                "先运行 CIFAR-100 训练，"
                "或指定 --transfer-checkpoint <path>。"
            )
            sys.exit(1)
        overrides["source_checkpoint"] = auto_ckpt
        ckpt_info = torch.load(
            auto_ckpt, weights_only=False
        )
        print(
            f"Auto-selected source: {auto_ckpt}"
            f" (accuracy: {ckpt_info.get('accuracy', 'N/A'):.4f})"
        )

    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.amp:
        overrides["use_amp"] = True
    if args.no_augmentation:
        from src.Q3.config import AugmentationConfig
        overrides["augmentation"] = dataclasses.replace(
            AugmentationConfig(), use_augmentation=False,
        )
    if args.data_root is not None:
        overrides["data_root"] = Path(args.data_root)
    if overrides:
        config = dataclasses.replace(config, **overrides)

    # 仅搜索模式 / Search-only mode
    if args.search_only:
        run_transfer_search(config)
        return

    # 从指定文件加载搜索结果 / Load search results from file
    if args.search_results:
        specific = Path(args.search_results)
        best_params = load_transfer_search_params(
            specific_file=specific,
        )
        if best_params is not None:
            config = dataclasses.replace(config, **best_params)
            print(
                f"\nLoaded transfer search params from: {specific}"
            )
            for k, v in best_params.items():
                print(f"  {k}: {v}")

    # 运行迁移学习（含可选搜索）/ Run transfer (with optional search)
    history = run_transfer(
        config, search=args.search,
    )

    # 最终评估 / Final evaluation
    train_config = _to_train_config(config)
    _, test_loader = get_cifar10_loaders(train_config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 重新加载最佳模型进行评估 / Reload best model for eval
    model = load_pretrained_model(
        config.source_checkpoint,
        source_num_classes=config.source_num_classes,
        target_num_classes=config.num_classes,
    )
    freeze_backbone(model)
    model = model.to(device)

    # 加载最佳检查点 / Load best checkpoint
    from src.Q3.checkpoint import load_full_checkpoint
    prefix = dataset_prefix(
        config.num_classes, train_config.task_tag,
    )
    best_path = train_config.checkpoint_dir / f"{prefix}_best.pth"
    if best_path.exists():
        load_full_checkpoint(best_path, model)
        print(f"  Loaded best checkpoint: {best_path}")

    # 评估 / Evaluate (内联计算，因为 evaluate() 已移至 utils/evaluate 但不含 loss/acc)
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
        f"\nTest Loss: {test_loss:.4f}"
        f" | Test Accuracy: {test_acc:.4f}"
    )

    # 每类准确率 / Per-class accuracy
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

    # 混淆矩阵 / Confusion matrix
    cm = confusion_matrix(
        model, test_loader, device, config.num_classes
    )

    # 可视化 / Visualizations
    vis_dir = train_config.checkpoint_dir / "plots"
    plot_training_curves(history, vis_dir)
    plot_confusion_matrix(cm, vis_dir)
    plot_lr_schedule(history, vis_dir)
    print(f"Plots saved to {vis_dir}")

    print("\nTransfer learning done! / 迁移学习完成！")


def main() -> None:
    """Main entry / 主入口。"""
    args = parse_args()

    # ---- Transfer learning branch / 迁移学习分支 ----
    if args.transfer:
        _run_transfer(args)
        return

    # ---- Torchvision pretrained transfer branch ----
    # ---- PyTorch 预训练模型迁移学习分支 ----
    if args.tv_transfer:
        _run_torchvision_transfer(args)
        return

    # ---- CIFAR-100 training branch / CIFAR-100 训练分支 ----
    # 生成时间戳运行目录 / Generate timestamped run dir
    from src.Q3.config import generate_timestamp, make_run_dir
    timestamp = generate_timestamp()
    run_dir = make_run_dir("Q3", timestamp=timestamp)

    config = build_config(args, checkpoint_dir=run_dir)

    print("=" * 60)
    print("ResNet-18 CIFAR-100 Training")
    print("ResNet-18 CIFAR-100 训练")
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

    # 加载数据集（用于 skorch）/ Load datasets (for skorch)
    print(
        "\nLoading CIFAR-100 dataset"
        " / 加载 CIFAR-100 数据集..."
    )
    train_ds, test_ds = get_cifar100_datasets(config)
    print(
        f"Train: {len(train_ds)} samples"
        f" | Test: {len(test_ds)} samples"
    )

    # ---- Hyperparameter search / 超参数搜索 ----
    if args.search or args.search_only:
        from src.Q3.config import SearchConfig

        # Build search config with strategy override
        # 根据命令行参数构建搜索配置
        search_overrides = {}
        if args.search_strategy is not None:
            search_overrides["strategy"] = args.search_strategy
        search_cfg = SearchConfig(**search_overrides)

        best_params = run_search(
            config, search_cfg=search_cfg,
        )

        if args.search_only:
            print(
                "\nSearch-only mode. Exiting."
                "\n仅搜索模式，退出。"
            )
            return

        # Apply best params for full training
        # 应用最优参数进行完整训练
        config = dataclasses.replace(
            config, **best_params
        )
        print(
            "\nUsing best params from search"
            " / 使用搜索得到的最佳配置:"
        )
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # batch_size may have changed → rebuild datasets
        # batch_size 可能改变 → 重建数据集
        train_ds, test_ds = get_cifar100_datasets(config)
    elif not args.ignore_search:
        # Auto-load existing search results if available
        # 自动加载已有搜索结果（如果存在）
        from src.Q3.search import load_best_search_params

        specific = (
            Path(args.search_results)
            if args.search_results else None
        )
        best_params = load_best_search_params(
            specific_file=specific,
        )
        if best_params is not None:
            config = dataclasses.replace(
                config, **best_params
            )
            print(
                "\nLoaded best params from search results"
                " / 从搜索结果加载最优参数:"
            )
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            # Rebuild with new batch_size
            # 用新 batch_size 重建
            train_ds, test_ds = get_cifar100_datasets(config)

    # Train using Q2 skorch pipeline / 使用 Q2 skorch 管线训练
    print(
        f"\nStarting training for {config.epochs} epochs"
        f" / 开始训练 {config.epochs} 轮..."
    )
    net, history = train_resnet(
        config, train_ds, test_ds,
        save_feature_extractor=True,
    )

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

    # 使用 DataLoader 进行评估（需要 loader 而非 dataset）
    _, test_loader = get_cifar100_loaders(config)

    # 内联计算 test loss/acc / Inline test loss/acc computation
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

    # Verify transfer learning readiness
    # 验证迁移学习就绪
    print(
        "\nVerifying transfer learning readiness"
        " / 验证迁移学习就绪..."
    )
    feature_state = get_feature_extractor_state(model)
    print(
        f"Feature extractor:"
        f" {len(feature_state)} parameter tensors"
        f" (FC excluded)"
    )
    print(
        "Feature extractor saved."
        " Ready for CIFAR-10 transfer learning."
    )

    print("\nDone! / 完成！")


if __name__ == "__main__":
    main()
