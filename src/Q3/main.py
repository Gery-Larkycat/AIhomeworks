"""
ResNet-18 CIFAR-100 训练和评估主入口。
Main entry point for ResNet-18 CIFAR-100 training and evaluation.

三个分支：
1. 默认：CIFAR-100 训练（skorch 管线）
2. --transfer：CIFAR-100 预训练 → CIFAR-10 微调
3. --tv-transfer：torchvision ImageNet 预训练 → CIFAR-10 微调

CLI 用法 / Usage:
    python -m Q3.main                            # CIFAR-100 训练
    python -m Q3.main --transfer                 # 迁移学习
    python -m Q3.main --tv-transfer              # torchvision 迁移学习
    python -m Q3.main --search                   # 先搜索再训练
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

# 将 src/ 添加到路径以便导入
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from models.registry import make_config  # noqa: E402
from utils.cli import (  # noqa: E402
    add_common_train_args, add_transfer_args, apply_cli_overrides,
)
from utils.config import (  # noqa: E402
    SearchConfig, generate_timestamp, make_run_dir, dataset_prefix,
)
from utils.data import get_datasets, get_loaders  # noqa: E402
from utils.pipeline import (  # noqa: E402
    train_skorch, evaluate_and_report,
)


def _run_transfer(args: argparse.Namespace) -> None:
    """
    迁移学习：CIFAR-100 预训练 → CIFAR-10 微调。
    支持 --search / --search-only 进行迁移超参搜索。
    """
    from Q3.config import TransferConfig
    from Q3.data import get_cifar10_loaders
    from Q3.transfer import (
        find_best_cifar100_checkpoint,
        load_pretrained_model,
        freeze_backbone,
        run_transfer,
        run_transfer_search,
        load_transfer_search_params,
        _to_train_config,
    )
    from Q3.checkpoint import load_full_checkpoint

    # 时间戳运行目录 / Timestamped run dir
    timestamp = generate_timestamp()
    run_dir = make_run_dir("Q3", timestamp=timestamp)
    print(f"Run directory: {run_dir}")

    # 构建 TransferConfig / Build TransferConfig
    config = TransferConfig()
    overrides = apply_cli_overrides(args)
    overrides["checkpoint_dir"] = run_dir

    # 自动发现或手动指定源检查点
    if args.transfer_checkpoint is not None:
        overrides["source_checkpoint"] = Path(args.transfer_checkpoint)
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
        ckpt_info = torch.load(auto_ckpt, weights_only=False)
        print(
            f"Auto-selected source: {auto_ckpt}"
            f" (accuracy: {ckpt_info.get('accuracy', 'N/A'):.4f})"
        )

    if overrides:
        config = dataclasses.replace(config, **overrides)

    # 仅搜索模式 / Search-only mode
    if args.search_only:
        run_transfer_search(config)
        return

    # 从指定文件加载搜索结果
    if args.search_results:
        specific = Path(args.search_results)
        best_params = load_transfer_search_params(specific_file=specific)
        if best_params is not None:
            config = dataclasses.replace(config, **best_params)
            print(f"\nLoaded transfer search params from: {specific}")
            for k, v in best_params.items():
                print(f"  {k}: {v}")

    # 运行迁移学习 / Run transfer
    history = run_transfer(config, search=args.search)

    # 最终评估 / Final evaluation
    train_config = _to_train_config(config)
    _, test_loader = get_cifar10_loaders(train_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 重新加载最佳模型 / Reload best model
    model = load_pretrained_model(
        config.source_checkpoint,
        source_num_classes=config.source_num_classes,
        target_num_classes=config.num_classes,
    )
    freeze_backbone(model)
    model = model.to(device)

    prefix = dataset_prefix(config.num_classes, train_config.task_tag)
    best_path = train_config.checkpoint_dir / f"{prefix}_best.pth"
    if best_path.exists():
        load_full_checkpoint(best_path, model)
        print(f"  Loaded best checkpoint: {best_path}")

    evaluate_and_report(
        model, test_loader, device, config.num_classes,
        history, train_config.checkpoint_dir,
    )
    print("\nTransfer learning done! / 迁移学习完成！")


def _run_torchvision_transfer(args: argparse.Namespace) -> None:
    """
    PyTorch 预训练 ResNet-18 → CIFAR-10 迁移学习。
    torchvision ImageNet pretrained → CIFAR-10 finetune.
    """
    from Q3.config import TorchvisionTransferConfig
    from Q3.torchvision_transfer import (
        load_torchvision_pretrained,
        get_cifar10_224_loaders,
        run_torchvision_transfer,
    )
    from Q3.checkpoint import load_full_checkpoint

    timestamp = generate_timestamp()
    run_dir = make_run_dir("Q3", timestamp=timestamp)
    print(f"Run directory: {run_dir}")

    # 构建配置 / Build config
    config = TorchvisionTransferConfig()
    overrides = apply_cli_overrides(args)
    overrides["checkpoint_dir"] = run_dir
    if overrides:
        config = dataclasses.replace(config, **overrides)

    # 运行迁移学习 / Run transfer learning
    history = run_torchvision_transfer(config)

    # 最终评估 / Final evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_torchvision_pretrained(config.num_classes).to(device)

    prefix = dataset_prefix(config.num_classes, "tvtransfer")
    best_path = run_dir / f"{prefix}_best.pth"
    if best_path.exists():
        load_full_checkpoint(best_path, model)
        print(f"  Loaded best checkpoint: {best_path}")

    _, test_loader = get_cifar10_224_loaders(config, augment=False)
    evaluate_and_report(
        model, test_loader, device, config.num_classes,
        history, run_dir,
    )
    print("\nTorchvision transfer learning done!"
          " / PyTorch 预训练迁移学习完成！")


def main() -> None:
    """ResNet-18 CIFAR-100 主入口 / Main entry."""
    parser = argparse.ArgumentParser(
        description="ResNet-18 CIFAR-100 Training"
        " / ResNet-18 CIFAR-100 训练",
    )
    add_common_train_args(parser)
    add_transfer_args(parser)
    args = parser.parse_args()

    # ---- Transfer learning branches / 迁移学习分支 ----
    if args.transfer:
        _run_transfer(args)
        return
    if args.tv_transfer:
        _run_torchvision_transfer(args)
        return

    # ---- CIFAR-100 training branch / CIFAR-100 训练分支 ----
    timestamp = generate_timestamp()
    run_dir = make_run_dir("Q3", timestamp=timestamp)
    overrides = apply_cli_overrides(args)
    overrides["checkpoint_dir"] = run_dir
    config = make_config("Q3", **overrides)

    # 打印信息头 / Print header
    print("=" * 60)
    print("ResNet-18 CIFAR-100 Training")
    print("ResNet-18 CIFAR-100 训练")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Config: {config}")
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 加载数据集 / Load datasets
    print("\nLoading CIFAR-100 dataset / 加载 CIFAR-100 数据集...")
    train_ds, test_ds = get_datasets(config)
    print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    # ---- Hyperparameter search / 超参数搜索 ----
    if args.search or args.search_only:
        from Q3.search import run_search
        search_overrides = {}
        if args.search_strategy is not None:
            search_overrides["strategy"] = args.search_strategy
        search_cfg = SearchConfig(**search_overrides)
        best_params = run_search(config, search_cfg=search_cfg)

        if args.search_only:
            print("\nSearch-only mode. Exiting. / 仅搜索模式，退出。")
            return

        config = dataclasses.replace(config, **best_params)
        print("\nUsing best params from search / 使用搜索得到的最佳配置:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        train_ds, test_ds = get_datasets(config)

    elif not args.ignore_search:
        from Q3.search import load_best_search_params
        specific = Path(args.search_results) if args.search_results else None
        best_params = load_best_search_params(specific_file=specific)
        if best_params is not None:
            config = dataclasses.replace(config, **best_params)
            print("\nLoaded best params from search results"
                  " / 从搜索结果加载最优参数:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            train_ds, test_ds = get_datasets(config)

    # 训练 / Train
    print(f"\nStarting training for {config.epochs} epochs"
          f" / 开始训练 {config.epochs} 轮...")
    net, history = train_skorch(config, train_ds, test_ds)

    # 评估 / Evaluate
    print("\n" + "=" * 60)
    print("Final Evaluation / 最终评估")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.module_.to(device)
    _, test_loader = get_loaders(config)
    evaluate_and_report(
        model, test_loader, device, config.num_classes,
        history, config.checkpoint_dir,
    )

    # 验证迁移学习就绪 / Verify transfer learning readiness
    from Q2.model import get_feature_extractor_state
    feature_state = get_feature_extractor_state(model)
    print(
        f"\nFeature extractor:"
        f" {len(feature_state)} parameter tensors (FC excluded)"
    )
    print(
        "Feature extractor saved."
        " Ready for CIFAR-10 transfer learning."
    )

    print("\nDone! / 完成！")


if __name__ == "__main__":
    main()
