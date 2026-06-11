"""
VGG-16 CIFAR-10 训练和评估主入口。
Main entry point for VGG-16 CIFAR-10 training and evaluation.

Orchestrates: data loading → hyperparameter search (optional) → training
              → evaluation → visualization.
编排：数据加载 → 超参数搜索（可选）→ 训练 → 评估 → 可视化。

CLI 用法 / Usage:
    python -m Q1.main                       # 默认训练
    python -m Q1.main --epochs 50           # 覆盖训练轮数
    python -m Q1.main --search              # 先搜索再训练
    python -m Q1.main --no-bn --no-dropout  # 禁用指定优化
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
from utils.cli import add_common_train_args, apply_cli_overrides  # noqa: E402
from utils.config import SearchConfig, generate_timestamp, make_run_dir  # noqa: E402
from utils.data import get_datasets, get_loaders  # noqa: E402
from utils.pipeline import (  # noqa: E402
    train_skorch, evaluate_and_report,
)
from Q1.search import run_q1_search, load_q1_best_params  # noqa: E402


def main() -> None:
    """VGG-16 CIFAR-10 主入口 / Main entry."""
    parser = argparse.ArgumentParser(
        description="VGG-16 CIFAR-10 Training / VGG-16 CIFAR-10 训练",
    )
    add_common_train_args(parser)
    args = parser.parse_args()

    # 构建配置：Q1 默认值 + CLI 覆盖
    timestamp = generate_timestamp()
    run_dir = make_run_dir("Q1", timestamp=timestamp)
    overrides = apply_cli_overrides(args)
    overrides["checkpoint_dir"] = run_dir
    config = make_config("Q1", **overrides)

    # 打印信息头 / Print header
    print("=" * 60)
    print("VGG-16 CIFAR-10 Training")
    print("VGG-16 CIFAR-10 训练")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Config: {config}")
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 加载数据集 / Load datasets
    print("\nLoading CIFAR-10 dataset / 加载 CIFAR-10 数据集...")
    train_ds, test_ds = get_datasets(config)
    print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

    # ---- Hyperparameter search / 超参数搜索 ----
    if args.search or args.search_only:
        search_overrides = {}
        if args.search_strategy is not None:
            search_overrides["strategy"] = args.search_strategy
        search_cfg = SearchConfig(**search_overrides)
        best_params = run_q1_search(config, search_cfg=search_cfg)

        if args.search_only:
            print("\nSearch-only mode. Exiting. / 仅搜索模式，退出。")
            return

        config = dataclasses.replace(config, **best_params)
        print("\nUsing best params from search / 使用搜索得到的最佳配置:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        # batch_size 可能改变 → 重建数据集
        train_ds, test_ds = get_datasets(config)

    elif not args.ignore_search:
        # 自动加载已有搜索结果 / Auto-load existing search results
        specific = Path(args.search_results) if args.search_results else None
        best_params = load_q1_best_params(specific_file=specific)
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

    print("\nDone! / 完成！")


if __name__ == "__main__":
    main()
