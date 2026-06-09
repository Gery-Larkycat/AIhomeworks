"""
Q1 VGG-16 消融实验入口。
Q1 VGG-16 CIFAR-10 ablation experiment entry point.

默认自动加载已有超参搜索的最优参数作为 baseline 配置，
与正常训练流程一致。可通过 --ignore-search 禁用。

用法 / Usage:
    python -m Q1.ablation                       # 全量 15 个实验
    python -m Q1.ablation --epochs 50           # 快速迭代
    python -m Q1.ablation --experiments baseline,no_bn  # 指定实验
    python -m Q1.ablation --ignore-search       # 忽略搜索结果，用默认参数
"""

import argparse
import dataclasses
import sys
from pathlib import Path

# 路径设置 / Path setup
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from Q1.config import Q1TrainConfig  # noqa: E402
from Q1.data import get_cifar10_datasets  # noqa: E402
from Q1.search import load_q1_best_params  # noqa: E402
from Q1.training import train_vgg  # noqa: E402
from utils.ablation import (  # noqa: E402
    filter_experiments,
    print_ablation_report,
    plot_ablation_results,
    run_ablation_suite,
    save_ablation_results,
)
from utils.config import generate_timestamp  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析消融实验 CLI 参数 / Parse ablation CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "VGG-16 CIFAR-10 Ablation Study"
            " / VGG-16 CIFAR-10 消融实验"
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs for all experiments / 覆盖训练轮数",
    )
    parser.add_argument(
        "--experiments", type=str, default=None,
        help=(
            "Comma-separated experiment names"
            " / 逗号分隔的实验名"
            " (e.g. baseline,no_bn)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory / 覆盖输出目录",
    )
    parser.add_argument(
        "--ignore-search", action="store_true",
        help=(
            "Ignore existing search results, use default params"
            " / 忽略已有搜索结果，使用默认参数"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """消融实验主入口 / Ablation experiment main entry."""
    args = parse_args()

    # 构建基础配置 / Build base config
    # 与 main.py 一致：自动加载搜索结果的最优参数
    base_config = Q1TrainConfig()
    if not args.ignore_search:
        best_params = load_q1_best_params()
        if best_params is not None:
            base_config = dataclasses.replace(base_config, **best_params)
            print(
                "Loaded best params from search results"
                " / 从搜索结果加载最优参数:"
            )
            for k, v in best_params.items():
                print(f"  {k}: {v}")
        else:
            print(
                "No search results found, using defaults"
                " / 未找到搜索结果，使用默认参数"
            )

    # 输出目录 / Output directory
    timestamp = generate_timestamp()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"outputs/Q1/ablation/{timestamp}")
    )

    print("=" * 60)
    print("VGG-16 CIFAR-10 Ablation Study")
    print("VGG-16 CIFAR-10 消融实验")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print()

    # 过滤实验 / Filter experiments
    experiments = filter_experiments(args.experiments)
    print(f"Experiments to run: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp.name}: {exp.description}")
    print()

    # 额外配置覆盖 / Extra config overrides
    extra_overrides = {}
    if args.epochs is not None:
        extra_overrides["epochs"] = args.epochs

    # 运行消融实验套件 / Run ablation suite
    results = run_ablation_suite(
        default_config=base_config,
        train_fn=train_vgg,
        get_datasets_fn=get_cifar10_datasets,
        output_dir=output_dir,
        question_name="Q1",
        experiments=experiments,
        extra_config_overrides=extra_overrides or None,
    )

    # 输出报告 / Output reports
    print_ablation_report(results)
    save_ablation_results(results, output_dir)
    plot_ablation_results(results, output_dir)

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
