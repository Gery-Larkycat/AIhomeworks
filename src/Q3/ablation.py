"""
Q3 ResNet-18 CIFAR-100 消融实验入口。
Q3 ResNet-18 CIFAR-100 ablation experiment entry point.

默认自动加载已有超参搜索的最优参数作为 baseline 配置。
可通过 --ignore-search 禁用。

用法 / Usage:
    python -m Q3.ablation                       # 全量 15 个实验
    python -m Q3.ablation --epochs 50           # 快速迭代
    python -m Q3.ablation --experiments baseline,no_bn
    python -m Q3.ablation --ignore-search       # 忽略搜索结果
"""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.ablation import parse_ablation_args, run_ablation_main  # noqa: E402
from Q3.search import load_best_search_params  # noqa: E402


def main() -> None:
    """消融实验主入口 / Ablation experiment main entry."""
    args, _ = parse_ablation_args(
        "ResNet-18 CIFAR-100 Ablation Study"
        " / ResNet-18 CIFAR-100 消融实验"
    )
    run_ablation_main("Q3", load_best_search_params, args)


if __name__ == "__main__":
    main()
