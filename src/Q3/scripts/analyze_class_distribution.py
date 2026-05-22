"""
CIFAR-100 类别分布分析脚本。
CIFAR-100 class distribution analysis script.

检查训练集和测试集的类别均衡性，
为后续数据增强和类别平衡策略提供依据。
Checks class balance in train/test splits,
informing data augmentation and class balancing strategies.
"""

import sys
from pathlib import Path

import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torchvision import datasets


def analyze_distribution(
    train_ds: datasets.CIFAR100,
    test_ds: datasets.CIFAR100,
) -> None:
    """
    分析并打印类别分布统计量。
    Analyze and print class distribution statistics.
    """
    train_labels = [label for _, label in train_ds]
    test_labels = [label for _, label in test_ds]

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    print("=" * 60)
    print("CIFAR-100 类别分布 / Class Distribution")
    print("=" * 60)
    print(f"训练集总数 / Train total: {len(train_labels)}")
    print(f"测试集总数 / Test total: {len(test_labels)}")
    print(f"类别数 / Num classes: {len(train_counts)}")
    print()

    # 逐类详情 / Per-class details
    print("--- 逐类分布 / Per-class ---")
    for cls in range(100):
        cls_name = train_ds.classes[cls]
        print(
            f"  {cls:3d} {cls_name:>20s}: "
            f"train={train_counts[cls]:4d} test={test_counts[cls]:4d}"
        )

    # 统计摘要 / Summary statistics
    vals = list(train_counts.values())
    test_vals = list(test_counts.values())
    print()
    print("--- 统计摘要 / Summary ---")
    print(
        f"  Train min={min(vals)} max={max(vals)} "
        f"mean={np.mean(vals):.1f} std={np.std(vals):.1f}"
    )
    print(
        f"  Test  min={min(test_vals)} max={max(test_vals)} "
        f"mean={np.mean(test_vals):.1f} std={np.std(test_vals):.1f}"
    )
    cv = np.std(vals) / np.mean(vals) * 100
    print(f"  变异系数 CV: {cv:.2f}%")
    print()

    if cv < 0.01:
        print("结论: 类别完全均衡，无需类别平衡算法。")
        print("Conclusion: Classes perfectly balanced, no balancing needed.")
    else:
        print("结论: 类别不均衡，需要考虑类别平衡策略。")
        print("Conclusion: Classes imbalanced, balancing needed.")


def main() -> None:
    train_ds = datasets.CIFAR100(
        root="data", train=True, download=True
    )
    test_ds = datasets.CIFAR100(
        root="data", train=False, download=True
    )
    analyze_distribution(train_ds, test_ds)


if __name__ == "__main__":
    main()
