"""
共享消融实验框架。
Shared ablation experiment framework.

设计思路:
- 标准化实验矩阵（15 个消融实验）
- 通用训练/评估流程，由各题目提供具体实现
- 自动汇总结果、生成对比图表、保存 JSON/CSV

被 Q1/Q2/Q3 的 ablation.py 调用，不在本模块内直接运行。
"""

import csv
import json
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# CJK 兼容字体 / CJK-compatible fonts
mpl.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


# ===========================================================================
# 实验矩阵 / Experiment matrix
# ===========================================================================


@dataclass(frozen=True)
class AblationExperiment:
    """
    消融实验定义：禁用某项优化技术后的训练配置。
    One ablation experiment: disable one technique.
    """
    # 实验标识 / Experiment identifier
    name: str
    # 人类可读描述 / Human-readable description
    description: str
    # QnTrainConfig 字段覆盖 / Config field overrides
    config_overrides: dict
    # AugmentationConfig 字段覆盖 / Augmentation overrides
    aug_overrides: dict


# 标准消融实验矩阵：baseline + 14 个 one-off 实验
# Standard ablation matrix: baseline + 14 one-off experiments
ABLATION_EXPERIMENTS: list[AblationExperiment] = [
    AblationExperiment(
        name="baseline",
        description="All techniques enabled / 全部优化启用",
        config_overrides={},
        aug_overrides={},
    ),
    AblationExperiment(
        name="no_scheduler",
        description="Disable LR scheduler / 禁用学习率调度",
        config_overrides={"use_scheduler": False},
        aug_overrides={},
    ),
    AblationExperiment(
        name="no_bn",
        description="Disable BatchNorm / 禁用 BN",
        config_overrides={"use_bn": False},
        aug_overrides={},
    ),
    AblationExperiment(
        name="no_weight_decay",
        description="Disable weight decay / 禁用权重衰减",
        config_overrides={"weight_decay": 0.0},
        aug_overrides={},
    ),
    AblationExperiment(
        name="no_label_smoothing",
        description="Disable label smoothing / 禁用标签平滑",
        config_overrides={"label_smoothing": 0.0},
        aug_overrides={},
    ),
    AblationExperiment(
        name="no_dropout",
        description="Disable dropout / 禁用 Dropout",
        config_overrides={"dropout_rate": 0.0},
        aug_overrides={},
    ),
    AblationExperiment(
        name="no_early_stopping",
        description="Disable early stopping / 禁用早停",
        config_overrides={"use_early_stopping": False},
        aug_overrides={},
    ),
    AblationExperiment(
        name="no_cutmix",
        description="Disable CutMix / 禁用 CutMix",
        config_overrides={},
        aug_overrides={"use_cutmix": False},
    ),
    AblationExperiment(
        name="no_mixup",
        description="Disable Mixup / 禁用 Mixup",
        config_overrides={},
        aug_overrides={"use_mixup": False},
    ),
    AblationExperiment(
        name="no_augmentation",
        description="Disable all augmentation / 禁用全部增强",
        config_overrides={},
        aug_overrides={"use_augmentation": False},
    ),
    AblationExperiment(
        name="no_geom_aug",
        description="Disable geometric aug / 禁用几何变换增强",
        config_overrides={},
        aug_overrides={"use_geom_aug": False},
    ),
    AblationExperiment(
        name="no_color_aug",
        description="Disable color aug / 禁用颜色变换增强",
        config_overrides={},
        aug_overrides={"use_color_aug": False},
    ),
    AblationExperiment(
        name="no_noise_aug",
        description="Disable noise aug / 禁用噪声增强",
        config_overrides={},
        aug_overrides={"use_noise_aug": False},
    ),
    AblationExperiment(
        name="no_weather_aug",
        description="Disable weather aug / 禁用天气增强",
        config_overrides={},
        aug_overrides={"use_weather_aug": False},
    ),
    AblationExperiment(
        name="no_mixing_aug",
        description="Disable batch mixing aug / 禁用批次混合增强",
        config_overrides={},
        aug_overrides={"use_mixing_aug": False},
    ),
]


# ===========================================================================
# 结果数据结构 / Result data structures
# ===========================================================================


@dataclass
class AblationResult:
    """单个消融实验结果 / Result of one ablation experiment."""
    name: str
    description: str
    best_test_acc: float
    best_test_loss: float
    final_train_acc: float
    total_epochs: int
    total_time_sec: float
    status: str  # "success" | "failed"
    error: str  # 错误信息，成功时为空
    history: dict[str, list[float]]  # 完整训练历史


# ===========================================================================
# 核心函数 / Core functions
# ===========================================================================


def build_experiment_config(
    default_config,
    experiment: AblationExperiment,
    checkpoint_dir: Path,
    extra_overrides: dict | None = None,
):
    """
    从默认配置 + 实验覆盖构建训练配置。
    Build training config from defaults + experiment overrides.

    Args:
        default_config: 默认配置（frozen dataclass）
        experiment: 消融实验定义
        checkpoint_dir: 该实验的输出目录
        extra_overrides: 额外覆盖（如 epochs）

    Returns:
        修改后的配置 dataclass 实例
    """
    overrides = dict(experiment.config_overrides)
    overrides["checkpoint_dir"] = checkpoint_dir

    # 额外覆盖（如 --epochs）
    if extra_overrides:
        overrides.update(extra_overrides)

    # 处理增强覆盖 / Handle augmentation overrides
    if experiment.aug_overrides:
        aug = replace(
            default_config.augmentation,
            **experiment.aug_overrides,
        )
        overrides["augmentation"] = aug

    return replace(default_config, **overrides)


def run_single_ablation(
    experiment: AblationExperiment,
    config,
    train_fn,
    train_dataset,
    test_dataset,
    checkpoint_dir: Path,
) -> AblationResult:
    """
    运行单个消融实验。
    Run a single ablation experiment.

    Args:
        experiment: 实验定义
        config: 已构建的训练配置
        train_fn: 训练函数 (config, train_ds, test_ds) -> (net_or_model, history)
        train_dataset: 训练集
        test_dataset: 测试集
        checkpoint_dir: 该实验的输出目录

    Returns:
        AblationResult 实例
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(
        f"Experiment: {experiment.name}"
        f" / 实验: {experiment.name}"
    )
    print(f"  {experiment.description}")
    print(f"{'=' * 60}")

    start_time = time.time()
    try:
        net_or_model, history = train_fn(
            config, train_dataset, test_dataset,
        )
        elapsed = time.time() - start_time

        # 提取指标 / Extract metrics
        test_accs = history.get("test_acc", [])
        test_losses = history.get("test_loss", [])
        train_accs = history.get("train_acc", [])

        result = AblationResult(
            name=experiment.name,
            description=experiment.description,
            best_test_acc=max(test_accs) if test_accs else 0.0,
            best_test_loss=min(test_losses) if test_losses else float("inf"),
            final_train_acc=train_accs[-1] if train_accs else 0.0,
            total_epochs=len(test_accs),
            total_time_sec=round(elapsed, 1),
            status="success",
            error="",
            history=history,
        )
        print(
            f"  ✓ Done: acc={result.best_test_acc:.4f}"
            f" | epochs={result.total_epochs}"
            f" | time={result.total_time_sec}s"
        )

    except Exception as e:
        elapsed = time.time() - start_time
        result = AblationResult(
            name=experiment.name,
            description=experiment.description,
            best_test_acc=0.0,
            best_test_loss=float("inf"),
            final_train_acc=0.0,
            total_epochs=0,
            total_time_sec=round(elapsed, 1),
            status="failed",
            error=str(e),
            history={},
        )
        print(f"  ✗ Failed: {e}")

    return result


def run_ablation_suite(
    default_config,
    train_fn,
    get_datasets_fn,
    output_dir: Path,
    question_name: str,
    experiments: list[AblationExperiment] | None = None,
    extra_config_overrides: dict | None = None,
    train_fn_extra_kwargs: dict | None = None,
) -> list[AblationResult]:
    """
    运行完整消融实验套件。
    Run the full ablation experiment suite.

    Args:
        default_config: 默认训练配置
        train_fn: 训练函数 (config, train_ds, test_ds, **kwargs) -> (net, history)
        get_datasets_fn: 数据集加载函数 (config) -> (train_ds, test_ds)
        output_dir: 消融实验输出根目录
        question_name: 题目标识（如 "Q1", "Q2", "Q3"）
        experiments: 要运行的实验列表（None = 全部）
        extra_config_overrides: 额外配置覆盖（如 {"epochs": 50}）
        train_fn_extra_kwargs: 传递给 train_fn 的额外关键字参数

    Returns:
        所有实验结果的列表
    """
    if experiments is None:
        experiments = ABLATION_EXPERIMENTS
    if train_fn_extra_kwargs is None:
        train_fn_extra_kwargs = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AblationResult] = []

    total = len(experiments)
    for idx, exp in enumerate(experiments, 1):
        print(
            f"\n[{idx}/{total}] Running: {exp.name}"
            f" / 正在运行: {exp.name}"
        )

        # 每个实验独立的子目录 / Per-experiment subdirectory
        exp_dir = output_dir / exp.name

        # 构建该实验的配置 / Build per-experiment config
        config = build_experiment_config(
            default_config,
            exp,
            checkpoint_dir=exp_dir,
            extra_overrides=extra_config_overrides,
        )

        # 加载数据集（配置可能影响 transform）
        # Load datasets (config may affect transforms)
        train_ds, test_ds = get_datasets_fn(config)

        # 包装 train_fn 以传递额外参数
        # Wrap train_fn to pass extra kwargs
        def _train_fn(cfg, t_ds, v_ds):
            return train_fn(cfg, t_ds, v_ds, **train_fn_extra_kwargs)

        result = run_single_ablation(
            experiment=exp,
            config=config,
            train_fn=_train_fn,
            train_dataset=train_ds,
            test_dataset=test_ds,
            checkpoint_dir=exp_dir,
        )
        results.append(result)

    return results


# ===========================================================================
# 结果保存与报告 / Result saving & reporting
# ===========================================================================


def save_ablation_results(
    results: list[AblationResult],
    output_dir: Path,
) -> None:
    """
    保存消融实验结果为 JSON + CSV。
    Save ablation results as JSON + CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON: 完整结果（含 history） / JSON: full results with history
    json_data = []
    for r in results:
        json_data.append({
            "name": r.name,
            "description": r.description,
            "best_test_acc": r.best_test_acc,
            "best_test_loss": r.best_test_loss,
            "final_train_acc": r.final_train_acc,
            "total_epochs": r.total_epochs,
            "total_time_sec": r.total_time_sec,
            "status": r.status,
            "error": r.error,
            "history": r.history,
        })
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {json_path}")

    # CSV: 摘要（不含 history）/ CSV: summary without history
    csv_path = output_dir / "ablation_summary.csv"
    baseline_acc = _get_baseline_acc(results)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "description", "test_acc", "test_loss",
            "train_acc", "epochs", "time_sec", "delta_acc", "status",
        ])
        for r in results:
            delta = (
                r.best_test_acc - baseline_acc
                if baseline_acc is not None
                else 0.0
            )
            writer.writerow([
                r.name, r.description,
                f"{r.best_test_acc:.4f}",
                f"{r.best_test_loss:.4f}",
                f"{r.final_train_acc:.4f}",
                r.total_epochs,
                r.total_time_sec,
                f"{delta:+.4f}",
                r.status,
            ])
    print(f"Summary saved to {csv_path}")


def print_ablation_report(
    results: list[AblationResult],
) -> None:
    """
    在控制台打印消融实验结果表格。
    Print ablation results as a formatted table.
    """
    baseline_acc = _get_baseline_acc(results)

    print(f"\n{'=' * 90}")
    print("Ablation Study Results / 消融实验结果")
    print(f"{'=' * 90}")
    print(
        f"{'Experiment':<22} | {'Test Acc':>8} | "
        f"{'Delta':>8} | {'Epochs':>6} | {'Time(s)':>7} | {'Status':>7}"
    )
    print("-" * 90)

    for r in results:
        delta = (
            r.best_test_acc - baseline_acc
            if baseline_acc is not None else 0.0
        )
        delta_str = (
            f"{delta:+.4f}" if r.name != "baseline" else "  --"
        )
        print(
            f"{r.name:<22} | {r.best_test_acc:>8.4f} | "
            f"{delta_str:>8} | {r.total_epochs:>6} | "
            f"{r.total_time_sec:>7.1f} | {r.status:>7}"
        )

    print(f"{'=' * 90}\n")


# ===========================================================================
# 可视化 / Visualization
# ===========================================================================


def plot_ablation_results(
    results: list[AblationResult],
    output_dir: Path,
) -> None:
    """
    生成消融实验对比图表。
    Generate ablation comparison charts.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 仅绘制成功的实验 / Only plot successful experiments
    successful = [r for r in results if r.status == "success"]
    if not successful:
        print("No successful experiments to plot.")
        return

    baseline_acc = _get_baseline_acc(results)
    names = [r.name for r in successful]
    accs = [r.best_test_acc for r in successful]
    deltas = [
        r.best_test_acc - baseline_acc if baseline_acc else 0.0
        for r in successful
    ]
    times = [r.total_time_sec for r in successful]

    # 1. Accuracy comparison / 准确率对比柱状图
    _plot_accuracy_comparison(names, accs, plots_dir)

    # 2. Delta from baseline / 相对 baseline 变化柱状图
    _plot_accuracy_delta(names, deltas, plots_dir)

    # 3. Training time / 训练时间对比
    _plot_training_time(names, times, plots_dir)

    # 4. Curves overlay / 曲线叠加
    _plot_curves_overlay(successful, plots_dir)

    print(f"Plots saved to {plots_dir}")


def _plot_accuracy_comparison(
    names: list[str],
    accs: list[float],
    save_dir: Path,
) -> None:
    """柱状图：各实验测试准确率 / Bar chart: test accuracy per experiment."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#2ecc71" if n == "baseline" else "#3498db"
        for n in names
    ]
    bars = ax.barh(range(len(names)), accs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Test Accuracy / 测试准确率")
    ax.set_title(
        "Ablation: Test Accuracy Comparison"
        " / 消融实验：测试准确率对比"
    )
    # 在柱上标注数值 / Annotate bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}", va="center", fontsize=9,
        )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_comparison.png", dpi=150)
    plt.close()


def _plot_accuracy_delta(
    names: list[str],
    deltas: list[float],
    save_dir: Path,
) -> None:
    """柱状图：相对 baseline 的准确率变化 / Bar chart: accuracy delta."""
    # 跳过 baseline 本身 / Skip baseline itself
    filtered = [
        (n, d) for n, d in zip(names, deltas)
        if n != "baseline"
    ]
    if not filtered:
        return
    fnames, fdeltas = zip(*filtered)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in fdeltas]
    bars = ax.barh(range(len(fnames)), fdeltas, color=colors)
    ax.set_yticks(range(len(fnames)))
    ax.set_yticklabels(fnames)
    ax.set_xlabel("Accuracy Delta / 准确率变化")
    ax.set_title(
        "Ablation: Accuracy Delta from Baseline"
        " / 消融实验：相对 Baseline 准确率变化"
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    for bar, d in zip(bars, fdeltas):
        offset = 0.001 if d >= 0 else -0.001
        ha = "left" if d >= 0 else "right"
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{d:+.4f}", va="center", ha=ha, fontsize=9,
        )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_delta.png", dpi=150)
    plt.close()


def _plot_training_time(
    names: list[str],
    times: list[float],
    save_dir: Path,
) -> None:
    """柱状图：训练耗时对比 / Bar chart: training time comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#2ecc71" if n == "baseline" else "#3498db"
        for n in names
    ]
    bars = ax.barh(range(len(names)), times, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Training Time (seconds) / 训练时间（秒）")
    ax.set_title(
        "Ablation: Training Time Comparison"
        " / 消融实验：训练时间对比"
    )
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}s", va="center", fontsize=9,
        )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "training_time.png", dpi=150)
    plt.close()


def _plot_curves_overlay(
    results: list[AblationResult],
    save_dir: Path,
) -> None:
    """叠加图：所有实验 test_acc 曲线 / Overlay: all test_acc curves."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for r in results:
        test_accs = r.history.get("test_acc", [])
        if not test_accs:
            continue
        epochs = range(1, len(test_accs) + 1)
        # baseline 用粗线 / Bold for baseline
        lw = 2.5 if r.name == "baseline" else 1.2
        alpha = 1.0 if r.name == "baseline" else 0.7
        ax.plot(epochs, test_accs, label=r.name, linewidth=lw, alpha=alpha)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy / 测试准确率")
    ax.set_title(
        "Ablation: Test Accuracy Curves"
        " / 消融实验：测试准确率曲线"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "curves_overlay.png", dpi=150)
    plt.close()


# ===========================================================================
# 辅助函数 / Helper functions
# ===========================================================================


def _get_baseline_acc(
    results: list[AblationResult],
) -> float | None:
    """获取 baseline 实验的测试准确率 / Get baseline test accuracy."""
    for r in results:
        if r.name == "baseline" and r.status == "success":
            return r.best_test_acc
    return None


def filter_experiments(
    experiment_names: str | None,
) -> list[AblationExperiment]:
    """
    根据逗号分隔的实验名过滤实验矩阵。
    Filter experiment matrix by comma-separated names.

    Args:
        experiment_names: 逗号分隔的实验名（如 "baseline,no_bn"），
                         None 表示全部

    Returns:
        过滤后的实验列表
    """
    if experiment_names is None:
        return ABLATION_EXPERIMENTS

    names = {n.strip() for n in experiment_names.split(",")}
    filtered = [e for e in ABLATION_EXPERIMENTS if e.name in names]
    if not filtered:
        available = ", ".join(e.name for e in ABLATION_EXPERIMENTS)
        raise ValueError(
            f"No matching experiments. Available: {available}"
        )
    return filtered
