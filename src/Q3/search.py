"""
Hyperparameter search via skorch + sklearn.model_selection.
通过 skorch + sklearn.model_selection 进行超参数搜索。

Wraps ResNet-18 with skorch NeuralNetClassifier, then delegates
search to sklearn's battle-tested CV tools:
将 ResNet-18 用 skorch NeuralNetClassifier 包装，然后委托给
sklearn 久经考验的交叉验证搜索工具：

  - "halving-random": HalvingRandomSearchCV (default, most efficient)
    逐步减半随机搜索（默认，最高效）
  - "random": RandomizedSearchCV
    随机搜索
  - "grid": GridSearchCV
    网格搜索
"""

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import loguniform, uniform
from skorch import NeuralNetClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)
from torchvision import datasets

from .config import SearchConfig, TrainConfig
from utils.config import (
    generate_timestamp,
    make_search_dir,
    find_best_search_result,
)
from Q2.model import ResNet18


# TrainConfig 有效字段集合，用于过滤搜索结果中的无关参数
# (HalvingRandomSearchCV 的 resource="max_epochs" 会出现在 best_params_ 中，
#  但那是 successive halving 的资源分配，不是正式训练的 epochs 参数)
_VALID_TRAIN_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


# ---------------------------------------------------------------------------
# skorch 参数名 → TrainConfig 字段名映射
# ---------------------------------------------------------------------------

PARAM_MAP: dict[str, str] = {
    "lr": "learning_rate",
    "optimizer__momentum": "momentum",
    "optimizer__weight_decay": "weight_decay",
    "batch_size": "batch_size",
    "module__dropout_rate": "dropout_rate",
}


# ---------------------------------------------------------------------------
# Data preparation / 数据准备
# ---------------------------------------------------------------------------

def _prepare_search_data(
    config: TrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    加载 CIFAR-100 训练集为 numpy 数组（仅 Normalize，无增强）。
    Load CIFAR-100 as numpy arrays (normalize only, no augmentation).

    skorch 接收 numpy 数组后自动转 Tensor，sklearn CV 用索引切分无需
    理解 Dataset。搜索阶段不用增强：目的是找 optimizer 参数，干净信号更可靠。

    Returns:
        X: shape (N, 3, 32, 32) float32, 归一化后
        y: shape (N,) int64
    """
    # load_data() 返回 (data, targets)，data 为 uint8 numpy
    dataset = datasets.CIFAR100(
        root=str(config.data_root),
        train=True,
        download=True,
    )
    # dataset.data shape: (50000, 32, 32, 3) uint8
    X = dataset.data.astype(np.float32) / 255.0  # → [0, 1]
    # HWC → CHW 以匹配 PyTorch conv 期望
    X = X.transpose(0, 3, 1, 2)

    # Normalize with CIFAR-100 stats / 用 CIFAR-100 统计量归一化
    mean = np.array(config.mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(config.std, dtype=np.float32).reshape(1, 3, 1, 1)
    X = (X - mean) / std

    y = np.array(dataset.targets, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# skorch net creation / skorch 网络创建
# ---------------------------------------------------------------------------

def _create_search_net(
    config: TrainConfig,
    search_cfg: SearchConfig,
) -> NeuralNetClassifier:
    """
    创建搜索用的 skorch NeuralNetClassifier。
    Create a skorch NeuralNetClassifier for hyperparameter search.

    设计决策：
    - 固定 optimizer=SGD：ResNet-18 标准，避免 optimizer 切换时参数不兼容
    - train_split=False：sklearn CV 负责数据划分，不重复 split
    - 无 scheduler：搜索轮数短，scheduler 不适用
    - 无 augmentation：搜索目的是找 optimizer 参数，干净信号更可靠
    - 无 label_smoothing：无 CutMix/Mixup 时 label_smoothing 意义不大
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return NeuralNetClassifier(
        ResNet18,
        module__num_classes=config.num_classes,
        module__dropout_rate=config.dropout_rate,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        lr=0.1,
        optimizer__momentum=0.9,
        optimizer__weight_decay=5e-4,
        max_epochs=search_cfg.search_epochs_min,
        batch_size=256,
        iterator_train__shuffle=True,
        iterator_train__num_workers=config.num_workers,
        iterator_valid__num_workers=config.num_workers,
        train_split=False,
        device=device,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Search space / 搜索空间
# ---------------------------------------------------------------------------

def _build_param_distributions(
    search_cfg: SearchConfig,
) -> dict:
    """
    构建 scipy.stats 分布参数空间，供 RandomizedSearchCV 使用。
    Build scipy.stats distribution parameter space.

    连续参数用分布采样，离散参数用候选列表。
    """
    return {
        "lr": loguniform(1e-4, 1.0),
        "optimizer__momentum": uniform(0.85, 0.14),  # [0.85, 0.99]
        "optimizer__weight_decay": loguniform(1e-6, 1e-2),
        "batch_size": list(search_cfg.batch_size_choices),
        "module__dropout_rate": uniform(0.0, 0.5),  # [0.0, 0.5]
    }


def _build_param_grid(search_cfg: SearchConfig) -> dict:
    """
    构建网格搜索参数空间。
    Build grid search parameter space.

    连续参数离散化为少量采样点，离散参数用候选列表。
    """
    import numpy as np

    return {
        "lr": np.logspace(-4, 0, 5).tolist(),
        "optimizer__momentum": np.linspace(0.85, 0.99, 4).tolist(),
        "optimizer__weight_decay": np.logspace(-6, -2, 4).tolist(),
        "batch_size": list(search_cfg.batch_size_choices),
        "module__dropout_rate": [0.0, 0.1, 0.3, 0.5],
    }


# ---------------------------------------------------------------------------
# Result logging / 结果保存
# ---------------------------------------------------------------------------

def _save_search_results(
    search_obj,
    search_cfg: SearchConfig,
    suffix: str = "cifar100_hp_search",
) -> Path:
    """
    将 sklearn 搜索结果保存为 JSON。
    Save sklearn search results to JSON.

    文件名格式：<timestamp>_<suffix>.json
    保存到 outputs/Q3/search_results/。
    """
    search_dir = make_search_dir("Q3")
    search_dir.mkdir(parents=True, exist_ok=True)
    timestamp = generate_timestamp()
    path = search_dir / f"{timestamp}_{suffix}.json"

    cv_results = search_obj.cv_results_

    # 提取所有候选的摘要 / Extract summary for all candidates
    all_candidates = []
    for i in range(len(cv_results["mean_test_score"])):
        params = {
            k: (
                float(v[i])
                if isinstance(v[i], (np.floating, float))
                else int(v[i])
                if isinstance(v[i], (np.integer, int))
                else str(v[i])
            )
            for k, v in cv_results.items()
            if k.startswith("param_")
        }
        all_candidates.append({
            "params": params,
            "mean_test_score": round(
                float(cv_results["mean_test_score"][i]), 6
            ),
            "std_test_score": round(
                float(cv_results["std_test_score"][i]), 6
            ),
            "mean_fit_time": round(
                float(cv_results["mean_fit_time"][i]), 2
            ),
            "rank": int(cv_results["rank_test_score"][i]),
        })

    # Best params / 最优参数
    best_params = {
        k: (
            float(v) if isinstance(v, (np.floating, float))
            else int(v) if isinstance(v, (np.integer, int))
            else str(v)
        )
        for k, v in search_obj.best_params_.items()
    }

    results = OrderedDict([
        ("search_config", {
            "strategy": search_cfg.strategy,
            "total_candidates": len(all_candidates),
            "cv": search_cfg.cv,
        }),
        ("best", OrderedDict([
            ("params", best_params),
            ("mean_test_score", round(float(search_obj.best_score_), 6)),
        ])),
        ("all_candidates", all_candidates),
    ])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return path


def load_best_search_params(
    specific_file: Path | None = None,
) -> dict[str, object] | None:
    """
    从搜索结果加载最优超参数（映射为 TrainConfig 字段名）。
    Load best params from search results, mapped to TrainConfig field names.

    如果指定 specific_file 则直接加载；
    否则扫描 outputs/Q3/search_results/*_cifar100_hp_search.json
    选 mean_test_score 最高的。

    Returns None if no matching file found.
    """
    if specific_file is not None:
        path = specific_file
        if not path.exists():
            return None
    else:
        search_dir = make_search_dir("Q3")
        path = find_best_search_result(
            search_dir,
            pattern="*_cifar100_hp_search.json",
        )
        if path is None:
            return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # 映射 skorch 参数名 → TrainConfig 字段名，过滤不属于 TrainConfig 的参数
    raw_params = data["best"]["params"]
    mapped = {PARAM_MAP.get(k, k): v for k, v in raw_params.items()}
    return {k: v for k, v in mapped.items() if k in _VALID_TRAIN_FIELDS}


# ---------------------------------------------------------------------------
# Public API / 公共接口
# ---------------------------------------------------------------------------

def run_search(
    config: TrainConfig,
    search_cfg: SearchConfig | None = None,
) -> dict[str, object]:
    """
    运行超参数搜索并保存结果。
    Run hyperparameter search and save results.

    用 skorch 包装 ResNet-18，委托给 sklearn 的搜索工具。
    数据仅做 Normalize（无增强），sklearn CV 自带 train/val split。

    Returns:
        映射后的最优参数字典（TrainConfig 字段名），可直接用于
        dataclasses.replace(config, **best_params)。
    """
    if search_cfg is None:
        search_cfg = SearchConfig()

    strategy = search_cfg.strategy.lower()
    print("=" * 60)
    print("Hyperparameter Search (skorch + sklearn)")
    print("超参数搜索（skorch + sklearn）")
    print("=" * 60)
    print(f"  Strategy: {strategy}")
    print(f"  CV folds: {search_cfg.cv}")
    print(f"  Scoring: {search_cfg.scoring}")
    print()

    # 准备数据 / Prepare data
    print("Loading CIFAR-100 for search / 加载搜索用数据...")
    X, y = _prepare_search_data(config)
    print(f"  X: {X.shape} {X.dtype}")
    print(f"  y: {y.shape} {y.dtype}")

    # 创建 skorch 网络 / Create skorch net
    net = _create_search_net(config, search_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 构建搜索器 / Build searcher
    if strategy == "halving-random":
        param_dist = _build_param_distributions(search_cfg)
        searcher = HalvingRandomSearchCV(
            net,
            param_dist,
            resource="max_epochs",
            max_resources=search_cfg.search_epochs_max,
            min_resources=search_cfg.search_epochs_min,
            factor=search_cfg.halving_factor,
            cv=search_cfg.cv,
            scoring=search_cfg.scoring,
            refit=False,
            random_state=42,
            n_jobs=1,
            verbose=1,
        )
    elif strategy == "random":
        param_dist = _build_param_distributions(search_cfg)
        searcher = RandomizedSearchCV(
            net,
            param_dist,
            n_iter=search_cfg.num_trials,
            cv=search_cfg.cv,
            scoring=search_cfg.scoring,
            refit=False,
            random_state=42,
            n_jobs=1,
            verbose=1,
        )
    elif strategy == "grid":
        param_grid = _build_param_grid(search_cfg)
        searcher = GridSearchCV(
            net,
            param_grid,
            cv=search_cfg.cv,
            scoring=search_cfg.scoring,
            refit=False,
            n_jobs=1,
            verbose=1,
        )
    else:
        raise ValueError(
            f"Unknown search strategy: {strategy}. "
            f"Supported: halving-random, random, grid"
        )

    # 执行搜索 / Run search
    print("\nStarting search / 开始搜索...")
    searcher.fit(X, y)

    # 保存结果 / Save results
    results_path = _save_search_results(searcher, search_cfg)

    # 输出摘要 / Print summary
    print("\n" + "=" * 60)
    print("Search Complete / 搜索完成")
    print("=" * 60)
    print(f"  Best score: {searcher.best_score_:.4f}")
    print("  Best params (raw):")
    for k, v in searcher.best_params_.items():
        print(f"    {k}: {v}")
    print(f"\n  Results saved to: {results_path}")

    # 返回映射后的参数 / Return mapped params (filtered to TrainConfig fields)
    raw_params = searcher.best_params_
    mapped = {PARAM_MAP.get(k, k): v for k, v in raw_params.items()}
    return {k: v for k, v in mapped.items() if k in _VALID_TRAIN_FIELDS}
