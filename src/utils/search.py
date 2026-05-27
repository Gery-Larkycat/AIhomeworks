"""
通用超参搜索模块：skorch + sklearn.model_selection。
Generic hyperparameter search via skorch + sklearn.

提供通用的搜索基础设施，每个作业只需：
1. 准备数据（调用 prepare_search_data）
2. 调用 run_search()

搜索空间（scipy.stats 分布）硬编码在此，适用于 ResNet-18。
如需为其他模型定制搜索空间，可传入自定义 param_distributions。
"""

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

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

from .config import SearchConfig


# ---------------------------------------------------------------------------
# skorch 参数名 → 常见训练配置字段名映射
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


def prepare_search_data(
    dataset, mean: tuple[float, ...], std: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 torchvision Dataset 转为归一化 numpy 数组。
    Convert torchvision Dataset to normalized numpy arrays.

    skorch 接收 numpy 数组后自动转 Tensor，sklearn CV 用索引切分。
    搜索阶段不用增强：目的是找 optimizer 参数，干净信号更可靠。

    Args:
        dataset: torchvision Dataset（需有 .data 和 .targets 属性）
        mean:    归一化均值
        std:     归一化标准差

    Returns:
        X: shape (N, 3, 32, 32) float32, 归一化后
        y: shape (N,) int64
    """
    X = dataset.data.astype(np.float32) / 255.0  # → [0, 1]
    # HWC → CHW 以匹配 PyTorch conv 期望
    X = X.transpose(0, 3, 1, 2)

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    X = (X - mean_arr) / std_arr

    y = np.array(dataset.targets, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Search space / 搜索空间
# ---------------------------------------------------------------------------


def _build_param_distributions(
    search_cfg: SearchConfig,
) -> dict:
    """
    构建 scipy.stats 分布参数空间，供 RandomizedSearchCV 使用。
    Build scipy.stats distribution parameter space.
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
    """
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
    checkpoint_dir: Path,
    search_cfg: SearchConfig,
) -> Path:
    """
    将 sklearn 搜索结果保存为 JSON。
    Save sklearn search results to JSON.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "hp_search_results.json"

    cv_results = search_obj.cv_results_

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
        all_candidates.append(
            {
                "params": params,
                "mean_test_score": round(float(cv_results["mean_test_score"][i]), 6),
                "std_test_score": round(float(cv_results["std_test_score"][i]), 6),
                "mean_fit_time": round(float(cv_results["mean_fit_time"][i]), 2),
                "rank": int(cv_results["rank_test_score"][i]),
            }
        )

    best_params = {
        k: (
            float(v)
            if isinstance(v, (np.floating, float))
            else int(v)
            if isinstance(v, (np.integer, int))
            else str(v)
        )
        for k, v in search_obj.best_params_.items()
    }

    results = OrderedDict(
        [
            (
                "search_config",
                {
                    "strategy": search_cfg.strategy,
                    "total_candidates": len(all_candidates),
                    "cv": search_cfg.cv,
                },
            ),
            (
                "best",
                OrderedDict(
                    [
                        ("params", best_params),
                        ("mean_test_score", round(float(search_obj.best_score_), 6)),
                    ]
                ),
            ),
            ("all_candidates", all_candidates),
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return path


def load_best_search_params(
    checkpoint_dir: Path,
    valid_fields: set[str] | None = None,
) -> dict[str, object] | None:
    """
    从 hp_search_results.json 加载最优超参数（映射为配置字段名）。
    Load best params from JSON, mapped to config field names.

    Args:
        checkpoint_dir: 检查点目录
        valid_fields:   有效字段集合，用于过滤无关参数。
                        None 时不过滤（返回所有映射后的字段）。

    Returns None if file doesn't exist.
    """
    path = checkpoint_dir / "hp_search_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    raw_params = data["best"]["params"]
    mapped = {PARAM_MAP.get(k, k): v for k, v in raw_params.items()}
    if valid_fields is not None:
        return {k: v for k, v in mapped.items() if k in valid_fields}
    return mapped


# ---------------------------------------------------------------------------
# Internal: skorch net for search / 搜索用 skorch 网络
# ---------------------------------------------------------------------------


def _create_search_net(
    model_class: type,
    model_kwargs: dict,
    search_cfg: SearchConfig,
    num_workers: int = 0,
) -> NeuralNetClassifier:
    """
    创建搜索用的 skorch NeuralNetClassifier。
    Create a skorch NeuralNetClassifier for hyperparameter search.

    设计决策（与 Q3 一致）:
    - 固定 optimizer=SGD：ResNet-18 标准选择
    - train_split=False：sklearn CV 负责数据划分
    - 无 scheduler/augmentation/label_smoothing：搜索目的是找 optimizer 参数
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return NeuralNetClassifier(
        model_class,
        **model_kwargs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        lr=0.1,
        optimizer__momentum=0.9,
        optimizer__weight_decay=5e-4,
        max_epochs=search_cfg.search_epochs_min,
        batch_size=256,
        iterator_train__shuffle=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__num_workers=num_workers,
        train_split=False,
        device=device,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Public API / 公共接口
# ---------------------------------------------------------------------------


def run_search(
    X: np.ndarray,
    y: np.ndarray,
    model_class: type,
    model_kwargs: dict,
    search_cfg: SearchConfig,
    checkpoint_dir: Path,
    num_workers: int = 0,
) -> dict[str, object]:
    """
    运行通用超参搜索并保存结果。
    Run generic hyperparameter search and save results.

    每个作业只需准备 (X, y) 数据，传入模型类和搜索配置。
    搜索空间（lr, momentum, weight_decay, batch_size, dropout_rate）
    硬编码在此，适用于 ResNet-18。如需定制可传入不同的 search_cfg。

    Args:
        X:             训练数据 (N, 3, 32, 32) float32
        y:             标签 (N,) int64
        model_class:   模型类（如 ResNet18）
        model_kwargs:  模型构造参数（如 {"num_classes": 100, "dropout_rate": 0.5}）
        search_cfg:    搜索配置
        checkpoint_dir: 结果保存目录
        num_workers:   DataLoader 工作线程数

    Returns:
        映射后的最优参数字典（配置字段名），可直接用于 dataclasses.replace()
    """
    strategy = search_cfg.strategy.lower()
    print("=" * 60)
    print("Hyperparameter Search (skorch + sklearn)")
    print("=" * 60)
    print(f"  Strategy: {strategy}")
    print(f"  CV folds: {search_cfg.cv}")
    print(f"  Scoring:  {search_cfg.scoring}")
    print()

    # 创建 skorch 网络
    net = _create_search_net(model_class, model_kwargs, search_cfg, num_workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 构建搜索器
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

    # 执行搜索
    print("\nStarting search / 开始搜索...")
    searcher.fit(X, y)

    # 保存结果
    results_path = _save_search_results(searcher, checkpoint_dir, search_cfg)

    # 输出摘要
    print("\n" + "=" * 60)
    print("Search Complete / 搜索完成")
    print("=" * 60)
    print(f"  Best score: {searcher.best_score_:.4f}")
    print("  Best params (raw):")
    for k, v in searcher.best_params_.items():
        print(f"    {k}: {v}")
    print(f"\n  Results saved to: {results_path}")

    # 返回映射后的参数
    raw_params = searcher.best_params_
    mapped = {PARAM_MAP.get(k, k): v for k, v in raw_params.items()}
    return mapped
