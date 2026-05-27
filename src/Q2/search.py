"""
Q2 超参搜索：CIFAR-10 数据准备 + 调用通用搜索。
Q2 hyperparameter search: CIFAR-10 data preparation + generic search.
"""

import dataclasses

from torchvision import datasets

from utils.config import SearchConfig
from utils.search import (
    load_best_search_params,
    prepare_search_data,
    run_search,
)

from .config import Q2TrainConfig
from .model import ResNet18


# Q2TrainConfig 有效字段集合，用于过滤搜索结果
_VALID_Q2_FIELDS = {f.name for f in dataclasses.fields(Q2TrainConfig)}


def run_q2_search(
    config: Q2TrainConfig,
    search_cfg: SearchConfig | None = None,
) -> dict:
    """
    运行 Q2 超参搜索。
    Run Q2 hyperparameter search on CIFAR-10.

    准备 CIFAR-10 数据 → 调用 utils.search.run_search()。
    """
    if search_cfg is None:
        search_cfg = SearchConfig()

    # 加载 CIFAR-10 训练集（仅 Normalize，无增强）
    raw_dataset = datasets.CIFAR10(
        root=str(config.data_root),
        train=True,
        download=True,
    )
    X, y = prepare_search_data(raw_dataset, config.mean, config.std)

    return run_search(
        X=X,
        y=y,
        model_class=ResNet18,
        model_kwargs={
            "num_classes": config.num_classes,
            "dropout_rate": config.dropout_rate,
        },
        search_cfg=search_cfg,
        checkpoint_dir=config.checkpoint_dir,
        num_workers=config.num_workers,
    )


def load_q2_best_params(config: Q2TrainConfig) -> dict | None:
    """加载 Q2 最优搜索参数（映射为 Q2TrainConfig 字段名）。"""
    result = load_best_search_params(config.checkpoint_dir, _VALID_Q2_FIELDS)
    return result
