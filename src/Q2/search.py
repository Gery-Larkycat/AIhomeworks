"""
Q2 超参搜索：CIFAR-10 数据准备 + 调用通用搜索。
Q2 hyperparameter search: CIFAR-10 data preparation + generic search.
"""

import dataclasses
from pathlib import Path

from torchvision import datasets

from utils.config import SearchConfig, TrainConfig, make_search_dir
from utils.search import (
    load_best_search_params,
    prepare_search_data,
    run_search,
)

from .model import ResNet18


# TrainConfig 有效字段集合，用于过滤搜索结果
_VALID_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def run_q2_search(
    config,
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
        search_dir=make_search_dir("Q2"),
        num_workers=config.num_workers,
        suffix="cifar10_hp_search",
    )


def load_q2_best_params(
    specific_file: Path | None = None,
) -> dict | None:
    """
    加载 Q2 最优搜索参数（映射为 Q2TrainConfig 字段名）。
    默认扫描 outputs/Q2/search_results/*_cifar10_hp_search.json。
    """
    search_dir = make_search_dir("Q2")
    result = load_best_search_params(
        search_dir,
        valid_fields=_VALID_FIELDS,
        pattern="*_cifar10_hp_search.json",
        specific_file=(
            Path(specific_file)
            if isinstance(specific_file, str)
            else specific_file
        ),
    )
    return result
