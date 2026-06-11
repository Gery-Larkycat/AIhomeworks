"""
Q3 CIFAR-100 超参搜索（薄包装）。
Q3 CIFAR-100 hyperparameter search (thin wrapper).

委托到 utils.search 的通用搜索管线。
以前自己实现了 408 行搜索逻辑，现在仅保留 Q3 特有的参数。
"""

import dataclasses
from pathlib import Path

from torchvision import datasets

from utils.config import TrainConfig, SearchConfig, make_search_dir
from utils.search import (
    run_search as _run_search,
    load_best_search_params as _load_best,
    prepare_search_data,
)
from models.resnet18 import ResNet18


# TrainConfig 有效字段集合，过滤搜索结果中的无关参数
# (HalvingRandomSearchCV 的 resource="max_epochs" 会出现在 best_params_ 中)
_VALID_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def run_search(
    config,
    search_cfg: SearchConfig | None = None,
) -> dict:
    """
    运行 Q3 超参搜索。
    Run Q3 hyperparameter search on CIFAR-100.

    准备 CIFAR-100 数据 → 委托到 utils.search.run_search()。

    Args:
        config:      训练配置（鸭子类型）
        search_cfg:  搜索配置；None 时使用默认

    Returns:
        映射后的最优参数字典，可直接用于 dataclasses.replace()
    """
    if search_cfg is None:
        search_cfg = SearchConfig()

    # 加载 CIFAR-100 训练集（仅 Normalize，无增强）
    raw_dataset = datasets.CIFAR100(
        root=str(config.data_root),
        train=True,
        download=True,
    )
    X, y = prepare_search_data(raw_dataset, config.mean, config.std)

    return _run_search(
        X=X,
        y=y,
        model_class=ResNet18,
        model_kwargs={
            "num_classes": config.num_classes,
            "dropout_rate": config.dropout_rate,
        },
        search_cfg=search_cfg,
        search_dir=make_search_dir("Q3"),
        num_workers=config.num_workers,
        suffix="cifar100_hp_search",
    )


def load_best_search_params(
    specific_file: Path | None = None,
) -> dict | None:
    """
    从搜索结果加载最优超参数（映射为 TrainConfig 字段名）。
    Load best params from search results, mapped to TrainConfig field names.

    默认扫描 outputs/Q3/search_results/*_cifar100_hp_search.json，
    选 mean_test_score 最高的。

    Returns None if no matching file found.
    """
    return _load_best(
        make_search_dir("Q3"),
        valid_fields=_VALID_FIELDS,
        pattern="*_cifar100_hp_search.json",
        specific_file=(
            Path(specific_file)
            if isinstance(specific_file, str)
            else specific_file
        ),
    )
