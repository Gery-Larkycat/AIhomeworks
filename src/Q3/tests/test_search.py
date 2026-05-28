"""
Tests for hyperparameter search via skorch + sklearn.
skorch + sklearn 超参数搜索测试。

Covers: data preparation, skorch net creation, param mapping,
search execution, result logging.
覆盖：数据准备、skorch 网络创建、参数映射、搜索执行、结果记录。
"""

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

from src.Q3.config import SearchConfig, TrainConfig
from src.Q3.search import (
    PARAM_MAP,
    _build_param_distributions,
    _build_param_grid,
    _create_search_net,
    _prepare_search_data,
    load_best_search_params,
)
from Q2.model import ResNet18


# ---------------------------------------------------------------------------
# Data preparation tests / 数据准备测试
# ---------------------------------------------------------------------------


class TestPrepareSearchData:
    """Test _prepare_search_data produces correct arrays."""

    def test_shapes_and_dtypes(self):
        """返回的 numpy 数组 shape 和 dtype 正确。"""
        config = TrainConfig()
        X, y = _prepare_search_data(config)
        assert X.shape == (50000, 3, 32, 32)
        assert X.dtype == np.float32
        assert y.shape == (50000,)
        assert y.dtype == np.int64

    def test_labels_in_valid_range(self):
        """标签在 [0, 99] 范围内（CIFAR-100 有 100 类）。"""
        config = TrainConfig()
        _, y = _prepare_search_data(config)
        assert y.min() >= 0
        assert y.max() <= 99

    def test_normalization_applied(self):
        """数据已归一化（均值接近 0，标准差接近 1）。"""
        config = TrainConfig()
        X, _ = _prepare_search_data(config)
        # Per-channel mean should be near 0
        channel_means = X.mean(axis=(0, 2, 3))
        assert all(abs(m) < 0.1 for m in channel_means)


# ---------------------------------------------------------------------------
# skorch net creation tests / skorch 网络创建测试
# ---------------------------------------------------------------------------


class TestCreateSearchNet:
    """Test _create_search_net produces valid skorch net."""

    def test_returns_neural_net_classifier(self):
        """返回 NeuralNetClassifier 实例。"""
        config = TrainConfig()
        search_cfg = SearchConfig()
        net = _create_search_net(config, search_cfg)
        assert isinstance(net, NeuralNetClassifier)

    def test_default_params(self):
        """默认参数设置正确。"""
        config = TrainConfig()
        search_cfg = SearchConfig()
        net = _create_search_net(config, search_cfg)
        assert net.module == ResNet18
        assert net.criterion == nn.CrossEntropyLoss
        assert net.optimizer == torch.optim.SGD
        assert net.train_split is False
        assert net.verbose == 0

    def test_module_num_classes(self):
        """module__num_classes 正确传递。"""
        config = TrainConfig(num_classes=100)
        search_cfg = SearchConfig()
        net = _create_search_net(config, search_cfg)
        # skorch stores module kwargs
        assert net.module__num_classes == 100


# ---------------------------------------------------------------------------
# Param distribution tests / 参数分布测试
# ---------------------------------------------------------------------------


class TestParamDistributions:
    """Test param space construction."""

    def test_distributions_contain_expected_keys(self):
        """参数分布包含所有搜索维度。"""
        search_cfg = SearchConfig()
        dist = _build_param_distributions(search_cfg)
        assert "lr" in dist
        assert "optimizer__momentum" in dist
        assert "optimizer__weight_decay" in dist
        assert "batch_size" in dist

    def test_grid_contains_expected_keys(self):
        """网格搜索参数包含所有搜索维度。"""
        search_cfg = SearchConfig()
        grid = _build_param_grid(search_cfg)
        assert "lr" in grid
        assert "optimizer__momentum" in grid
        assert "optimizer__weight_decay" in grid
        assert "batch_size" in grid

    def test_grid_lr_values_in_range(self):
        """网格 lr 值在 [1e-4, 1.0] 范围内。"""
        search_cfg = SearchConfig()
        grid = _build_param_grid(search_cfg)
        for lr in grid["lr"]:
            assert 1e-4 <= lr <= 1.0


# ---------------------------------------------------------------------------
# Param mapping tests / 参数映射测试
# ---------------------------------------------------------------------------


class TestParamMap:
    """Test skorch param → TrainConfig field mapping."""

    def test_map_covers_all_search_params(self):
        """PARAM_MAP 覆盖所有搜索参数。"""
        assert "lr" in PARAM_MAP
        assert "optimizer__momentum" in PARAM_MAP
        assert "optimizer__weight_decay" in PARAM_MAP
        assert "batch_size" in PARAM_MAP

    def test_map_values_are_valid_config_fields(self):
        """映射目标是 TrainConfig 的有效字段。"""
        config = TrainConfig()
        for target in PARAM_MAP.values():
            assert hasattr(config, target)


# ---------------------------------------------------------------------------
# Result logging tests / 结果记录测试
# ---------------------------------------------------------------------------


class TestResultLogging:
    """Test JSON result loading."""

    def test_load_nonexistent_returns_none(self, tmp_path):
        """不存在的 specific_file 返回 None。"""
        result = load_best_search_params(
            specific_file=tmp_path / "nonexistent.json",
        )
        assert result is None

    def test_load_roundtrip(self, tmp_path):
        """写入 JSON 后能正确读回映射后的参数。"""
        results = {
            "search_config": {"strategy": "random"},
            "best": {
                "params": {
                    "lr": 0.05,
                    "optimizer__momentum": 0.92,
                    "optimizer__weight_decay": 3e-4,
                    "batch_size": 256,
                },
                "mean_test_score": 0.35,
            },
            "all_candidates": [],
        }
        path = tmp_path / "2026-05-27_143022_cifar100_hp_search.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f)

        loaded = load_best_search_params(specific_file=path)
        assert loaded is not None
        assert loaded["learning_rate"] == 0.05
        assert loaded["momentum"] == 0.92
        assert loaded["weight_decay"] == 3e-4
        assert loaded["batch_size"] == 256

    def test_load_filters_invalid_fields(self, tmp_path):
        """max_epochs (halving resource) 被过滤，不会导致 replace 报错。"""
        config = TrainConfig()
        results = {
            "search_config": {"strategy": "halving-random"},
            "best": {
                "params": {
                    "lr": 0.05,
                    "optimizer__momentum": 0.92,
                    "optimizer__weight_decay": 3e-4,
                    "batch_size": 256,
                    "max_epochs": 18,  # halving resource 参数
                },
                "mean_test_score": 0.35,
            },
            "all_candidates": [],
        }
        path = tmp_path / "2026-05-27_143022_cifar100_hp_search.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f)

        loaded = load_best_search_params(specific_file=path)
        assert loaded is not None
        assert "max_epochs" not in loaded
        # 可以安全传给 dataclasses.replace
        dataclasses.replace(config, **loaded)

    def test_load_specific_file(self, tmp_path):
        """指定 specific_file 时直接加载该文件。"""
        results = {
            "search_config": {"strategy": "random"},
            "best": {
                "params": {
                    "lr": 0.03,
                    "optimizer__momentum": 0.95,
                    "optimizer__weight_decay": 1e-4,
                    "batch_size": 128,
                },
                "mean_test_score": 0.40,
            },
            "all_candidates": [],
        }
        path = tmp_path / "specific_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f)

        loaded = load_best_search_params(specific_file=path)
        assert loaded is not None
        assert loaded["learning_rate"] == 0.03

    def test_load_no_args_returns_none(self):
        """无参数且搜索目录为空时返回 None。"""
        # outputs/Q3/search_results/ 不存在或为空
        result = load_best_search_params()
        assert result is None


# ---------------------------------------------------------------------------
# Search config tests / 搜索配置测试
# ---------------------------------------------------------------------------


class TestSearchConfig:
    """Test SearchConfig dataclass."""

    def test_default_strategy(self):
        """默认策略为 halving-random。"""
        cfg = SearchConfig()
        assert cfg.strategy == "halving-random"

    def test_frozen(self):
        """SearchConfig 不可变。"""
        cfg = SearchConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.strategy = "grid"  # type: ignore[misc]

    def test_custom_strategy(self):
        """可设置策略。"""
        cfg = dataclasses.replace(SearchConfig(), strategy="grid")
        assert cfg.strategy == "grid"

    def test_sensible_defaults(self):
        """默认值合理。"""
        cfg = SearchConfig()
        assert cfg.search_epochs_min == 2
        assert cfg.search_epochs_max == 20
        assert cfg.cv == 3
        assert cfg.num_trials == 50
        assert cfg.scoring == "accuracy"
