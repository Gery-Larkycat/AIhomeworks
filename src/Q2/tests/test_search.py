"""
Tests for Q2 hyperparameter search.
Q2 超参数搜索测试。

Covers: parameter loading, mapping, filtering, SearchConfig defaults.
覆盖：参数加载、映射、过滤、SearchConfig 默认值。
"""

import dataclasses
import json

import pytest

from src.Q2.config import Q2TrainConfig
from src.Q2.search import load_q2_best_params
from utils.config import SearchConfig


# ---------------------------------------------------------------------------
# Search result JSON fixture / 搜索结果 JSON 模板
# ---------------------------------------------------------------------------

def _make_search_json(
    tmp_path,
    params,
    score=0.35,
    filename="2026-05-29_120000_cifar10_hp_search.json",
    extra_params=None,
):
    """
    Helper: write a search result JSON file.
    辅助函数：写入搜索结果 JSON 文件。
    """
    all_params = dict(params)
    if extra_params:
        all_params.update(extra_params)
    results = {
        "search_config": {"strategy": "random", "total_candidates": 1, "cv": 3},
        "best": {
            "params": all_params,
            "mean_test_score": score,
        },
        "all_candidates": [],
    }
    path = tmp_path / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    return path


# ---------------------------------------------------------------------------
# Load and mapping tests / 加载和映射测试
# ---------------------------------------------------------------------------


class TestLoadSearchResults:
    """Test load_q2_best_params loading and mapping."""

    def test_load_nonexistent_returns_none(self, tmp_path):
        """不存在的 specific_file 返回 None。"""
        result = load_q2_best_params(
            specific_file=tmp_path / "nonexistent.json",
        )
        assert result is None

    def test_load_roundtrip(self, tmp_path):
        """写入 JSON 后能正确读回映射后的参数。"""
        path = _make_search_json(tmp_path, {
            "lr": 0.05,
            "optimizer__momentum": 0.92,
            "optimizer__weight_decay": 3e-4,
            "batch_size": 256,
        })
        loaded = load_q2_best_params(specific_file=path)
        assert loaded is not None
        assert loaded["learning_rate"] == 0.05
        assert loaded["momentum"] == 0.92
        assert loaded["weight_decay"] == 3e-4
        assert loaded["batch_size"] == 256

    def test_load_filters_invalid_fields(self, tmp_path):
        """max_epochs (halving resource) 被过滤，不会导致 replace 报错。"""
        config = Q2TrainConfig()
        path = _make_search_json(tmp_path, {
            "lr": 0.05,
            "optimizer__momentum": 0.92,
            "optimizer__weight_decay": 3e-4,
            "batch_size": 256,
        }, extra_params={"max_epochs": 18})
        loaded = load_q2_best_params(specific_file=path)
        assert loaded is not None
        assert "max_epochs" not in loaded
        # 可以安全传给 dataclasses.replace
        dataclasses.replace(config, **loaded)

    def test_load_no_args_returns_none(self, tmp_path, monkeypatch):
        """搜索目录为空时返回 None。"""
        import src.Q2.search as q2search
        # 猴子补丁：让 make_search_dir 返回临时空目录
        monkeypatch.setattr(
            q2search, "make_search_dir", lambda q: tmp_path,
        )
        result = load_q2_best_params()
        assert result is None


# ---------------------------------------------------------------------------
# SearchConfig tests / 搜索配置测试
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
