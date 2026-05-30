"""
Tests for Q1 hyperparameter search.
Q1 超参数搜索测试。
"""

import dataclasses
import json

import pytest

from src.Q1.config import Q1TrainConfig
from src.Q1.search import load_q1_best_params
from utils.config import SearchConfig


def _make_search_json(
    tmp_path, params, score=0.35,
    filename="2026-05-30_120000_vgg16_cifar10_hp_search.json",
    extra_params=None,
):
    all_params = dict(params)
    if extra_params:
        all_params.update(extra_params)
    results = {
        "search_config": {"strategy": "random", "total_candidates": 1, "cv": 3},
        "best": {"params": all_params, "mean_test_score": score},
        "all_candidates": [],
    }
    path = tmp_path / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    return path


class TestLoadSearchResults:

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_q1_best_params(specific_file=tmp_path / "no.json")
        assert result is None

    def test_load_roundtrip(self, tmp_path):
        path = _make_search_json(tmp_path, {
            "lr": 0.05,
            "optimizer__momentum": 0.92,
            "optimizer__weight_decay": 3e-4,
            "batch_size": 256,
        })
        loaded = load_q1_best_params(specific_file=path)
        assert loaded is not None
        assert loaded["learning_rate"] == 0.05
        assert loaded["momentum"] == 0.92

    def test_load_filters_invalid_fields(self, tmp_path):
        config = Q1TrainConfig()
        path = _make_search_json(tmp_path, {
            "lr": 0.05, "optimizer__momentum": 0.92,
            "optimizer__weight_decay": 3e-4, "batch_size": 256,
        }, extra_params={"max_epochs": 18})
        loaded = load_q1_best_params(specific_file=path)
        assert loaded is not None
        assert "max_epochs" not in loaded
        dataclasses.replace(config, **loaded)

    def test_load_no_args_returns_none(self, tmp_path, monkeypatch):
        import src.Q1.search as q1search
        monkeypatch.setattr(
            q1search, "make_search_dir", lambda q: tmp_path,
        )
        result = load_q1_best_params()
        assert result is None


class TestSearchConfig:

    def test_default_strategy(self):
        assert SearchConfig().strategy == "halving-random"

    def test_frozen(self):
        cfg = SearchConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.strategy = "grid"  # type: ignore[misc]

    def test_sensible_defaults(self):
        cfg = SearchConfig()
        assert cfg.cv == 3
        assert cfg.num_trials == 50
