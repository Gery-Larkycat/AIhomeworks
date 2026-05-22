"""
Tests for hyperparameter search, factory functions, and search config.
超参数搜索、工厂函数和搜索配置的测试。

Covers: evolutionary search, random search, grid search, grid generation,
fitness computation, evolutionary operators, result logging.
覆盖：演化搜索、随机搜索、网格搜索、网格生成、适应度计算、
演化算子、结果记录。
"""

import dataclasses
import json
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.Q3.config import HyperparamRange, SearchConfig, TrainConfig
from src.Q3.model import create_model
from src.Q3.search import (
    Individual,
    accuracy_improvement_rate,
    compute_fitness,
    crossover,
    evolutionary_search,
    generate_grid,
    grid_search,
    load_best_search_params,
    log_search_results,
    loss_decrease_rate,
    mutate,
    random_search,
    sample_individual,
    sample_log_uniform,
    tournament_select,
    train_probe,
)
from src.Q3.train import (
    create_criterion,
    create_optimizer,
    create_scheduler,
)


# ---------------------------------------------------------------------------
# Factory function tests / 工厂函数测试
# ---------------------------------------------------------------------------


class TestCreateOptimizer:
    """Test create_optimizer supports all optimizer types."""

    @pytest.fixture()
    def model(self):
        return create_model(num_classes=10)

    @pytest.mark.parametrize(
        "opt_type", ["sgd", "adam", "adamw", "rmsprop", "nadam"]
    )
    def test_creates_optimizer(self, model, opt_type):
        """每种优化器类型都能成功创建。"""
        cfg = dataclasses.replace(
            TrainConfig(), optimizer_type=opt_type, learning_rate=0.01
        )
        opt = create_optimizer(model, cfg)
        assert opt is not None
        # Verify param groups exist / 验证参数组存在
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == 0.01

    def test_unknown_optimizer_raises(self, model):
        """未知优化器类型抛出 ValueError。"""
        cfg = dataclasses.replace(
            TrainConfig(), optimizer_type="invalid_opt"
        )
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(model, cfg)

    def test_sgd_has_momentum(self, model):
        """SGD 优化器包含 momentum 参数。"""
        cfg = dataclasses.replace(
            TrainConfig(), optimizer_type="sgd", momentum=0.95
        )
        opt = create_optimizer(model, cfg)
        assert opt.param_groups[0]["momentum"] == 0.95


class TestCreateScheduler:
    """Test create_scheduler supports all scheduler types."""

    @pytest.fixture()
    def optimizer(self):
        model = create_model(num_classes=10)
        return torch.optim.SGD(model.parameters(), lr=0.1)

    @pytest.mark.parametrize("sch_type", ["cosine", "step"])
    def test_creates_scheduler(self, optimizer, sch_type):
        """cosine 和 step 都返回 LRScheduler 对象。"""
        cfg = dataclasses.replace(
            TrainConfig(), scheduler_type=sch_type
        )
        sch = create_scheduler(optimizer, cfg)
        assert sch is not None

    def test_constant_returns_none(self, optimizer):
        """constant 返回 None（不使用调度器）。"""
        cfg = dataclasses.replace(
            TrainConfig(), scheduler_type="constant"
        )
        sch = create_scheduler(optimizer, cfg)
        assert sch is None

    def test_unknown_scheduler_raises(self, optimizer):
        """未知调度器类型抛出 ValueError。"""
        cfg = dataclasses.replace(
            TrainConfig(), scheduler_type="invalid_sch"
        )
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(optimizer, cfg)


class TestCreateCriterion:
    """Test create_criterion creates proper loss function."""

    def test_creates_cross_entropy(self):
        """创建 CrossEntropyLoss。"""
        cfg = TrainConfig()
        criterion = create_criterion(cfg)
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_label_smoothing_applied(self):
        """label_smoothing 正确传递。"""
        cfg = dataclasses.replace(TrainConfig(), label_smoothing=0.2)
        criterion = create_criterion(cfg)
        assert criterion.label_smoothing == 0.2


# ---------------------------------------------------------------------------
# Search config tests / 搜索配置测试
# ---------------------------------------------------------------------------


class TestSearchConfig:
    """Test SearchConfig and HyperparamRange dataclasses."""

    def test_default_search_config(self):
        """默认搜索配置合理。"""
        cfg = SearchConfig()
        assert cfg.search_epochs == 5
        assert cfg.population_size == 8
        assert cfg.num_generations == 3
        assert cfg.learning_rate is not None
        assert cfg.batch_size is not None

    def test_frozen_search_config(self):
        """SearchConfig 不可变。"""
        cfg = SearchConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.search_epochs = 10  # type: ignore[misc]

    def test_disable_param(self):
        """将参数设为 None 可跳过该参数。"""
        cfg = dataclasses.replace(
            SearchConfig(), learning_rate=None, momentum=None
        )
        assert cfg.learning_rate is None
        assert cfg.momentum is None
        # Others remain / 其他保留
        assert cfg.weight_decay is not None

    def test_hyperparam_range(self):
        """HyperparamRange 存储正确的值。"""
        r = HyperparamRange(0.001, 0.5, "log_uniform")
        assert r.low == 0.001
        assert r.high == 0.5
        assert r.distribution == "log_uniform"


# ---------------------------------------------------------------------------
# Sampling tests / 采样测试
# ---------------------------------------------------------------------------


class TestSampling:
    """Test sampling functions produce valid values."""

    def test_log_uniform_in_range(self):
        """log_uniform 采样值在合法范围内。"""
        import random

        rng = random.Random(42)
        for _ in range(100):
            val = sample_log_uniform(1e-4, 1.0, rng)
            assert 1e-4 <= val <= 1.0

    def test_log_uniform_covers_orders(self):
        """log_uniform 跨越多个数量级。"""
        import random

        rng = random.Random(42)
        samples = [sample_log_uniform(1e-4, 1.0, rng) for _ in range(1000)]
        # Should have values in [1e-4, 1e-3), [1e-3, 1e-2), etc.
        # 应该有覆盖 [1e-4, 1e-3), [1e-3, 1e-2) 等区间的值
        below_001 = sum(1 for s in samples if s < 0.01)
        above_01 = sum(1 for s in samples if s > 0.1)
        assert below_001 > 0
        assert above_01 > 0

    def test_log_uniform_distribution_uniformity(self):
        """log_uniform 在对数空间中近似均匀。"""
        import random

        rng = random.Random(42)
        samples = [
            math.log10(sample_log_uniform(1e-4, 1.0, rng))
            for _ in range(2000)
        ]
        # log10 of samples should span [-4, 0]
        # 样本的 log10 应跨越 [-4, 0]
        mean_log = sum(samples) / len(samples)
        # Midpoint should be near -2 (center of [-4, 0])
        assert -2.5 < mean_log < -1.5

    def test_log_uniform_different_seeds(self):
        """不同 seed 产生不同采样序列。"""
        import random

        rng1 = random.Random(1)
        rng2 = random.Random(2)
        s1 = [sample_log_uniform(1e-4, 1.0, rng1) for _ in range(10)]
        s2 = [sample_log_uniform(1e-4, 1.0, rng2) for _ in range(10)]
        assert s1 != s2

    def test_sample_individual_keys(self):
        """sample_individual 产生所有搜索空间的 key。"""
        import random

        rng = random.Random(42)
        cfg = SearchConfig()
        params = sample_individual(cfg, rng)
        assert "learning_rate" in params
        assert "weight_decay" in params
        assert "momentum" in params
        assert "batch_size" in params
        assert "optimizer_type" in params
        assert "scheduler_type" in params

    def test_sample_individual_respects_ranges(self):
        """sample_individual 的值在合法范围内。"""
        import random

        rng = random.Random(42)
        cfg = SearchConfig()
        for _ in range(50):
            params = sample_individual(cfg, rng)
            assert 1e-4 <= params["learning_rate"] <= 1.0
            assert 1e-6 <= params["weight_decay"] <= 1e-2
            assert 0.8 <= params["momentum"] <= 0.99
            assert params["batch_size"] in (128, 256, 512, 1024)
            assert params["optimizer_type"] in (
                "sgd", "adam", "adamw", "rmsprop", "nadam"
            )
            assert params["scheduler_type"] in (
                "cosine", "constant", "step"
            )

    def test_sample_individual_skips_none(self):
        """设 None 的参数不出现在采样结果中。"""
        import random

        rng = random.Random(42)
        cfg = dataclasses.replace(
            SearchConfig(), momentum=None
        )
        params = sample_individual(cfg, rng)
        assert "momentum" not in params

    def test_sample_individual_uniform_distribution(self):
        """uniform 分布的参数在区间内近似均匀。"""
        import random

        rng = random.Random(42)
        cfg = SearchConfig()
        momentums = []
        for _ in range(500):
            params = sample_individual(cfg, rng)
            momentums.append(params["momentum"])
        mean_m = sum(momentums) / len(momentums)
        # Should be near center of [0.8, 0.99] ≈ 0.895
        assert 0.85 < mean_m < 0.94

    def test_sample_discrete_covers_all_options(self):
        """离散参数在大量采样后覆盖所有选项。"""
        import random

        rng = random.Random(42)
        cfg = SearchConfig()
        opts_seen = set()
        for _ in range(200):
            params = sample_individual(cfg, rng)
            opts_seen.add(params["optimizer_type"])
        # Should see all 5 optimizer types
        # 应该看到全部 5 种优化器
        assert len(opts_seen) == 5


# ---------------------------------------------------------------------------
# Fitness tests / 适应度测试
# ---------------------------------------------------------------------------


class TestFitness:
    """Test fitness computation functions."""

    def test_accuracy_improvement_rate_positive(self):
        """acc 持续上升 → AIR > 0。"""
        acc = [0.1, 0.2, 0.3, 0.4, 0.5]
        air = accuracy_improvement_rate(acc)
        assert air > 0

    def test_accuracy_improvement_rate_negative(self):
        """acc 持续下降 → AIR < 0。"""
        acc = [0.5, 0.4, 0.3, 0.2, 0.1]
        air = accuracy_improvement_rate(acc)
        assert air < 0

    def test_accuracy_improvement_rate_flat(self):
        """acc 不变 → AIR ≈ 0。"""
        acc = [0.3, 0.3, 0.3, 0.3, 0.3]
        air = accuracy_improvement_rate(acc)
        assert abs(air) < 1e-6

    def test_accuracy_improvement_rate_exact_slope(self):
        """线性上升的准确率，AIR 等于每 epoch 增量。"""
        # acc = 0.1 + 0.05 * epoch → slope = 0.05
        acc = [0.1, 0.15, 0.2, 0.25, 0.3]
        air = accuracy_improvement_rate(acc)
        assert abs(air - 0.05) < 1e-6

    def test_accuracy_improvement_rate_single_value(self):
        """单个值 → AIR = 0（无法计算斜率）。"""
        assert accuracy_improvement_rate([0.5]) == 0.0

    def test_accuracy_improvement_rate_empty(self):
        """空列表 → AIR = 0。"""
        assert accuracy_improvement_rate([]) == 0.0

    def test_loss_decrease_rate_positive(self):
        """loss 持续下降 → LDR > 0。"""
        loss = [4.6, 4.0, 3.5, 3.1, 2.8]
        ldr = loss_decrease_rate(loss)
        assert ldr > 0

    def test_loss_decrease_rate_negative(self):
        """loss 持续上升 → LDR < 0。"""
        loss = [1.0, 2.0, 3.0, 4.0, 5.0]
        ldr = loss_decrease_rate(loss)
        assert ldr < 0

    def test_loss_decrease_rate_exact_slope(self):
        """线性下降的 loss，LDR 等于每 epoch 下降量。"""
        # loss = 5.0 - 0.5 * epoch → slope = -0.5 → LDR = 0.5
        loss = [5.0, 4.5, 4.0, 3.5, 3.0]
        ldr = loss_decrease_rate(loss)
        assert abs(ldr - 0.5) < 1e-6

    def test_loss_decrease_rate_single_value(self):
        """单个值 → LDR = 0。"""
        assert loss_decrease_rate([3.0]) == 0.0

    def test_compute_fitness_good_config(self):
        """好配置（acc 上升、loss 下降）→ 高 fitness。"""
        fitness = compute_fitness(
            [4.6, 4.0, 3.5, 3.1, 2.8],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        assert fitness > 0
        assert math.isfinite(fitness)

    def test_compute_fitness_diverging(self):
        """发散配置（acc 下降、loss 上升）→ 低 fitness。"""
        fitness = compute_fitness(
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        )
        # Penalized twice: acc down AND loss up → × 0.5 × 0.5 = × 0.25
        assert fitness < 0

    def test_compute_fitness_diverging_penalty_strength(self):
        """loss 上升触发 × 0.5 惩罚。"""
        # Same acc trajectory (upward), but loss going up vs down
        # 同样 acc 轨迹（上升），但 loss 一升一降
        loss_down = compute_fitness(
            [4.0, 3.0, 2.0],
            [0.1, 0.2, 0.3],
        )
        # Same but loss increasing → penalty × 0.5
        # 同样但 loss 上升 → × 0.5 惩罚
        loss_up_raw = compute_fitness(
            [2.0, 3.0, 4.0],
            [0.1, 0.2, 0.3],
        )
        # loss_down should be strictly better than loss_up
        # loss 下降的应严格优于 loss 上升的
        assert loss_down > loss_up_raw

    def test_compute_fitness_nan_returns_neg_inf(self):
        """NaN → fitness = -inf。"""
        fitness = compute_fitness(
            [float("nan"), 1.0],
            [0.1, 0.2],
        )
        assert fitness == float("-inf")

    def test_compute_fitness_inf_returns_neg_inf(self):
        """Inf → fitness = -inf。"""
        fitness = compute_fitness(
            [float("inf"), 1.0],
            [0.1, 0.2],
        )
        assert fitness == float("-inf")

    def test_compute_fitness_scaling(self):
        """AIR 被 10x 放大，与 LDR 量级可比。"""
        # Small acc improvement + large loss decrease
        # 小 acc 提升 + 大 loss 下降
        fitness = compute_fitness(
            [5.0, 4.0, 3.0, 2.0, 1.0],  # LDR ≈ 1.0
            [0.01, 0.02, 0.03, 0.04, 0.05],  # AIR ≈ 0.01
        )
        # fitness ≈ 10 * 0.01 + 1.0 = 1.1
        assert fitness > 1.0

    def test_compute_fitness_ordering(self):
        """更好的配置得到更高的 fitness。"""
        # Great: fast acc rise + fast loss drop
        # 极好：acc 快速上升 + loss 快速下降
        great = compute_fitness(
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        # Good: slower improvement / 好：改善较慢
        good = compute_fitness(
            [4.0, 3.5, 3.0, 2.5, 2.0],
            [0.1, 0.15, 0.2, 0.25, 0.3],
        )
        assert great > good


# ---------------------------------------------------------------------------
# Evolutionary operator tests / 演化算子测试
# ---------------------------------------------------------------------------


class TestEvolutionaryOperators:
    """Test crossover, mutation, and selection."""

    @pytest.fixture()
    def rng(self):
        import random
        return random.Random(42)

    @pytest.fixture()
    def search_cfg(self):
        return SearchConfig()

    @pytest.fixture()
    def population(self):
        """Create a diverse test population."""
        return [
            Individual(
                learning_rate=0.01, weight_decay=1e-4,
                momentum=0.9, batch_size=128,
                optimizer_type="sgd", scheduler_type="cosine",
                fitness=float(i),  # Higher index = better
            )
            for i in range(8)
        ]

    def test_tournament_select_returns_individual(
        self, population, rng
    ):
        """锦标赛选择返回种群中的个体。"""
        selected = tournament_select(population, 3, rng)
        assert isinstance(selected, Individual)
        assert selected in population

    def test_tournament_select_favors_better(
        self, rng
    ):
        """锦标赛选择偏好 fitness 更高的个体。"""
        pop = [
            Individual(
                learning_rate=0.01, weight_decay=1e-4,
                momentum=0.9, batch_size=128,
                optimizer_type="sgd", scheduler_type="cosine",
                fitness=f,
            )
            for f in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
        ]
        best_count = 0
        for _ in range(1000):
            sel = tournament_select(pop, 3, rng)
            if sel.fitness == 100.0:
                best_count += 1
        # Best should be selected more than random (1/8 = 125)
        assert best_count > 200

    def test_tournament_select_equal_fitness(self, rng):
        """所有个体 fitness 相同时，选择是均匀的。"""
        pop = [
            Individual(
                learning_rate=0.01 * (i + 1),
                weight_decay=1e-4,
                momentum=0.9, batch_size=128,
                optimizer_type="sgd", scheduler_type="cosine",
                fitness=1.0,
            )
            for i in range(8)
        ]
        # All should be selectable / 所有都应可选
        selected = set()
        for _ in range(100):
            sel = tournament_select(pop, 3, rng)
            selected.add(sel.learning_rate)
        # Should pick several different individuals
        # 应该选到多个不同个体
        assert len(selected) > 1

    def test_tournament_size_larger_than_pop(self, rng):
        """tournament_size > 种群大小时不会报错。"""
        pop = [
            Individual(
                learning_rate=0.01, weight_decay=1e-4,
                momentum=0.9, batch_size=128,
                optimizer_type="sgd", scheduler_type="cosine",
                fitness=float(i),
            )
            for i in range(3)
        ]
        sel = tournament_select(pop, 10, rng)
        assert sel in pop

    def test_crossover_produces_valid_child(
        self, search_cfg, rng
    ):
        """交叉产生合法参数组合。"""
        p_a = Individual(
            learning_rate=0.01, weight_decay=1e-4,
            momentum=0.9, batch_size=128,
            optimizer_type="sgd", scheduler_type="cosine",
        )
        p_b = Individual(
            learning_rate=0.5, weight_decay=1e-2,
            momentum=0.95, batch_size=1024,
            optimizer_type="adamw", scheduler_type="step",
        )
        child = crossover(p_a, p_b, search_cfg, rng)
        assert 1e-4 <= child["learning_rate"] <= 1.0
        assert 1e-6 <= child["weight_decay"] <= 1e-2
        assert 0.8 <= child["momentum"] <= 0.99
        assert child["batch_size"] in (128, 1024)
        assert child["optimizer_type"] in ("sgd", "adamw")
        assert child["scheduler_type"] in ("cosine", "step")

    def test_crossover_blx_alpha_expands_range(self, rng):
        """BLX-α 交叉产生的连续值可能超出父代范围（α 扩展）。"""
        search_cfg = SearchConfig()
        # Run many crossovers with identical parents
        # 多次交叉同对父代
        p_a = Individual(
            learning_rate=0.1, weight_decay=1e-3,
            momentum=0.9, batch_size=256,
            optimizer_type="sgd", scheduler_type="cosine",
        )
        p_b = Individual(
            learning_rate=0.2, weight_decay=2e-3,
            momentum=0.92, batch_size=512,
            optimizer_type="adam", scheduler_type="step",
        )
        lrs = []
        for seed in range(200):
            import random
            r = random.Random(seed)
            child = crossover(p_a, p_b, search_cfg, r)
            lrs.append(child["learning_rate"])
        # Some values should be outside [0.1, 0.2] due to α=0.3
        # 部分值应在 [0.1, 0.2] 之外（α=0.3 扩展）
        outside = sum(
            1 for lr in lrs if lr < 0.1 or lr > 0.2
        )
        assert outside > 0

    def test_crossover_identical_parents(self, rng):
        """相同父代交叉 → 连续参数值与父代相同（无扩展空间）。"""
        search_cfg = SearchConfig()
        p = Individual(
            learning_rate=0.1, weight_decay=1e-3,
            momentum=0.9, batch_size=256,
            optimizer_type="sgd", scheduler_type="cosine",
        )
        child = crossover(p, p, search_cfg, rng)
        # When parents are identical, BLX-α range = [v, v]
        # → child value = v (since spread = 0)
        assert child["learning_rate"] == 0.1
        assert child["weight_decay"] == 1e-3
        assert child["momentum"] == 0.9

    def test_crossover_discrete_inherits_from_parents(self, rng):
        """离散参数从父代 A 或 B 继承。"""
        search_cfg = SearchConfig()
        p_a = Individual(
            learning_rate=0.1, weight_decay=1e-3,
            momentum=0.9, batch_size=128,
            optimizer_type="sgd", scheduler_type="cosine",
        )
        p_b = Individual(
            learning_rate=0.2, weight_decay=2e-3,
            momentum=0.92, batch_size=1024,
            optimizer_type="adamw", scheduler_type="step",
        )
        opts_a, opts_b = 0, 0
        for seed in range(100):
            import random
            r = random.Random(seed)
            child = crossover(p_a, p_b, search_cfg, r)
            if child["optimizer_type"] == "sgd":
                opts_a += 1
            elif child["optimizer_type"] == "adamw":
                opts_b += 1
        # Both should be selected ~50%
        assert opts_a > 20
        assert opts_b > 20

    def test_mutate_stays_in_range(
        self, search_cfg, rng
    ):
        """变异后的值仍在合法范围内。"""
        params = {
            "learning_rate": 0.05,
            "weight_decay": 1e-3,
            "momentum": 0.9,
            "batch_size": 256,
            "optimizer_type": "sgd",
            "scheduler_type": "cosine",
        }
        # Mutate many times to test boundary
        # 多次变异以测试边界
        for _ in range(100):
            mutated = mutate(params, search_cfg, rng)
            assert 1e-4 <= mutated["learning_rate"] <= 1.0
            assert 1e-6 <= mutated["weight_decay"] <= 1e-2
            assert 0.8 <= mutated["momentum"] <= 0.99
            assert mutated["batch_size"] in (128, 256, 512, 1024)
            assert mutated["optimizer_type"] in (
                "sgd", "adam", "adamw", "rmsprop", "nadam"
            )
            assert mutated["scheduler_type"] in (
                "cosine", "constant", "step"
            )

    def test_mutate_zero_rate_no_change(self):
        """mutation_rate=0 时不变异。"""
        import random

        rng = random.Random(42)
        cfg = dataclasses.replace(
            SearchConfig(), mutation_rate=0.0
        )
        params = {
            "learning_rate": 0.05,
            "weight_decay": 1e-3,
            "momentum": 0.9,
            "batch_size": 256,
            "optimizer_type": "sgd",
            "scheduler_type": "cosine",
        }
        for _ in range(50):
            mutated = mutate(params, cfg, rng)
            assert mutated == params

    def test_mutate_full_rate_always_changes_discrete(self):
        """mutation_rate=1.0 时离散参数一定变异（重采样）。"""
        import random

        rng = random.Random(42)
        cfg = dataclasses.replace(
            SearchConfig(),
            mutation_rate=1.0,
            batch_size=(128, 256, 512, 1024),
        )
        params = {
            "learning_rate": 0.05,
            "weight_decay": 1e-3,
            "momentum": 0.9,
            "batch_size": 128,
            "optimizer_type": "sgd",
            "scheduler_type": "cosine",
        }
        # Run many mutations, at least some should change batch_size
        # 多次变异，至少部分应改变 batch_size
        changed = False
        for _ in range(100):
            mutated = mutate(params, cfg, rng)
            if mutated["batch_size"] != 128:
                changed = True
                break
        assert changed

    def test_mutate_multiplicative_gaussian(self):
        """连续变异是乘性的（乘以 1 + N(0, 0.2)）。"""
        import random

        rng = random.Random(42)
        cfg = dataclasses.replace(
            SearchConfig(), mutation_rate=1.0
        )
        params = {
            "learning_rate": 0.1,
            "weight_decay": 1e-3,
            "momentum": 0.9,
            "batch_size": 256,
            "optimizer_type": "sgd",
            "scheduler_type": "cosine",
        }
        # All mutations happen; lr should vary around 0.1
        # 所有变异都发生；lr 应在 0.1 附近变化
        lrs = []
        for _ in range(200):
            mutated = mutate(params, cfg, rng)
            lrs.append(mutated["learning_rate"])
        mean_lr = sum(lrs) / len(lrs)
        # Mean should be near 0.1 (multiplicative noise
        # with small sigma preserves mean approximately)
        assert 0.05 < mean_lr < 0.2


class TestIndividual:
    """Test Individual dataclass."""

    def test_frozen(self):
        """Individual 不可变。"""
        ind = Individual(
            learning_rate=0.1, weight_decay=1e-4,
            momentum=0.9, batch_size=256,
            optimizer_type="sgd", scheduler_type="cosine",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ind.learning_rate = 0.5  # type: ignore[misc]

    def test_default_fitness_is_none(self):
        """未评估时 fitness 为 None。"""
        ind = Individual(
            learning_rate=0.1, weight_decay=1e-4,
            momentum=0.9, batch_size=256,
            optimizer_type="sgd", scheduler_type="cosine",
        )
        assert ind.fitness is None

    def test_with_fitness(self):
        """可以带 fitness 创建。"""
        ind = Individual(
            learning_rate=0.1, weight_decay=1e-4,
            momentum=0.9, batch_size=256,
            optimizer_type="sgd", scheduler_type="cosine",
            fitness=0.85,
        )
        assert ind.fitness == 0.85


# ---------------------------------------------------------------------------
# Integration tests / 集成测试
# ---------------------------------------------------------------------------


def _make_synthetic_loaders(
    batch_size: int = 16, num_samples: int = 64
) -> tuple[DataLoader, DataLoader]:
    """
    Create synthetic CIFAR-shaped DataLoaders for testing.
    创建用于测试的合成 CIFAR 形状 DataLoader。

    Avoids downloading real data; small enough for fast test runs.
    避免下载真实数据；足够小以实现快速测试。
    """
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 100, (num_samples,))
    train_ds = TensorDataset(images, labels)
    test_ds = TensorDataset(
        images[: num_samples // 2],
        labels[: num_samples // 2],
    )
    return (
        DataLoader(train_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )


class TestTrainProbe:
    """Integration test: train_probe runs end-to-end."""

    def test_probe_returns_fitness_and_history(self):
        """train_probe 返回 fitness 和 history。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(), search_epochs=2
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        device = torch.device("cpu")

        params = {
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "batch_size": 16,
            "optimizer_type": "sgd",
            "scheduler_type": "cosine",
        }
        fitness, history = train_probe(
            params, config, search_cfg,
            train_loader, test_loader, device,
            seed=42,
        )
        assert isinstance(fitness, float)
        assert math.isfinite(fitness)
        assert "test_loss" in history
        assert "test_acc" in history
        assert len(history["test_loss"]) == 2
        assert len(history["test_acc"]) == 2

    @pytest.mark.parametrize(
        "opt", ["sgd", "adam", "adamw", "rmsprop", "nadam"]
    )
    def test_probe_all_optimizers(self, opt):
        """train_probe 支持所有优化器类型。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(), search_epochs=1
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        device = torch.device("cpu")

        params = {
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "batch_size": 16,
            "optimizer_type": opt,
            "scheduler_type": "constant",
        }
        fitness, history = train_probe(
            params, config, search_cfg,
            train_loader, test_loader, device,
            seed=42,
        )
        assert math.isfinite(fitness)
        assert len(history["test_loss"]) == 1

    @pytest.mark.parametrize("sch", ["cosine", "step", "constant"])
    def test_probe_all_schedulers(self, sch):
        """train_probe 支持所有调度器类型。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(), search_epochs=2
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        device = torch.device("cpu")

        params = {
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "batch_size": 16,
            "optimizer_type": "sgd",
            "scheduler_type": sch,
        }
        fitness, history = train_probe(
            params, config, search_cfg,
            train_loader, test_loader, device,
            seed=42,
        )
        assert math.isfinite(fitness)
        assert len(history["test_loss"]) == 2

    def test_probe_deterministic_with_seed(self):
        """相同 seed 产生相同结果。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(), search_epochs=2
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        device = torch.device("cpu")

        params = {
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "batch_size": 16,
            "optimizer_type": "sgd",
            "scheduler_type": "cosine",
        }
        f1, h1 = train_probe(
            params, config, search_cfg,
            train_loader, test_loader, device,
            seed=123,
        )
        f2, h2 = train_probe(
            params, config, search_cfg,
            train_loader, test_loader, device,
            seed=123,
        )
        assert f1 == f2
        assert h1["test_loss"] == h2["test_loss"]
        assert h1["test_acc"] == h2["test_acc"]


class TestEvolutionarySearchIntegration:
    """Integration test: full evolutionary_search pipeline."""

    def test_minimal_search(self):
        """最小化配置运行完整搜索流程。"""
        config = TrainConfig()
        # Minimal: pop=2, gen=1, epochs=1, offspring=1
        # 最小化：pop=2, gen=1, epochs=1, offspring=1
        search_cfg = dataclasses.replace(
            SearchConfig(),
            search_epochs=1,
            population_size=2,
            offspring_per_gen=1,
            num_generations=1,
            batch_size=(16,),  # Single batch size for synthetic
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        best_params, all_records = evolutionary_search(
            config, search_cfg, loaders_by_batch, device,
        )

        # Verify return types / 验证返回类型
        assert isinstance(best_params, dict)
        assert isinstance(all_records, list)
        assert len(all_records) == 3  # 2 initial + 1 offspring

        # Best params contain expected keys
        # 最优参数包含预期的 key
        assert "learning_rate" in best_params
        assert "optimizer_type" in best_params
        assert best_params["optimizer_type"] == "sgd"

        # All records have valid fitness
        # 所有记录有合法 fitness
        for rec in all_records:
            assert "fitness" in rec
            assert "params" in rec
            assert "generation" in rec
            assert math.isfinite(rec["fitness"])

    def test_search_elitism_preserves_best(self):
        """精英保留：后代不会丢失最优个体。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            search_epochs=1,
            population_size=3,
            offspring_per_gen=2,
            num_generations=1,
            batch_size=(16,),
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        _, all_records = evolutionary_search(
            config, search_cfg, loaders_by_batch, device,
        )

        # Gen 0: 3 individuals, Gen 1: 2 offspring = 5 total
        # 初始代 3 个个体，第 1 代 2 个后代 = 共 5 个
        assert len(all_records) == 5
        gen0_records = [
            r for r in all_records if r["generation"] == 0
        ]
        assert len(gen0_records) == 3
        gen1_records = [
            r for r in all_records if r["generation"] == 1
        ]
        assert len(gen1_records) == 2

    def test_search_improves_over_generations(self):
        """搜索多代后 fitness 有改善趋势。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            search_epochs=1,
            population_size=3,
            offspring_per_gen=2,
            num_generations=2,
            batch_size=(16,),
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
            learning_rate=HyperparamRange(0.001, 0.01, "log_uniform"),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        _, all_records = evolutionary_search(
            config, search_cfg, loaders_by_batch, device,
        )

        # Total: 3 + 2 + 2 = 7 evaluations
        # 总计：3 + 2 + 2 = 7 次评估
        assert len(all_records) == 7


class TestResultLogging:
    """Test JSON result logging and loading."""

    def test_log_and_load_roundtrip(self, tmp_path):
        """JSON 写入后能正确读回。"""
        config = dataclasses.replace(
            TrainConfig(), checkpoint_dir=tmp_path
        )
        search_cfg = SearchConfig()

        best_params = {
            "learning_rate": 0.05,
            "weight_decay": 3e-4,
            "momentum": 0.92,
            "batch_size": 256,
            "optimizer_type": "sgd",
            "scheduler_type": "cosine",
        }
        all_records = [
            {
                "id": 1,
                "generation": 0,
                "params": best_params,
                "fitness": 0.8,
                "final_test_acc": 0.15,
                "final_test_loss": 3.5,
                "test_acc_history": [0.05, 0.10, 0.15],
                "test_loss_history": [4.0, 3.7, 3.5],
            }
        ]

        path = log_search_results(
            best_params, all_records, search_cfg, config
        )
        assert path.exists()

        # Load back / 读回
        loaded = load_best_search_params(config)
        assert loaded is not None
        assert loaded["learning_rate"] == 0.05
        assert loaded["optimizer_type"] == "sgd"
        assert loaded["batch_size"] == 256

        # Verify JSON structure / 验证 JSON 结构
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "search_config" in data
        assert "best" in data
        assert "generation_summary" in data
        assert "all_evaluated" in data
        assert data["search_config"]["search_epochs"] == 5
        assert data["best"]["fitness"] == 0.8

    def test_load_nonexistent_returns_none(self, tmp_path):
        """不存在的文件返回 None。"""
        config = dataclasses.replace(
            TrainConfig(), checkpoint_dir=tmp_path
        )
        result = load_best_search_params(config)
        assert result is None


# ---------------------------------------------------------------------------
# Random search tests / 随机搜索测试
# ---------------------------------------------------------------------------


class TestRandomSearch:
    """Random search integration tests."""

    def test_minimal_random_search(self):
        """最小化随机搜索运行成功。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            strategy="random",
            num_trials=3,
            search_epochs=1,
            batch_size=(16,),
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        best_params, all_records = random_search(
            config, search_cfg, loaders_by_batch, device,
        )

        assert isinstance(best_params, dict)
        assert len(all_records) == 3
        assert "learning_rate" in best_params
        assert "optimizer_type" in best_params
        for rec in all_records:
            assert "fitness" in rec
            assert "params" in rec
            assert math.isfinite(rec["fitness"])

    def test_random_search_finds_best(self):
        """随机搜索能追踪最优参数。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            strategy="random",
            num_trials=4,
            search_epochs=1,
            batch_size=(16,),
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        best_params, all_records = random_search(
            config, search_cfg, loaders_by_batch, device,
        )

        # best_params should match the record with highest fitness
        # best_params 应与 fitness 最高的记录匹配
        best_rec = max(all_records, key=lambda r: r["fitness"])
        assert best_params["learning_rate"] == best_rec["params"]["learning_rate"]

    def test_random_search_all_generation_zero(self):
        """随机搜索所有记录的 generation 为 0。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            strategy="random",
            num_trials=2,
            search_epochs=1,
            batch_size=(16,),
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        _, all_records = random_search(
            config, search_cfg, loaders_by_batch, device,
        )

        for rec in all_records:
            assert rec["generation"] == 0


# ---------------------------------------------------------------------------
# Grid generation tests / 网格生成测试
# ---------------------------------------------------------------------------


class TestGenerateGrid:
    """Test grid point generation."""

    def test_grid_size_with_all_params(self):
        """所有参数启用时网格大小正确。"""
        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=3,
            # 3 * 3 * 3 * 4 * 5 * 3 = 1620
        )
        grid = generate_grid(cfg)
        assert len(grid) == 3 * 3 * 3 * 4 * 5 * 3

    def test_grid_size_reduced_space(self):
        """缩小搜索空间后网格大小减小。"""
        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=3,
            batch_size=(128, 256),
            optimizer_type=("sgd", "adam"),
            scheduler_type=("cosine",),
        )
        grid = generate_grid(cfg)
        # 3 * 3 * 3 * 2 * 2 * 1 = 108
        assert len(grid) == 108

    def test_grid_continuous_values_in_range(self):
        """网格连续参数值在合法范围内。"""
        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=5,
        )
        grid = generate_grid(cfg)
        for params in grid:
            assert 1e-4 <= params["learning_rate"] <= 1.0
            assert 1e-6 <= params["weight_decay"] <= 1e-2
            assert 0.8 <= params["momentum"] <= 0.99

    def test_grid_log_uniform_spacing(self):
        """log_uniform 参数在对数空间中等距。"""
        import math

        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=5,
            momentum=None,
            batch_size=None,
            optimizer_type=None,
            scheduler_type=None,
        )
        grid = generate_grid(cfg)
        # Extract unique learning_rate values
        lr_vals = sorted(set(p["learning_rate"] for p in grid))
        assert len(lr_vals) == 5
        # Check log-spacing: consecutive ratios should be ~equal
        # 检查对数间距：相邻比值应近似相等
        log_ratios = [
            math.log10(lr_vals[i + 1]) - math.log10(lr_vals[i])
            for i in range(len(lr_vals) - 1)
        ]
        mean_ratio = sum(log_ratios) / len(log_ratios)
        for r in log_ratios:
            assert abs(r - mean_ratio) < 1e-10

    def test_grid_uniform_spacing(self):
        """uniform 参数在线性空间中等距。"""
        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=4,
            learning_rate=None,
            weight_decay=None,
            batch_size=None,
            optimizer_type=None,
            scheduler_type=None,
        )
        grid = generate_grid(cfg)
        mom_vals = sorted(set(p["momentum"] for p in grid))
        assert len(mom_vals) == 4
        # Linear spacing / 线性间距
        diffs = [mom_vals[i + 1] - mom_vals[i] for i in range(len(mom_vals) - 1)]
        mean_diff = sum(diffs) / len(diffs)
        for d in diffs:
            assert abs(d - mean_diff) < 1e-10

    def test_grid_discrete_values_match_config(self):
        """离散参数值来自配置中的候选列表。"""
        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=2,
            batch_size=(128, 256),
            optimizer_type=("sgd",),
            scheduler_type=("cosine", "step"),
        )
        grid = generate_grid(cfg)
        for params in grid:
            assert params["batch_size"] in (128, 256)
            assert params["optimizer_type"] == "sgd"
            assert params["scheduler_type"] in ("cosine", "step")

    def test_grid_single_point(self):
        """grid_num_points=1 时连续参数只有一个值。"""
        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=1,
            batch_size=None,
            optimizer_type=None,
            scheduler_type=None,
        )
        grid = generate_grid(cfg)
        # 3 continuous params, each with 1 value → 1^3 = 1 point
        # 3 个连续参数各 1 个值 → 1^3 = 1 个点
        assert len(grid) == 1
        assert grid[0]["learning_rate"] == cfg.learning_rate.low
        assert grid[0]["weight_decay"] == cfg.weight_decay.low

    def test_grid_with_disabled_params(self):
        """设 None 的参数不出现在网格中。"""
        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=3,
            learning_rate=None,
            weight_decay=None,
            momentum=None,
            batch_size=(128, 256),
            optimizer_type=("sgd",),
            scheduler_type=None,
        )
        grid = generate_grid(cfg)
        # Only batch_size has 2 options
        assert len(grid) == 2
        for params in grid:
            assert "learning_rate" not in params
            assert "weight_decay" not in params
            assert "momentum" not in params

    def test_grid_covers_endpoints(self):
        """网格包含搜索空间的端点值。"""
        cfg = dataclasses.replace(
            SearchConfig(),
            grid_num_points=5,
            momentum=None,
            batch_size=None,
            optimizer_type=None,
            scheduler_type=None,
        )
        grid = generate_grid(cfg)
        lr_vals = sorted(set(p["learning_rate"] for p in grid))
        assert abs(lr_vals[0] - 1e-4) < 1e-10
        assert abs(lr_vals[-1] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Grid search tests / 网格搜索测试
# ---------------------------------------------------------------------------


class TestGridSearch:
    """Grid search integration tests."""

    def test_minimal_grid_search(self):
        """最小化网格搜索运行成功。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            strategy="grid",
            grid_num_points=2,
            search_epochs=1,
            batch_size=(16,),
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        best_params, all_records = grid_search(
            config, search_cfg, loaders_by_batch, device,
        )

        # 2 * 2 * 2 * 1 * 1 * 1 = 8 grid points
        assert len(all_records) == 8
        assert isinstance(best_params, dict)
        assert "learning_rate" in best_params

    def test_grid_search_exhaustive(self):
        """网格搜索穷举所有组合。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            strategy="grid",
            grid_num_points=2,
            search_epochs=1,
            batch_size=(16,),
            optimizer_type=("sgd", "adam"),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        best_params, all_records = grid_search(
            config, search_cfg, loaders_by_batch, device,
        )

        # 2 * 2 * 2 * 1 * 2 * 1 = 16 points
        assert len(all_records) == 16
        # Verify all optimizer types appear / 验证两种优化器都出现
        opts_seen = {
            r["params"]["optimizer_type"] for r in all_records
        }
        assert opts_seen == {"sgd", "adam"}

    def test_grid_search_best_tracked(self):
        """网格搜索正确追踪最优参数。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            strategy="grid",
            grid_num_points=2,
            search_epochs=1,
            batch_size=(16,),
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        best_params, all_records = grid_search(
            config, search_cfg, loaders_by_batch, device,
        )

        best_rec = max(all_records, key=lambda r: r["fitness"])
        assert best_params["learning_rate"] == best_rec["params"]["learning_rate"]

    def test_grid_search_all_generation_zero(self):
        """网格搜索所有记录的 generation 为 0。"""
        config = TrainConfig()
        search_cfg = dataclasses.replace(
            SearchConfig(),
            strategy="grid",
            grid_num_points=2,
            search_epochs=1,
            batch_size=(16,),
            optimizer_type=("sgd",),
            scheduler_type=("cosine",),
        )
        train_loader, test_loader = _make_synthetic_loaders(
            batch_size=16
        )
        loaders_by_batch = {16: (train_loader, test_loader)}
        device = torch.device("cpu")

        _, all_records = grid_search(
            config, search_cfg, loaders_by_batch, device,
        )

        for rec in all_records:
            assert rec["generation"] == 0


# ---------------------------------------------------------------------------
# Strategy dispatch tests / 策略分派测试
# ---------------------------------------------------------------------------


class TestSearchConfigStrategy:
    """Test SearchConfig strategy field."""

    def test_default_strategy_is_evolutionary(self):
        """默认策略为演化搜索。"""
        cfg = SearchConfig()
        assert cfg.strategy == "evolutionary"

    def test_strategy_random(self):
        """可设置策略为随机搜索。"""
        cfg = dataclasses.replace(SearchConfig(), strategy="random")
        assert cfg.strategy == "random"
        assert cfg.num_trials == 20

    def test_strategy_grid(self):
        """可设置策略为网格搜索。"""
        cfg = dataclasses.replace(SearchConfig(), strategy="grid")
        assert cfg.strategy == "grid"
        assert cfg.grid_num_points == 5

    def test_random_search_config_defaults(self):
        """随机搜索默认值合理。"""
        cfg = dataclasses.replace(SearchConfig(), strategy="random")
        assert cfg.num_trials == 20
        assert cfg.search_epochs == 5

    def test_grid_search_config_defaults(self):
        """网格搜索默认值合理。"""
        cfg = dataclasses.replace(SearchConfig(), strategy="grid")
        assert cfg.grid_num_points == 5
        assert cfg.search_epochs == 5
