"""
Evolutionary hyperparameter search for ResNet-18 on CIFAR-100.
ResNet-18 CIFAR-100 进化超参数搜索。

Uses a (μ + λ) evolution strategy with tournament selection,
BLX-α crossover, and Gaussian mutation on mixed continuous/discrete params.
使用 (μ + λ) 演化策略，锦标赛选择、BLX-α 交叉和高斯变异，
适用于混合连续/离散参数空间。

Search space is defined in config.SearchConfig; this module reads from it.
搜索空间定义在 config.SearchConfig 中；本模块从中读取。
"""

import dataclasses
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import HyperparamRange, SearchConfig, TrainConfig
from .evaluate import evaluate
from .model import create_model
from .train import create_criterion, create_optimizer, create_scheduler, train_one_epoch


# ---------------------------------------------------------------------------
# Individual representation / 个体表示
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Individual:
    """
    A candidate hyperparameter combination + its fitness.
    候选超参数组合及其适应度。

    fitness is None before evaluation / 评估前 fitness 为 None。
    """

    learning_rate: float
    weight_decay: float
    momentum: float
    batch_size: int
    optimizer_type: str
    scheduler_type: str
    fitness: float | None = None


# ---------------------------------------------------------------------------
# Sampling helpers / 采样辅助函数
# ---------------------------------------------------------------------------

_BASE_SEED = 42


def sample_log_uniform(
    low: float, high: float, rng: random.Random
) -> float:
    """
    Sample from log-uniform distribution / 从对数均匀分布采样。

    Equal probability across orders of magnitude.
    跨数量级等概率采样。适用于 LR、weight_decay 等参数。
    """
    log_low, log_high = math.log10(low), math.log10(high)
    return 10 ** rng.uniform(log_low, log_high)


def sample_individual(
    search_cfg: SearchConfig, rng: random.Random
) -> dict[str, object]:
    """
    Sample a random hyperparameter dict from search space.
    从搜索空间随机采样一组超参数字典。

    Only includes params whose range/choices is not None.
    只包含范围/选项非 None 的参数。
    """
    params: dict[str, object] = {}

    # Continuous params / 连续参数
    if search_cfg.learning_rate is not None:
        r = search_cfg.learning_rate
        params["learning_rate"] = (
            sample_log_uniform(r.low, r.high, rng)
            if r.distribution == "log_uniform"
            else rng.uniform(r.low, r.high)
        )
    if search_cfg.weight_decay is not None:
        r = search_cfg.weight_decay
        params["weight_decay"] = (
            sample_log_uniform(r.low, r.high, rng)
            if r.distribution == "log_uniform"
            else rng.uniform(r.low, r.high)
        )
    if search_cfg.momentum is not None:
        r = search_cfg.momentum
        params["momentum"] = rng.uniform(r.low, r.high)

    # Discrete params / 离散参数
    if search_cfg.batch_size is not None:
        params["batch_size"] = rng.choice(search_cfg.batch_size)
    if search_cfg.optimizer_type is not None:
        params["optimizer_type"] = rng.choice(
            search_cfg.optimizer_type
        )
    if search_cfg.scheduler_type is not None:
        params["scheduler_type"] = rng.choice(
            search_cfg.scheduler_type
        )

    return params


# ---------------------------------------------------------------------------
# Fitness functions / 适应度函数
# ---------------------------------------------------------------------------

def _linear_slope(values: list[float]) -> float:
    """
    Compute linear regression slope of values over epochs.
    计算值随 epoch 的线性回归斜率。

    Simple O(n) implementation without external deps.
    无外部依赖的 O(n) 实现。
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in enumerate(values)
    )
    den = sum((x - mean_x) ** 2 for x in range(n))
    return num / den if den > 0 else 0.0


def accuracy_improvement_rate(
    acc_history: list[float],
) -> float:
    """
    AIR: slope of test accuracy over epochs (higher = better).
    准确率提升速率：测试准确率随 epoch 的回归斜率。
    """
    return _linear_slope(acc_history)


def loss_decrease_rate(
    loss_history: list[float],
) -> float:
    """
    LDR: negated slope of test loss over epochs (higher = better).
    损失下降速率：测试损失斜率取反（越大越好）。
    """
    return -_linear_slope(loss_history)


def compute_fitness(
    test_loss_history: list[float],
    test_acc_history: list[float],
) -> float:
    """
    Combined fitness: weighted AIR + LDR.
    综合适应度：加权准确率提升速率 + 损失下降速率。

    Uses test metrics (not train) to favor generalization.
    使用测试指标（非训练）以偏好泛化性好的配置。
    """
    air = accuracy_improvement_rate(test_acc_history)
    ldr = loss_decrease_rate(test_loss_history)

    # Scale AIR × 10 to match LDR magnitude
    # AIR 通常 0.01-0.10/epoch，LDR 通常 0.1-0.5/epoch
    fitness = 10.0 * air + ldr

    # Penalty for divergence / 发散惩罚
    if test_acc_history[-1] < test_acc_history[0]:
        fitness *= 0.5
    if test_loss_history[-1] > test_loss_history[0]:
        fitness *= 0.5

    # NaN/Inf guard / 异常值保护
    if not math.isfinite(fitness):
        return float("-inf")

    return fitness


# ---------------------------------------------------------------------------
# Evolutionary operators / 演化算子
# ---------------------------------------------------------------------------

def tournament_select(
    population: list[Individual],
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    """
    Tournament selection: pick best among k random individuals.
    锦标赛选择：从 k 个随机个体中选最优。
    """
    candidates = rng.sample(
        population, min(tournament_size, len(population))
    )
    # fitness is guaranteed set after evaluation
    return max(candidates, key=lambda ind: ind.fitness or float("-inf"))


def _clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    search_cfg: SearchConfig,
    rng: random.Random,
) -> dict[str, object]:
    """
    BLX-α crossover for continuous, uniform for discrete.
    连续参数用 BLX-α 交叉，离散参数均匀选择。
    """
    alpha = 0.3
    child: dict[str, object] = {}

    # Continuous params: BLX-α / 连续参数：BLX-α 交叉
    for name in ("learning_rate", "weight_decay", "momentum"):
        va = getattr(parent_a, name)
        vb = getattr(parent_b, name)
        lo, hi = min(va, vb), max(va, vb)
        spread = hi - lo
        # Get range from search_cfg
        hp_range: HyperparamRange | None = getattr(
            search_cfg, name
        )
        if hp_range is not None:
            child[name] = _clip(
                rng.uniform(
                    lo - alpha * spread, hi + alpha * spread
                ),
                hp_range.low,
                hp_range.high,
            )
        else:
            child[name] = rng.choice([va, vb])

    # Discrete params: uniform selection / 离散参数：均匀选择
    for name in ("batch_size", "optimizer_type", "scheduler_type"):
        va = getattr(parent_a, name)
        vb = getattr(parent_b, name)
        child[name] = rng.choice([va, vb])

    return child


def mutate(
    params: dict[str, object],
    search_cfg: SearchConfig,
    rng: random.Random,
) -> dict[str, object]:
    """
    Per-gene mutation: Gaussian for continuous, resample for discrete.
    逐基因变异：连续参数高斯扰动，离散参数重新采样。
    """
    result = dict(params)

    for name in ("learning_rate", "weight_decay", "momentum"):
        if rng.random() > search_cfg.mutation_rate:
            continue
        hp_range = getattr(search_cfg, name)
        if hp_range is None:
            continue
        val = float(result[name])
        # Multiplicative Gaussian noise / 乘性高斯噪声
        val *= 1.0 + rng.gauss(0, 0.2)
        result[name] = _clip(val, hp_range.low, hp_range.high)

    for name, choices in [
        ("batch_size", search_cfg.batch_size),
        ("optimizer_type", search_cfg.optimizer_type),
        ("scheduler_type", search_cfg.scheduler_type),
    ]:
        if choices is None:
            continue
        if rng.random() > search_cfg.mutation_rate:
            continue
        result[name] = rng.choice(choices)

    return result


# ---------------------------------------------------------------------------
# Training probe / 短训练探针
# ---------------------------------------------------------------------------

def train_probe(
    params: dict[str, object],
    base_config: TrainConfig,
    search_cfg: SearchConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    seed: int,
) -> tuple[float, dict[str, list[float]]]:
    """
    Train a fresh model for search_epochs and compute fitness.
    用全新模型训练 search_epochs 轮并计算适应度。

    Returns (fitness, history_dict).
    返回 (适应度, 历史字典)。

    Each probe gets a deterministic seed for reproducibility.
    每次探针使用确定性种子以保证可复现。
    """
    torch.manual_seed(seed)

    # Build config for this probe / 为本次探针构建配置
    overrides = {
        "epochs": search_cfg.search_epochs,
        "patience": 999,  # Disable early stopping / 禁用早停
        "scheduler_t_max": search_cfg.search_epochs,
    }
    for k, v in params.items():
        overrides[k] = v
    config = dataclasses.replace(base_config, **overrides)

    # Fresh model each probe / 每次探针使用全新模型
    model = create_model(
        num_classes=config.num_classes
    ).to(device)

    try:
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        criterion = create_criterion(config)

        test_loss_hist: list[float] = []
        test_acc_hist: list[float] = []

        for epoch in range(1, search_cfg.search_epochs + 1):
            train_one_epoch(
                model, train_loader, optimizer,
                criterion, device, epoch,
            )
            test_loss, test_acc = evaluate(
                model, test_loader, device
            )
            if scheduler is not None:
                scheduler.step()

            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)

        fitness = compute_fitness(test_loss_hist, test_acc_hist)
        history = {
            "test_loss": test_loss_hist,
            "test_acc": test_acc_hist,
        }
        return fitness, history

    except RuntimeError as e:
        # CUDA OOM or other runtime errors → penalize
        # CUDA OOM 或其他运行时错误 → 惩罚
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
        return float("-inf"), {
            "test_loss": [], "test_acc": [],
        }


# ---------------------------------------------------------------------------
# Main evolutionary loop / 演化主循环
# ---------------------------------------------------------------------------

def evolutionary_search(
    base_config: TrainConfig,
    search_cfg: SearchConfig,
    loaders_by_batch: dict[int, tuple[DataLoader, DataLoader]],
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """
    Run the (μ + λ) evolutionary search.
    运行 (μ + λ) 演化搜索。

    Returns (best_params_dict, all_evaluated_records).
    返回 (最优参数字典, 所有评估记录列表)。
    """
    rng = random.Random(_BASE_SEED)
    eval_id = 0
    all_records: list[dict[str, object]] = []

    # Generation 0: random initialization / 初始代：随机初始化
    print("\n--- Generation 0: Random Initialization ---")
    population: list[Individual] = []
    for i in range(search_cfg.population_size):
        params = sample_individual(search_cfg, rng)
        bs = int(params.get("batch_size", base_config.batch_size))
        train_loader, test_loader = loaders_by_batch[bs]
        fitness, history = train_probe(
            params, base_config, search_cfg,
            train_loader, test_loader, device,
            seed=_BASE_SEED + eval_id,
        )
        ind = Individual(
            fitness=fitness, **params
        )
        population.append(ind)

        eval_id += 1
        record = _make_record(
            eval_id, 0, params, fitness, history
        )
        all_records.append(record)
        print(
            f"  [{eval_id}] fitness={fitness:.4f} "
            f"| lr={params.get('learning_rate', 'N/A'):.4f} "
            f"| opt={params.get('optimizer_type', 'N/A')} "
            f"| bs={params.get('batch_size', 'N/A')}"
        )

    # Evolutionary generations / 演化世代
    for gen in range(1, search_cfg.num_generations + 1):
        print(f"\n--- Generation {gen} ---")
        offspring: list[Individual] = []

        for _ in range(search_cfg.offspring_per_gen):
            # Select parents / 选择父代
            p_a = tournament_select(
                population, search_cfg.tournament_size, rng
            )
            p_b = tournament_select(
                population, search_cfg.tournament_size, rng
            )

            # Crossover + mutate / 交叉 + 变异
            child_params = crossover(
                p_a, p_b, search_cfg, rng
            )
            child_params = mutate(
                child_params, search_cfg, rng
            )

            # Evaluate / 评估
            bs = int(
                child_params.get(
                    "batch_size", base_config.batch_size
                )
            )
            train_loader, test_loader = loaders_by_batch[bs]
            fitness, history = train_probe(
                child_params, base_config, search_cfg,
                train_loader, test_loader, device,
                seed=_BASE_SEED + eval_id,
            )

            child = Individual(
                fitness=fitness, **child_params
            )
            offspring.append(child)

            eval_id += 1
            record = _make_record(
                eval_id, gen, child_params, fitness, history
            )
            all_records.append(record)
            print(
                f"  [{eval_id}] fitness={fitness:.4f} "
                f"| lr={child_params.get('learning_rate', 'N/A'):.4f} "
                f"| opt={child_params.get('optimizer_type', 'N/A')}"
                f" | bs={child_params.get('batch_size', 'N/A')}"
            )

        # (μ + λ) selection: merge and keep top μ
        # (μ + λ) 选择：合并后保留前 μ 个
        combined = population + offspring
        combined.sort(
            key=lambda ind: ind.fitness or float("-inf"),
            reverse=True,
        )
        population = combined[:search_cfg.population_size]

        best_fitness = population[0].fitness or 0.0
        mean_fitness = sum(
            ind.fitness or 0.0 for ind in population
        ) / len(population)
        print(
            f"  Gen {gen} best={best_fitness:.4f} "
            f"mean={mean_fitness:.4f}"
        )

    # Return best individual's params / 返回最优个体参数
    best = population[0]
    best_params = {
        "learning_rate": best.learning_rate,
        "weight_decay": best.weight_decay,
        "momentum": best.momentum,
        "batch_size": best.batch_size,
        "optimizer_type": best.optimizer_type,
        "scheduler_type": best.scheduler_type,
    }
    return best_params, all_records


# ---------------------------------------------------------------------------
# Result logging / 结果记录
# ---------------------------------------------------------------------------

def _make_record(
    eval_id: int,
    generation: int,
    params: dict[str, object],
    fitness: float,
    history: dict[str, list[float]],
) -> dict[str, object]:
    """Build a JSON-serializable evaluation record."""
    acc_hist = history.get("test_acc", [])
    loss_hist = history.get("test_loss", [])
    return {
        "id": eval_id,
        "generation": generation,
        "params": params,
        "fitness": round(fitness, 6),
        "final_test_acc": round(acc_hist[-1], 4)
        if acc_hist else None,
        "final_test_loss": round(loss_hist[-1], 4)
        if loss_hist else None,
        "test_acc_history": [round(v, 4) for v in acc_hist],
        "test_loss_history": [round(v, 4) for v in loss_hist],
    }


def log_search_results(
    best_params: dict[str, object],
    all_records: list[dict[str, object]],
    search_cfg: SearchConfig,
    config: TrainConfig,
) -> Path:
    """
    Save search results to JSON / 将搜索结果保存为 JSON。

    File: {checkpoint_dir}/hp_search_results.json
    """
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_dir / "hp_search_results.json"

    # Generation summaries / 每代摘要
    gen_summary: list[dict[str, float]] = []
    max_gen = search_cfg.num_generations
    for g in range(max_gen + 1):
        gen_records = [
            r for r in all_records
            if r["generation"] == g
        ]
        if not gen_records:
            continue
        fitnesses = [
            r["fitness"] for r in gen_records
            if r["fitness"] is not None
        ]
        gen_summary.append({
            "generation": g,
            "best_fitness": round(max(fitnesses), 4),
            "mean_fitness": round(
                sum(fitnesses) / len(fitnesses), 4
            ),
            "worst_fitness": round(min(fitnesses), 4),
            "num_evaluated": len(gen_records),
        })

    # Best individual details / 最优个体详情
    best_record = max(
        all_records, key=lambda r: r["fitness"] or float("-inf")
    )

    results = OrderedDict([
        ("search_config", {
            "population_size": search_cfg.population_size,
            "offspring_per_gen": search_cfg.offspring_per_gen,
            "num_generations": search_cfg.num_generations,
            "search_epochs": search_cfg.search_epochs,
            "total_evaluations": len(all_records),
        }),
        ("best", OrderedDict([
            ("params", best_params),
            ("fitness", best_record.get("fitness")),
            ("final_test_acc", best_record.get("final_test_acc")),
            ("final_test_loss", best_record.get("final_test_loss")),
            ("test_acc_history", best_record.get("test_acc_history")),
            ("test_loss_history", best_record.get("test_loss_history")),
        ])),
        ("generation_summary", gen_summary),
        ("all_evaluated", all_records),
    ])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return path


def load_best_search_params(
    config: TrainConfig,
) -> dict[str, object] | None:
    """
    Load best params from hp_search_results.json if it exists.
    如果搜索结果文件存在，加载最优超参数。

    Returns None if file doesn't exist.
    文件不存在时返回 None。
    """
    path = config.checkpoint_dir / "hp_search_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["best"]["params"]


# ---------------------------------------------------------------------------
# Public API / 公共接口
# ---------------------------------------------------------------------------

def run_search(
    config: TrainConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    search_cfg: SearchConfig | None = None,
) -> dict[str, object]:
    """
    Run evolutionary hyperparameter search and save results.
    运行进化超参数搜索并保存结果。

    Returns dict of best params for use with
    dataclasses.replace(config, **best_params).
    返回最优参数字典，可用于
    dataclasses.replace(config, **best_params)。

    Handles DataLoader creation for different batch_sizes by
    pre-creating loaders for each candidate batch_size.
    通过预创建每种候选 batch_size 的 DataLoader
    来处理不同批次大小。
    """
    if search_cfg is None:
        search_cfg = SearchConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Evolutionary Hyperparameter Search")
    print("进化超参数搜索")
    print("=" * 60)
    print(f"  Population: {search_cfg.population_size}")
    print(f"  Offspring/gen: {search_cfg.offspring_per_gen}")
    print(f"  Generations: {search_cfg.num_generations}")
    print(
        f"  Search epochs: {search_cfg.search_epochs}"
    )
    print(
        f"  Total evaluations: "
        f"{search_cfg.population_size + search_cfg.num_generations * search_cfg.offspring_per_gen}"
    )
    print(f"  Device: {device}")
    print()

    # Pre-create loaders for each batch_size candidate
    # 预创建每种 batch_size 的 DataLoader
    loaders_by_batch: dict[int, tuple[DataLoader, DataLoader]] = {}
    if search_cfg.batch_size is not None:
        for bs in search_cfg.batch_size:
            if bs == config.batch_size:
                loaders_by_batch[bs] = (
                    train_loader, test_loader
                )
            else:
                from .data import get_cifar100_loaders
                cfg_bs = dataclasses.replace(
                    config, batch_size=bs
                )
                loaders_by_batch[bs] = get_cifar100_loaders(
                    cfg_bs
                )
    else:
        loaders_by_batch[config.batch_size] = (
            train_loader, test_loader
        )

    # Run search / 运行搜索
    best_params, all_records = evolutionary_search(
        config, search_cfg, loaders_by_batch, device,
    )

    # Save results / 保存结果
    results_path = log_search_results(
        best_params, all_records, search_cfg, config,
    )

    print("\n" + "=" * 60)
    print("Search Complete / 搜索完成")
    print("=" * 60)
    print(f"  Best fitness: {max(r['fitness'] for r in all_records):.4f}")
    print(f"  Best params:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Results saved to: {results_path}")

    return best_params
