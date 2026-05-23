"""
Hyperparameter search for ResNet-18 on CIFAR-100.
ResNet-18 CIFAR-100 超参数搜索。

Supports three strategies selected via SearchConfig.strategy:
支持三种策略，通过 SearchConfig.strategy 选择：

  - "evolutionary": (μ + λ) evolution strategy with tournament selection,
    BLX-α crossover, and Gaussian mutation.
    使用 (μ + λ) 演化策略，锦标赛选择、BLX-α 交叉和高斯变异。
  - "random": Uniform random sampling from the search space.
    从搜索空间均匀随机采样。
  - "grid": Exhaustive Cartesian product over all parameter combinations.
    对所有参数组合进行穷举笛卡尔积搜索。

Search space is defined in config.SearchConfig; this module reads from it.
搜索空间定义在 config.SearchConfig 中；本模块从中读取。
"""

import dataclasses
import itertools
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
    train_acc_history: list[float],
    val_loss_history: list[float],
    val_acc_history: list[float],
) -> float:
    """
    泛化导向适应度：最终验证准确率减去过拟合惩罚。
    Generalization-focused fitness: final val accuracy minus overfit penalty.

    fitness = val_acc_final - OVERFIT_PENALTY * max(0, train_acc - val_acc)

    偏好高验证准确率且低 train/val gap 的配置，惩罚死记硬背。
    train=60%, val=40% → fitness = 40% - 1.0*20% = 20%
    train=35%, val=33% → fitness = 33% - 1.0*2%  = 31%  ← 胜出

    Args:
        train_acc_history: 每个 epoch 的训练准确率
        val_loss_history: 每个 epoch 的验证损失
        val_acc_history: 每个 epoch 的验证准确率
    """
    OVERFIT_PENALTY = 1.0

    if not val_acc_history or not train_acc_history:
        return float("-inf")

    val_acc_final = val_acc_history[-1]
    train_acc_final = train_acc_history[-1]

    overfit_gap = max(0.0, train_acc_final - val_acc_final)
    fitness = val_acc_final - OVERFIT_PENALTY * overfit_gap

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
    val_loader: DataLoader,
    device: torch.device,
    seed: int,
) -> tuple[float, dict[str, list[float]]]:
    """
    Train a fresh model for search_epochs and compute fitness.
    用全新模型训练 search_epochs 轮并计算适应度。

    Returns (fitness, history_dict).
    返回 (适应度, 历史字典)。
    history 包含 train_acc, val_loss, val_acc 三组指标。

    Each probe gets a deterministic seed for reproducibility.
    每次探针使用确定性种子以保证可复现。
    """
    torch.manual_seed(seed)
    random.seed(seed)

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

        train_acc_hist: list[float] = []
        val_loss_hist: list[float] = []
        val_acc_hist: list[float] = []

        for epoch in range(1, search_cfg.search_epochs + 1):
            # 捕获训练损失和准确率 / Capture train loss & accuracy
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer,
                criterion, device, epoch,
                aug_config=config.augmentation,
                num_classes=config.num_classes,
            )
            val_loss, val_acc = evaluate(
                model, val_loader, device
            )
            if scheduler is not None:
                scheduler.step()

            train_acc_hist.append(train_acc)
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_acc)

        fitness = compute_fitness(
            train_acc_hist, val_loss_hist, val_acc_hist
        )
        history = {
            "train_acc": train_acc_hist,
            "val_loss": val_loss_hist,
            "val_acc": val_acc_hist,
        }
        return fitness, history

    except RuntimeError as e:
        # CUDA OOM or other runtime errors → penalize
        # CUDA OOM 或其他运行时错误 → 惩罚
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
        return float("-inf"), {
            "train_acc": [], "val_loss": [], "val_acc": [],
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
        train_loader, val_loader = loaders_by_batch[bs]
        fitness, history = train_probe(
            params, base_config, search_cfg,
            train_loader, val_loader, device,
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
            train_loader, val_loader = loaders_by_batch[bs]
            fitness, history = train_probe(
                child_params, base_config, search_cfg,
                train_loader, val_loader, device,
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
# Random search / 随机搜索
# ---------------------------------------------------------------------------

def random_search(
    base_config: TrainConfig,
    search_cfg: SearchConfig,
    loaders_by_batch: dict[int, tuple[DataLoader, DataLoader]],
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """
    Random search: sample num_trials random configs and evaluate each.
    随机搜索：从搜索空间采样 num_trials 组随机配置并逐一评估。

    Simple but effective baseline; often competitive with evolutionary
    for low-dimensional search spaces.
    简单有效的基线方法；在低维搜索空间中常与演化方法不相上下。

    Returns (best_params_dict, all_evaluated_records).
    返回 (最优参数字典, 所有评估记录列表)。
    """
    rng = random.Random(_BASE_SEED)
    all_records: list[dict[str, object]] = []
    best_fitness = float("-inf")
    best_params: dict[str, object] = {}

    print(f"\n--- Random Search: {search_cfg.num_trials} trials ---")

    for i in range(search_cfg.num_trials):
        params = sample_individual(search_cfg, rng)
        bs = int(params.get("batch_size", base_config.batch_size))
        train_loader, val_loader = loaders_by_batch[bs]

        fitness, history = train_probe(
            params, base_config, search_cfg,
            train_loader, val_loader, device,
            seed=_BASE_SEED + i,
        )

        # Track best / 追踪最优
        if fitness > best_fitness:
            best_fitness = fitness
            best_params = dict(params)

        eval_id = i + 1
        record = _make_record(eval_id, 0, params, fitness, history)
        all_records.append(record)
        print(
            f"  [{eval_id}/{search_cfg.num_trials}]"
            f" fitness={fitness:.4f}"
            f" | lr={params.get('learning_rate', 'N/A'):.4f}"
            f" | opt={params.get('optimizer_type', 'N/A')}"
            f" | bs={params.get('batch_size', 'N/A')}"
        )

    print(
        f"\n  Random search best fitness: {best_fitness:.4f}"
    )
    return best_params, all_records


# ---------------------------------------------------------------------------
# Grid search / 网格搜索
# ---------------------------------------------------------------------------

def _linspace(start: float, stop: float, num: int) -> list[float]:
    """
    Generate num evenly-spaced values in [start, stop].
    在 [start, stop] 中生成 num 个等距值。

    Pure Python equivalent of numpy.linspace (avoids numpy dependency).
    纯 Python 实现的 numpy.linspace（避免引入 numpy 依赖）。
    """
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def _logspace(
    log_start: float, log_stop: float, num: int,
) -> list[float]:
    """
    Generate num values evenly-spaced in log10 space.
    在 log10 空间中生成 num 个等距值。

    Equivalent to 10 ** numpy.linspace(log_start, log_stop, num).
    等价于 10 ** numpy.linspace(log_start, log_stop, num)。
    """
    logs = _linspace(log_start, log_stop, num)
    return [10 ** x for x in logs]


def generate_grid(search_cfg: SearchConfig) -> list[dict[str, object]]:
    """
    Generate all grid points as Cartesian product of parameter values.
    生成所有参数值笛卡尔积构成的网格点。

    Continuous params: linspace or logspace based on distribution.
    连续参数：根据分布类型使用 linspace 或 logspace。

    Discrete params: use candidate tuples directly.
    离散参数：直接使用候选元组。

    Returns list of param dicts for train_probe evaluation.
    返回用于 train_probe 评估的参数字典列表。
    """
    dim_values: dict[str, list[object]] = {}

    # Continuous params / 连续参数
    if search_cfg.learning_rate is not None:
        r = search_cfg.learning_rate
        if r.distribution == "log_uniform":
            dim_values["learning_rate"] = _logspace(
                math.log10(r.low), math.log10(r.high),
                search_cfg.grid_num_points,
            )
        else:
            dim_values["learning_rate"] = _linspace(
                r.low, r.high, search_cfg.grid_num_points,
            )

    if search_cfg.weight_decay is not None:
        r = search_cfg.weight_decay
        if r.distribution == "log_uniform":
            dim_values["weight_decay"] = _logspace(
                math.log10(r.low), math.log10(r.high),
                search_cfg.grid_num_points,
            )
        else:
            dim_values["weight_decay"] = _linspace(
                r.low, r.high, search_cfg.grid_num_points,
            )

    if search_cfg.momentum is not None:
        r = search_cfg.momentum
        dim_values["momentum"] = _linspace(
            r.low, r.high, search_cfg.grid_num_points,
        )

    # Discrete params / 离散参数
    if search_cfg.batch_size is not None:
        dim_values["batch_size"] = list(search_cfg.batch_size)
    if search_cfg.optimizer_type is not None:
        dim_values["optimizer_type"] = list(search_cfg.optimizer_type)
    if search_cfg.scheduler_type is not None:
        dim_values["scheduler_type"] = list(search_cfg.scheduler_type)

    # Cartesian product / 笛卡尔积
    keys = sorted(dim_values.keys())
    values = [dim_values[k] for k in keys]

    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*values)
    ]


def grid_search(
    base_config: TrainConfig,
    search_cfg: SearchConfig,
    loaders_by_batch: dict[int, tuple[DataLoader, DataLoader]],
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """
    Grid search: exhaustive evaluation of all parameter combinations.
    网格搜索：穷举评估所有参数组合。

    Warning: grid size grows exponentially with the number of parameters.
    Use SearchConfig.grid_num_points and set non-essential params to None
    to control total evaluations.
    注意：网格大小随参数数量指数增长。用 grid_num_points 控制连续参数密度，
    将非关键参数设为 None 以控制总评估数。

    Returns (best_params_dict, all_evaluated_records).
    返回 (最优参数字典, 所有评估记录列表)。
    """
    grid_points = generate_grid(search_cfg)
    total = len(grid_points)

    print(f"\n--- Grid Search: {total} combinations ---")

    all_records: list[dict[str, object]] = []
    best_fitness = float("-inf")
    best_params: dict[str, object] = {}

    for i, params in enumerate(grid_points):
        bs = int(params.get("batch_size", base_config.batch_size))
        train_loader, val_loader = loaders_by_batch[bs]

        fitness, history = train_probe(
            params, base_config, search_cfg,
            train_loader, val_loader, device,
            seed=_BASE_SEED + i,
        )

        # Track best / 追踪最优
        if fitness > best_fitness:
            best_fitness = fitness
            best_params = dict(params)

        eval_id = i + 1
        record = _make_record(eval_id, 0, params, fitness, history)
        all_records.append(record)
        print(
            f"  [{eval_id}/{total}]"
            f" fitness={fitness:.4f}"
            f" | lr={params.get('learning_rate', 'N/A'):.6f}"
            f" | opt={params.get('optimizer_type', 'N/A')}"
            f" | bs={params.get('batch_size', 'N/A')}"
        )

    print(
        f"\n  Grid search best fitness: {best_fitness:.4f}"
    )
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
    train_acc_hist = history.get("train_acc", [])
    val_acc_hist = history.get("val_acc", [])
    val_loss_hist = history.get("val_loss", [])
    return {
        "id": eval_id,
        "generation": generation,
        "params": params,
        "fitness": round(fitness, 6),
        "final_val_acc": round(val_acc_hist[-1], 4)
        if val_acc_hist else None,
        "final_val_loss": round(val_loss_hist[-1], 4)
        if val_loss_hist else None,
        "train_acc_history": [round(v, 4) for v in train_acc_hist],
        "val_acc_history": [round(v, 4) for v in val_acc_hist],
        "val_loss_history": [round(v, 4) for v in val_loss_hist],
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
            "strategy": search_cfg.strategy,
            "search_epochs": search_cfg.search_epochs,
            "total_evaluations": len(all_records),
        }),
        ("best", OrderedDict([
            ("params", best_params),
            ("fitness", best_record.get("fitness")),
            ("final_val_acc", best_record.get("final_val_acc")),
            ("final_val_loss", best_record.get("final_val_loss")),
            ("train_acc_history", best_record.get("train_acc_history")),
            ("val_acc_history", best_record.get("val_acc_history")),
            ("val_loss_history", best_record.get("val_loss_history")),
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

def _estimate_total_evaluations(search_cfg: SearchConfig) -> int:
    """
    Estimate total evaluations for the given strategy.
    估算给定策略的总评估次数。
    """
    strategy = search_cfg.strategy.lower()
    if strategy == "evolutionary":
        return (
            search_cfg.population_size
            + search_cfg.num_generations
            * search_cfg.offspring_per_gen
        )
    elif strategy == "random":
        return search_cfg.num_trials
    elif strategy == "grid":
        return len(generate_grid(search_cfg))
    else:
        return 0


# Strategy label map / 策略标签映射
_STRATEGY_LABELS: dict[str, tuple[str, str]] = {
    "evolutionary": (
        "Evolutionary Hyperparameter Search",
        "进化超参数搜索",
    ),
    "random": (
        "Random Hyperparameter Search",
        "随机超参数搜索",
    ),
    "grid": (
        "Grid Hyperparameter Search",
        "网格超参数搜索",
    ),
}


def run_search(
    config: TrainConfig,
    train_loader: DataLoader,
    search_cfg: SearchConfig | None = None,
) -> dict[str, object]:
    """
    Run hyperparameter search and save results.
    运行超参数搜索并保存结果。

    Strategy is selected via search_cfg.strategy:
    "evolutionary", "random", or "grid".
    通过 search_cfg.strategy 选择策略。

    Returns dict of best params for use with
    dataclasses.replace(config, **best_params).
    返回最优参数字典，可用于
    dataclasses.replace(config, **best_params)。

    Uses validation split (get_cifar100_search_loaders) during search;
    test set is never touched during search.
    搜索期间使用验证集划分；test set 在搜索期间完全不碰。

    Handles DataLoader creation for different batch_sizes by
    pre-creating loaders for each candidate batch_size.
    通过预创建每种候选 batch_size 的 DataLoader
    来处理不同批次大小。
    """
    if search_cfg is None:
        search_cfg = SearchConfig()

    strategy = search_cfg.strategy.lower()
    if strategy not in _STRATEGY_LABELS:
        raise ValueError(
            f"Unknown search strategy: {search_cfg.strategy}. "
            f"Supported: evolutionary, random, grid"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    label_en, label_cn = _STRATEGY_LABELS[strategy]
    total_evals = _estimate_total_evaluations(search_cfg)

    print("=" * 60)
    print(label_en)
    print(label_cn)
    print("=" * 60)
    if strategy == "evolutionary":
        print(f"  Population: {search_cfg.population_size}")
        print(f"  Offspring/gen: {search_cfg.offspring_per_gen}")
        print(f"  Generations: {search_cfg.num_generations}")
    elif strategy == "random":
        print(f"  Trials: {search_cfg.num_trials}")
    elif strategy == "grid":
        print(f"  Grid points/dim: {search_cfg.grid_num_points}")
    print(
        f"  Search epochs: {search_cfg.search_epochs}"
    )
    print(
        f"  Total evaluations: {total_evals}"
    )
    print(f"  Device: {device}")
    print()

    # Pre-create search loaders (train_subset + val) for each batch_size
    # 预创建每种 batch_size 的搜索用 DataLoader（训练子集 + 验证集）
    from .data import get_cifar100_search_loaders

    loaders_by_batch: dict[int, tuple[DataLoader, DataLoader]] = {}
    batch_sizes = (
        search_cfg.batch_size
        if search_cfg.batch_size is not None
        else (config.batch_size,)
    )
    for bs in batch_sizes:
        cfg_bs = dataclasses.replace(config, batch_size=bs)
        loaders_by_batch[bs] = get_cifar100_search_loaders(
            cfg_bs
        )

    # Run search / 运行搜索
    if strategy == "evolutionary":
        best_params, all_records = evolutionary_search(
            config, search_cfg, loaders_by_batch, device,
        )
    elif strategy == "random":
        best_params, all_records = random_search(
            config, search_cfg, loaders_by_batch, device,
        )
    elif strategy == "grid":
        best_params, all_records = grid_search(
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
