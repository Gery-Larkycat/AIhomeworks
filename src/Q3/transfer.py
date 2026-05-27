"""
Transfer learning: CIFAR-100 pretrained → CIFAR-10 fine-tune.
迁移学习：CIFAR-100 预训练 → CIFAR-10 微调。

Loads the full pretrained model, freezes backbone, replaces FC,
and trains only the FC layer on CIFAR-10. Supports hyperparameter
search via skorch + sklearn (same framework as CIFAR-100 search).

加载完整预训练模型，冻结 backbone，替换 FC，
仅训练 FC 层。支持通过 skorch + sklearn 做超参搜索。
"""

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import loguniform, uniform
from skorch import NeuralNetClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from torchvision import datasets

from .checkpoint import save_training_history
from .config import TrainConfig, TransferConfig
from .data import get_cifar10_loaders
from Q2.model import ResNet18
from .train import train


# ---------------------------------------------------------------------------
# Model preparation / 模型准备
# ---------------------------------------------------------------------------


def load_pretrained_model(
    source_checkpoint: Path,
    source_num_classes: int = 100,
    target_num_classes: int = 10,
) -> ResNet18:
    """
    加载 CIFAR-100 完整检查点，保留原 FC 并追加新分类层。

    流程：
    1. 创建 ResNet18(source_num_classes) 并加载完整预训练权重
    2. 保留原始 FC（512→source_num_classes）用于微调
    3. 在原始 FC 后追加新分类层（source_num_classes→target_num_classes）

    非 FC 层保留预训练权重（特征提取能力），
    原始 FC 从预训练权重继续微调，新分类层随机初始化。

    模型 forward 路径变为：
    backbone → 原始 FC (512→100) → 新分类层 (100→10)

    Args:
        source_checkpoint: CIFAR-100 检查点路径（含 model_state_dict）
        source_num_classes: 源模型 FC 输出维度（CIFAR-100 = 100）
        target_num_classes: 目标分类层输出维度（CIFAR-10 = 10）

    Returns:
        模型（backbone 已加载预训练权重，原 FC 保留，新分类层随机初始化）
    """
    model = ResNet18(num_classes=source_num_classes)
    checkpoint = torch.load(source_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # 保留原始 FC，在其后追加新分类层
    # 原始 FC (512→100) 从预训练权重微调
    # 新分类层 (100→10) 随机初始化
    original_fc = model.fc
    model.fc = nn.Sequential(
        original_fc,
        nn.Linear(source_num_classes, target_num_classes),
    )
    return model


def freeze_backbone(model: ResNet18) -> None:
    """
    冻结除 FC 外的所有参数（conv, bn, avgpool）。
    in-place 操作，仅 FC 层保持 requires_grad=True。
    """
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def print_transfer_summary(model: ResNet18) -> None:
    """
    打印迁移学习摘要：冻结参数数、可训练参数数、可训练层名。
    """
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_names = [
        n for n, p in model.named_parameters() if p.requires_grad
    ]
    print(f"  Frozen params: {frozen:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Trainable layers: {trainable_names}")


def find_best_cifar100_checkpoint(
    checkpoints_root: Path = Path("checkpoints"),
    best_filename: str = "resnet18_cifar100_best.pth",
) -> Path | None:
    """
    从 checkpoints_root 下找到 CIFAR-100 准确率最高的最佳检查点。

    扫描所有子目录，读取每个检查点的 accuracy 字段进行比较，
    返回准确率最高的检查点路径。同准确率时取最新的（按目录名排序）。

    回退兼容：直接检查 checkpoints_root/best_filename（旧平面结构）。

    Args:
        checkpoints_root: 检查点根目录，默认 checkpoints
        best_filename: 检查点文件名，默认 resnet18_cifar100_best.pth

    Returns:
        准确率最高的检查点路径，不存在则 None
    """
    if not checkpoints_root.exists():
        return None

    candidates: list[tuple[float, Path]] = []

    # 按名称倒序扫描（ISO 格式时间戳 = 字典序 = 时间序）
    # 同准确率时取最新的
    subdirs = sorted(
        (d for d in checkpoints_root.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    for subdir in subdirs:
        ckpt_path = subdir / best_filename
        if not ckpt_path.exists():
            continue
        # 仅读取 accuracy 元数据，不加载完整权重到 GPU
        ckpt = torch.load(ckpt_path, weights_only=False)
        acc = float(ckpt.get("accuracy", 0.0))
        candidates.append((acc, ckpt_path))

    # 回退：平面目录兼容旧结构
    flat = checkpoints_root / best_filename
    if flat.exists():
        ckpt = torch.load(flat, weights_only=False)
        acc = float(ckpt.get("accuracy", 0.0))
        candidates.append((acc, flat))

    if not candidates:
        return None

    # 按准确率降序，取最优
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# skorch TransferNetClassifier / skorch 迁移学习包装器
# ---------------------------------------------------------------------------


class _TransferNetClassifier(NeuralNetClassifier):
    """
    skorch 包装器：每次初始化模块后自动加载预训练权重并冻结 backbone。

    使 sklearn CV 的每次 clone/fit 都从相同的预训练状态开始，
    仅 FC 层可训练。覆写 initialize_module() 在模块创建后注入预训练权重。

    设计决策：
    - 不覆写 __init__，通过 module__ 前缀传递参数
    - source_checkpoint 在 initialize_module 中读取，不作为构造参数
      （避免 sklearn clone 序列化问题）
    """

    def initialize_module(self):
        """
        创建模块 → 加载完整预训练权重 → 保留原 FC + 追加新分类层 → 冻结 backbone。
        每次 fit() 都会重新调用，确保 CV 每折从相同预训练状态开始。
        """
        super().initialize_module()
        # 从 module kwargs 中获取 source_checkpoint
        source_ckpt = self.module__source_checkpoint
        if source_ckpt is not None and Path(source_ckpt).exists():
            checkpoint = torch.load(source_ckpt, weights_only=False)
            # 加载完整权重（含 FC），非 strict 因为 module 创建时
            # 可能已有不同维度
            self.module_.load_state_dict(
                checkpoint["model_state_dict"], strict=True,
            )
            # 保留原始 FC，追加新分类层
            original_fc = self.module_.fc
            target = self.module__target_num_classes
            self.module_.fc = nn.Sequential(
                original_fc,
                nn.Linear(original_fc.out_features, target),
            )
        # 冻结 backbone，仅 FC 层可训练
        # fc.0 = 原始 FC（微调），fc.1 = 新分类层
        for name, param in self.module_.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        return self


# ---------------------------------------------------------------------------
# Transfer search helpers / 迁移搜索辅助
# ---------------------------------------------------------------------------


def _prepare_cifar10_data(
    config: TransferConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    加载 CIFAR-10 训练集为 numpy 数组（仅 Normalize，无增强）。
    与 search.py 的 _prepare_search_data 结构一致，但针对 CIFAR-10。

    Returns:
        X: shape (N, 3, 32, 32) float32, 归一化后
        y: shape (N,) int64
    """
    dataset = datasets.CIFAR10(
        root=str(config.data_root),
        train=True,
        download=True,
    )
    X = dataset.data.astype(np.float32) / 255.0
    X = X.transpose(0, 3, 1, 2)  # HWC → CHW

    mean = np.array(config.mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(config.std, dtype=np.float32).reshape(1, 3, 1, 1)
    X = (X - mean) / std

    y = np.array(dataset.targets, dtype=np.int64)
    return X, y


def _build_transfer_param_distributions(
    config: TransferConfig,
) -> dict:
    """
    构建迁移学习搜索参数空间。
    FC-only 微调：学习率范围比全量训练小。
    """
    return {
        "lr": loguniform(1e-4, 0.1),
        "optimizer__momentum": uniform(0.85, 0.14),
        "optimizer__weight_decay": loguniform(1e-6, 1e-2),
        "batch_size": list(config.batch_size_choices),
    }


# TrainConfig 有效字段集合，用于过滤搜索结果
_VALID_TRAIN_FIELDS = {f.name for f in dataclasses.fields(TrainConfig)}


def _save_transfer_search_results(
    searcher,
    config: TransferConfig,
) -> Path:
    """
    保存迁移学习搜索结果为 JSON。
    """
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_dir / "transfer_hp_search_results.json"

    cv_results = searcher.cv_results_
    all_candidates = []
    for i in range(len(cv_results["mean_test_score"])):
        params = {
            k: (
                float(v[i]) if isinstance(v[i], (np.floating, float))
                else int(v[i]) if isinstance(v[i], (np.integer, int))
                else str(v[i])
            )
            for k, v in cv_results.items()
            if k.startswith("param_")
        }
        all_candidates.append({
            "params": params,
            "mean_test_score": round(
                float(cv_results["mean_test_score"][i]), 6
            ),
            "std_test_score": round(float(cv_results["std_test_score"][i]), 6),
            "mean_fit_time": round(float(cv_results["mean_fit_time"][i]), 2),
            "rank": int(cv_results["rank_test_score"][i]),
        })

    best_params = {
        k: (
            float(v) if isinstance(v, (np.floating, float))
            else int(v) if isinstance(v, (np.integer, int))
            else str(v)
        )
        for k, v in searcher.best_params_.items()
    }

    results = OrderedDict([
        ("search_config", {
            "strategy": "halving-random",
            "total_candidates": len(all_candidates),
            "cv": config.cv,
            "source_checkpoint": str(config.source_checkpoint),
        }),
        ("best", OrderedDict([
            ("params", best_params),
            ("mean_test_score", round(float(searcher.best_score_), 6)),
        ])),
        ("all_candidates", all_candidates),
    ])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return path


# skorch 参数名 → TransferConfig 字段名映射
_TRANSFER_PARAM_MAP: dict[str, str] = {
    "lr": "learning_rate",
    "optimizer__momentum": "momentum",
    "optimizer__weight_decay": "weight_decay",
    "batch_size": "batch_size",
}


# ---------------------------------------------------------------------------
# Public API / 公共接口
# ---------------------------------------------------------------------------


def run_transfer_search(
    config: TransferConfig,
) -> dict[str, object]:
    """
    迁移学习超参搜索。

    用 _TransferNetClassifier 包装 ResNet-18（每次初始化自动加载
    预训练权重并冻结 backbone），委托 HalvingRandomSearchCV 搜索
    最优 FC-only 微调超参数。

    Returns:
        映射后的最优参数字典（TransferConfig 字段名）。
    """
    print("=" * 60)
    print("Transfer Learning Hyperparameter Search")
    print("迁移学习超参数搜索")
    print("=" * 60)
    print(f"  Source: {config.source_checkpoint}")
    print(f"  CV folds: {config.cv}")
    print()

    # 准备 CIFAR-10 数据 / Prepare CIFAR-10 data
    print("Loading CIFAR-10 for search / 加载搜索用数据...")
    X, y = _prepare_cifar10_data(config)
    print(f"  X: {X.shape} {X.dtype}")
    print(f"  y: {y.shape} {y.dtype}")

    # 创建 skorch 网络 / Create skorch net
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    net = _TransferNetClassifier(
        ResNet18,
        module__num_classes=config.source_num_classes,
        module__source_checkpoint=str(config.source_checkpoint),
        module__target_num_classes=config.num_classes,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        lr=config.learning_rate,
        optimizer__momentum=config.momentum,
        optimizer__weight_decay=config.weight_decay,
        max_epochs=config.search_epochs_min,
        batch_size=config.batch_size,
        iterator_train__shuffle=True,
        iterator_train__num_workers=config.num_workers,
        iterator_valid__num_workers=config.num_workers,
        train_split=False,
        device=device,
        verbose=0,
    )

    # 构建搜索器 / Build searcher
    param_dist = _build_transfer_param_distributions(config)
    searcher = HalvingRandomSearchCV(
        net,
        param_dist,
        resource="max_epochs",
        max_resources=config.search_epochs_max,
        min_resources=config.search_epochs_min,
        factor=config.halving_factor,
        cv=config.cv,
        scoring="accuracy",
        refit=False,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )

    # 执行搜索 / Run search
    print("\nStarting transfer search / 开始迁移搜索...")
    searcher.fit(X, y)

    # 保存结果 / Save results
    results_path = _save_transfer_search_results(searcher, config)

    # 输出摘要 / Print summary
    print("\n" + "=" * 60)
    print("Transfer Search Complete / 迁移搜索完成")
    print("=" * 60)
    print(f"  Best score: {searcher.best_score_:.4f}")
    print("  Best params (raw):")
    for k, v in searcher.best_params_.items():
        print(f"    {k}: {v}")
    print(f"\n  Results saved to: {results_path}")

    # 映射并过滤参数 / Map and filter params
    raw_params = searcher.best_params_
    mapped = {
        _TRANSFER_PARAM_MAP.get(k, k): v
        for k, v in raw_params.items()
    }
    return {
        k: v for k, v in mapped.items()
        if k in _VALID_TRAIN_FIELDS
    }


def _to_train_config(config: TransferConfig) -> TrainConfig:
    """
    将 TransferConfig 转为 TrainConfig（取同名字段）。

    TransferConfig 和 TrainConfig 有大量同名字段（num_classes,
    batch_size, epochs, learning_rate 等），直接映射即可。
    source_checkpoint, source_num_classes, search_* 等独有字段被忽略。
    """
    train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    overrides = {
        f: getattr(config, f)
        for f in train_fields
        if hasattr(config, f)
    }
    return TrainConfig(**overrides)


def run_transfer(
    config: TransferConfig,
    search: bool = False,
) -> dict[str, list[float]]:
    """
    迁移学习主流程。

    流程：
    1. (可选) 超参搜索 → 更新 config
    2. 加载预训练模型 → 冻结 backbone → FC 随机初始化
    3. 构造 TrainConfig，加载 CIFAR-10
    4. train() 训练（仅 FC 层）
    5. 保存检查点和训练历史

    Args:
        config: 迁移学习配置
        search: 是否先运行超参搜索

    Returns:
        训练历史 dict
    """
    print("=" * 60)
    print("Transfer Learning: CIFAR-100 → CIFAR-10")
    print("迁移学习：CIFAR-100 → CIFAR-10")
    print("=" * 60)
    print(f"  Source: {config.source_checkpoint}")
    print(f"  Target classes: {config.num_classes}")
    print()

    # ---- 可选：超参搜索 / Optional: hyperparameter search ----
    if search:
        best_params = run_transfer_search(config)
        config = dataclasses.replace(config, **best_params)
        print(
            "\nUsing best params from transfer search"
            " / 使用迁移搜索得到的最佳配置:"
        )
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    # ---- 加载预训练模型 / Load pretrained model ----
    print("\nLoading pretrained model / 加载预训练模型...")
    model = load_pretrained_model(
        config.source_checkpoint,
        source_num_classes=config.source_num_classes,
        target_num_classes=config.num_classes,
    )
    freeze_backbone(model)
    print_transfer_summary(model)

    # ---- 构造 TrainConfig 并加载数据 ----
    train_config = _to_train_config(config)
    train_loader, test_loader = get_cifar10_loaders(train_config)
    print(
        f"\nCIFAR-10: Train {len(train_loader.dataset)}"
        f" | Test {len(test_loader.dataset)} samples"
    )

    # ---- 训练 / Train ----
    print(
        f"\nStarting transfer training for {config.epochs} epochs"
        f" / 开始迁移训练 {config.epochs} 轮..."
    )
    history = train(model, train_loader, test_loader, train_config)

    # ---- 保存 / Save ----
    save_training_history(history, train_config)

    print(
        "\nTransfer training complete."
        " / 迁移训练完成。"
    )
    return history
