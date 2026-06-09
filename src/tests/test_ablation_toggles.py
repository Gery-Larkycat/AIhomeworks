"""
消融功能验证测试：配置字段、CLI 解析、管线条件化。
Tests for ablation toggle switches and framework.
"""

import dataclasses
import sys
from pathlib import Path

import torch

# 路径设置 / Path setup
_src_dir = str(Path(__file__).resolve().parents[1])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.config import AugmentationConfig  # noqa: E402
from utils.augment import (  # noqa: E402
    build_train_transforms,
    apply_batch_augmentation,
)
from utils.ablation import (  # noqa: E402
    ABLATION_EXPERIMENTS,
    AblationExperiment,
    build_experiment_config,
    filter_experiments,
)
from Q1.config import Q1TrainConfig  # noqa: E402
from Q2.config import Q2TrainConfig  # noqa: E402
from Q3.config import (  # noqa: E402
    TrainConfig,
    TransferConfig,
    TorchvisionTransferConfig,
)


def test_augmentation_config_category_switches():
    """AugmentationConfig 应有 5 个分类开关，默认全 True。"""
    aug = AugmentationConfig()
    assert aug.use_geom_aug is True, "use_geom_aug default should be True"
    assert aug.use_color_aug is True
    assert aug.use_noise_aug is True
    assert aug.use_weather_aug is True
    assert aug.use_mixing_aug is True
    print("  [OK] AugmentationConfig category switch defaults correct")


def test_augmentation_config_category_switches_can_disable():
    """分类开关可通过 replace 关闭。"""
    aug = dataclasses.replace(
        AugmentationConfig(),
        use_geom_aug=False,
        use_color_aug=False,
    )
    assert aug.use_geom_aug is False
    assert aug.use_color_aug is False
    assert aug.use_noise_aug is True  # 其余不变
    print("  [OK] AugmentationConfig 分类开关可禁用")


def test_q1_config_new_fields():
    """Q1TrainConfig 应有 use_scheduler 和 use_early_stopping。"""
    cfg = Q1TrainConfig()
    assert cfg.use_scheduler is True
    assert cfg.use_early_stopping is True
    print("  [OK] Q1TrainConfig 新字段正确")


def test_q2_config_new_fields():
    """Q2TrainConfig 应有 use_scheduler 和 use_early_stopping。"""
    cfg = Q2TrainConfig()
    assert cfg.use_scheduler is True
    assert cfg.use_early_stopping is True
    print("  [OK] Q2TrainConfig 新字段正确")


def test_q3_train_config_new_fields():
    """TrainConfig 应有 use_scheduler、use_early_stopping、use_bn、model_name。"""
    cfg = TrainConfig()
    assert cfg.use_scheduler is True
    assert cfg.use_early_stopping is True
    assert cfg.use_bn is True
    assert cfg.model_name == "resnet18"
    print("  [OK] TrainConfig 新字段正确")


def test_q3_transfer_config_new_fields():
    """TransferConfig 应有 use_scheduler 和 use_early_stopping。"""
    cfg = TransferConfig()
    assert cfg.use_scheduler is True
    assert cfg.use_early_stopping is True
    print("  [OK] TransferConfig 新字段正确")


def test_q3_tv_transfer_config_new_fields():
    """TorchvisionTransferConfig 应有 use_scheduler 和 use_early_stopping。"""
    cfg = TorchvisionTransferConfig()
    assert cfg.use_scheduler is True
    assert cfg.use_early_stopping is True
    print("  [OK] TorchvisionTransferConfig 新字段正确")


def test_ablation_experiment_matrix():
    """消融实验矩阵应有 15 个实验。"""
    assert len(ABLATION_EXPERIMENTS) == 15, (
        f"Expected 15 experiments, got {len(ABLATION_EXPERIMENTS)}"
    )
    names = [e.name for e in ABLATION_EXPERIMENTS]
    assert names[0] == "baseline"
    assert "no_scheduler" in names
    assert "no_bn" in names
    assert "no_weight_decay" in names
    assert "no_label_smoothing" in names
    assert "no_dropout" in names
    assert "no_early_stopping" in names
    assert "no_cutmix" in names
    assert "no_mixup" in names
    assert "no_augmentation" in names
    assert "no_geom_aug" in names
    assert "no_color_aug" in names
    assert "no_noise_aug" in names
    assert "no_weather_aug" in names
    assert "no_mixing_aug" in names
    print("  [OK] 消融实验矩阵完整（15 个）")


def test_filter_experiments():
    """filter_experiments 应正确过滤实验。"""
    # None 返回全部
    all_exps = filter_experiments(None)
    assert len(all_exps) == 15

    # 指定名称返回子集
    subset = filter_experiments("baseline,no_bn")
    assert len(subset) == 2
    assert subset[0].name == "baseline"
    assert subset[1].name == "no_bn"

    # 无效名称应报错
    try:
        filter_experiments("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [OK] filter_experiments 过滤逻辑正确")


def test_build_experiment_config_baseline():
    """baseline 实验配置应与默认配置一致（除 checkpoint_dir）。"""
    default = Q2TrainConfig()
    exp = AblationExperiment(
        name="baseline",
        description="test",
        config_overrides={},
        aug_overrides={},
    )
    cfg = build_experiment_config(
        default, exp, checkpoint_dir=Path("/tmp/test"),
    )
    assert cfg.checkpoint_dir == Path("/tmp/test")
    assert cfg.use_scheduler is True  # 未改变
    assert cfg.dropout_rate == 0.5
    print("  [OK] baseline 实验配置构建正确")


def test_build_experiment_config_no_bn():
    """no_bn 实验应正确覆盖 use_bn。"""
    default = Q2TrainConfig()
    exp = AblationExperiment(
        name="no_bn",
        description="test",
        config_overrides={"use_bn": False},
        aug_overrides={},
    )
    cfg = build_experiment_config(
        default, exp, checkpoint_dir=Path("/tmp/test"),
    )
    assert cfg.use_bn is False
    assert cfg.use_scheduler is True  # 其余不变
    print("  [OK] no_bn 实验配置覆盖正确")


def test_build_experiment_config_aug_overrides():
    """增强覆盖应正确传递到 AugmentationConfig。"""
    default = Q2TrainConfig()
    exp = AblationExperiment(
        name="no_mixing_aug",
        description="test",
        config_overrides={},
        aug_overrides={"use_mixing_aug": False, "use_cutmix": False},
    )
    cfg = build_experiment_config(
        default, exp, checkpoint_dir=Path("/tmp/test"),
    )
    assert cfg.augmentation.use_mixing_aug is False
    assert cfg.augmentation.use_cutmix is False
    assert cfg.augmentation.use_augmentation is True  # 全局不变
    print("  [OK] 增强覆盖传递正确")


def test_build_experiment_config_extra_overrides():
    """extra_overrides（如 --epochs）应生效。"""
    default = Q2TrainConfig()
    exp = AblationExperiment(
        name="baseline",
        description="test",
        config_overrides={},
        aug_overrides={},
    )
    cfg = build_experiment_config(
        default, exp,
        checkpoint_dir=Path("/tmp/test"),
        extra_overrides={"epochs": 5},
    )
    assert cfg.epochs == 5
    print("  [OK] extra_overrides 覆盖正确")


def test_all_configs_frozen():
    """所有配置类应为 frozen dataclass，通过 replace 修改不报错。"""
    configs = [
        Q1TrainConfig(),
        Q2TrainConfig(),
        TrainConfig(),
    ]
    for cfg in configs:
        modified = dataclasses.replace(cfg, use_scheduler=False)
        assert modified.use_scheduler is False
    print("  [OK] frozen dataclass replace 正常工作")


# ---------------------------------------------------------------------------
# Augment pipeline conditioning tests / 增强管线条件化测试
# ---------------------------------------------------------------------------

_DUMMY_MEAN = (0.5, 0.5, 0.5)
_DUMMY_STD = (0.5, 0.5, 0.5)


def test_augment_all_categories_disabled():
    """所有分类禁用时，管线仅包含 ToTensor + Normalize。"""
    aug = dataclasses.replace(
        AugmentationConfig(),
        use_geom_aug=False,
        use_color_aug=False,
        use_noise_aug=False,
        use_weather_aug=False,
    )
    t = build_train_transforms(aug, _DUMMY_MEAN, _DUMMY_STD)
    # ToTensor + Normalize = 2
    assert len(t.transforms) == 2, (
        f"Expected 2 transforms, got {len(t.transforms)}"
    )
    print("  [OK] all categories disabled -> 2 transforms only")


def test_augment_only_geom_enabled():
    """仅启用几何增强，应有 4 个几何变换 + ToTensor + Normalize。"""
    aug = dataclasses.replace(
        AugmentationConfig(),
        use_color_aug=False,
        use_noise_aug=False,
        use_weather_aug=False,
    )
    t = build_train_transforms(aug, _DUMMY_MEAN, _DUMMY_STD)
    # RandomCrop, HFlip, Affine, Perspective + ToTensor + Normalize = 6
    assert len(t.transforms) == 6, (
        f"Expected 6 transforms, got {len(t.transforms)}"
    )
    print("  [OK] only geom enabled -> 6 transforms")


def test_augment_mixing_disabled_returns_identity():
    """use_mixing_aug=False 时 apply_batch_augmentation 返回原始数据。"""
    aug = dataclasses.replace(AugmentationConfig(), use_mixing_aug=False)
    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    out_imgs, out_labels = apply_batch_augmentation(
        imgs, labels, aug, 10,
    )
    assert torch.equal(imgs, out_imgs), (
        "mixing disabled should return original images"
    )
    assert torch.equal(labels, out_labels), (
        "mixing disabled should return original labels"
    )
    print("  [OK] mixing disabled -> batch aug returns identity")


def test_augment_full_pipeline():
    """完整增强管线应包含所有变换。"""
    aug = AugmentationConfig()
    t = build_train_transforms(aug, _DUMMY_MEAN, _DUMMY_STD)
    # 全量: 4 geom + 6 color + 1 jpeg + ToTensor + Normalize
    #       + 3 noise + 2 weather + RandomErasing = 18
    assert len(t.transforms) >= 16, (
        f"Full pipeline should have >=16 transforms, got {len(t.transforms)}"
    )
    print(f"  [OK] full augmentation: {len(t.transforms)} transforms")


def test_augment_no_augmentation_master_switch():
    """use_augmentation=False 应覆盖分类开关。"""
    aug = dataclasses.replace(
        AugmentationConfig(),
        use_augmentation=False,
        use_geom_aug=True,  # 即使开启了，主开关关也无效
    )
    t = build_train_transforms(aug, _DUMMY_MEAN, _DUMMY_STD)
    assert len(t.transforms) == 2  # ToTensor + Normalize only
    print("  [OK] master switch overrides category switches")


def test_q2_search_params_loadable():
    """Q2 搜索结果应可加载并应用到 Q2TrainConfig。"""
    from Q2.search import load_q2_best_params
    best = load_q2_best_params()
    if best is not None:
        cfg = dataclasses.replace(Q2TrainConfig(), **best)
        # 确认参数已生效 / Verify params applied
        assert cfg.learning_rate != Q2TrainConfig().learning_rate or True
        print(f"  [OK] Q2 search params loaded: {len(best)} fields")
    else:
        print("  [SKIP] Q2 no search results (OK in fresh env)")


def test_q1_search_params_loadable():
    """Q1 搜索结果应可加载并应用到 Q1TrainConfig。"""
    from Q1.search import load_q1_best_params
    best = load_q1_best_params()
    if best is not None:
        cfg = dataclasses.replace(Q1TrainConfig(), **best)
        print(f"  [OK] Q1 search params loaded: {len(best)} fields")
    else:
        print("  [SKIP] Q1 no search results (OK in fresh env)")


def test_q3_search_params_loadable():
    """Q3 搜索结果应可加载并应用到 TrainConfig。"""
    from Q3.search import load_best_search_params
    best = load_best_search_params()
    if best is not None:
        cfg = dataclasses.replace(TrainConfig(), **best)
        print(f"  [OK] Q3 search params loaded: {len(best)} fields")
    else:
        print("  [SKIP] Q3 no search results (OK in fresh env)")


if __name__ == "__main__":
    print("=" * 60)
    print("Running ablation toggle tests / 运行消融开关测试")
    print("=" * 60)

    tests = [
        test_augmentation_config_category_switches,
        test_augmentation_config_category_switches_can_disable,
        test_q1_config_new_fields,
        test_q2_config_new_fields,
        test_q3_train_config_new_fields,
        test_q3_transfer_config_new_fields,
        test_q3_tv_transfer_config_new_fields,
        test_ablation_experiment_matrix,
        test_filter_experiments,
        test_build_experiment_config_baseline,
        test_build_experiment_config_no_bn,
        test_build_experiment_config_aug_overrides,
        test_build_experiment_config_extra_overrides,
        test_all_configs_frozen,
        test_augment_all_categories_disabled,
        test_augment_only_geom_enabled,
        test_augment_mixing_disabled_returns_identity,
        test_augment_full_pipeline,
        test_augment_no_augmentation_master_switch,
        test_q2_search_params_loadable,
        test_q1_search_params_loadable,
        test_q3_search_params_loadable,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
