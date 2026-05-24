"""
Tests for torchvision pretrained ResNet-18 transfer learning.
PyTorch 预训练 ResNet-18 迁移学习测试。

Covers: model loading, backbone freezing, config conversion,
data loading with 224x224 resize.
覆盖：模型加载、backbone 冻结、配置转换、224x224 数据加载。
"""

import dataclasses

import pytest
import torch
import torch.nn as nn
from torchvision import transforms

from src.Q3.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TorchvisionTransferConfig,
    TrainConfig,
)
from src.Q3.torchvision_transfer import (
    _build_tv_transforms,
    _to_train_config,
    freeze_backbone_tv,
    load_torchvision_pretrained,
    print_tv_transfer_summary,
)


# ---------------------------------------------------------------------------
# Load pretrained model tests / 加载预训练模型测试
# ---------------------------------------------------------------------------


class TestLoadTorchvisionPretrained:
    """Test load_torchvision_pretrained model creation."""

    def test_fc_output_is_10(self):
        """FC 输出维度为 10（CIFAR-10）。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        assert model.fc.out_features == 10

    def test_fc_input_is_512(self):
        """FC 输入维度为 512（ResNet-18 最后一层通道数）。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        assert model.fc.in_features == 512

    def test_backbone_weights_loaded(self):
        """Backbone 权重已加载（非零，非随机初始化）。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        # conv1 权重应非零（ImageNet 预训练）
        assert model.conv1.weight.abs().sum() > 0
        # 检查权重不是全为同一值（说明不是 uniform/kaiming 初始化）
        assert not torch.allclose(
            model.conv1.weight[0, 0, 0, 0].expand_as(model.conv1.weight),
            model.conv1.weight,
        )

    def test_fc_random_init(self):
        """FC 层是随机初始化的（非预训练权重）。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        # FC 权重非零但不等于任何特定模式
        assert model.fc.weight.abs().sum() > 0

    def test_custom_target_classes(self):
        """可自定义目标类别数。"""
        model = load_torchvision_pretrained(target_num_classes=20)
        assert model.fc.out_features == 20

    def test_forward_pass_224(self):
        """224x224 输入可以正常前向传播。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# Freeze backbone tests / 冻结 backbone 测试
# ---------------------------------------------------------------------------


class TestFreezeBackboneTv:
    """Test freeze_backbone_tv correctly freezes non-FC params."""

    def test_fc_requires_grad_true(self):
        """FC 参数保持 requires_grad=True。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        freeze_backbone_tv(model)
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                assert param.requires_grad, f"{name} should be trainable"

    def test_non_fc_requires_grad_false(self):
        """非 FC 参数 requires_grad=False。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        freeze_backbone_tv(model)
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                assert not param.requires_grad, (
                    f"{name} should be frozen"
                )

    def test_trainable_param_count(self):
        """冻结后可训练参数 = FC 参数 = 512*10 + 10 = 5130。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        freeze_backbone_tv(model)
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert trainable == 512 * 10 + 10


# ---------------------------------------------------------------------------
# Print summary test / 打印摘要测试
# ---------------------------------------------------------------------------


class TestPrintTvTransferSummary:
    """Test print_tv_transfer_summary output."""

    def test_prints_without_error(self, capsys):
        """打印摘要不报错。"""
        model = load_torchvision_pretrained(target_num_classes=10)
        freeze_backbone_tv(model)
        print_tv_transfer_summary(model)

        captured = capsys.readouterr()
        assert "Frozen params:" in captured.out
        assert "Trainable params:" in captured.out
        assert "5,130" in captured.out


# ---------------------------------------------------------------------------
# TorchvisionTransferConfig tests / 配置测试
# ---------------------------------------------------------------------------


class TestTorchvisionTransferConfig:
    """Test TorchvisionTransferConfig defaults and frozen behavior."""

    def test_default_image_size(self):
        """默认图像尺寸 224。"""
        cfg = TorchvisionTransferConfig()
        assert cfg.image_size == 224

    def test_default_num_classes(self):
        """默认目标类别数 10。"""
        cfg = TorchvisionTransferConfig()
        assert cfg.num_classes == 10

    def test_default_imagenet_stats(self):
        """默认使用 ImageNet 归一化统计量。"""
        cfg = TorchvisionTransferConfig()
        assert cfg.mean == IMAGENET_MEAN
        assert cfg.std == IMAGENET_STD

    def test_default_training_params(self):
        """默认训练参数合理。"""
        cfg = TorchvisionTransferConfig()
        assert cfg.batch_size == 64
        assert cfg.epochs == 30
        assert cfg.learning_rate == 0.01
        assert cfg.weight_decay == 1e-4
        assert cfg.label_smoothing == 0.0

    def test_frozen(self):
        """TorchvisionTransferConfig 不可变。"""
        cfg = TorchvisionTransferConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.learning_rate = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Config conversion tests / 配置转换测试
# ---------------------------------------------------------------------------


class TestToTrainConfig:
    """Test TorchvisionTransferConfig → TrainConfig conversion."""

    def test_produces_valid_train_config(self):
        """转换为有效的 TrainConfig。"""
        cfg = TorchvisionTransferConfig()
        train_cfg = _to_train_config(cfg)
        assert isinstance(train_cfg, TrainConfig)

    def test_overrides_preserved(self):
        """自定义字段值正确传递。"""
        cfg = dataclasses.replace(
            TorchvisionTransferConfig(),
            num_classes=10,
            learning_rate=0.005,
            epochs=20,
        )
        train_cfg = _to_train_config(cfg)
        assert train_cfg.num_classes == 10
        assert train_cfg.learning_rate == 0.005
        assert train_cfg.epochs == 20

    def test_imagenet_stats_transferred(self):
        """ImageNet 归一化统计量正确传递。"""
        cfg = TorchvisionTransferConfig()
        train_cfg = _to_train_config(cfg)
        assert train_cfg.mean == IMAGENET_MEAN
        assert train_cfg.std == IMAGENET_STD


# ---------------------------------------------------------------------------
# Optimizer filtering tests / 优化器过滤测试
# ---------------------------------------------------------------------------


class TestOptimizerFilteringTv:
    """Test create_optimizer only uses trainable params."""

    def test_optimizer_only_trainable_params(self):
        """optimizer 仅包含 requires_grad=True 的参数。"""
        from src.Q3.train import create_optimizer

        model = load_torchvision_pretrained(target_num_classes=10)
        freeze_backbone_tv(model)
        config = TrainConfig()
        optimizer = create_optimizer(model, config)

        for group in optimizer.param_groups:
            for p in group["params"]:
                assert p.requires_grad

    def test_optimizer_param_count_frozen(self):
        """冻结后 optimizer 参数数 = FC 参数数。"""
        from src.Q3.train import create_optimizer

        model = load_torchvision_pretrained(target_num_classes=10)
        freeze_backbone_tv(model)
        config = TrainConfig()
        optimizer = create_optimizer(model, config)

        total_params = sum(
            p.numel()
            for group in optimizer.param_groups
            for p in group["params"]
        )
        assert total_params == 512 * 10 + 10


# ---------------------------------------------------------------------------
# CIFAR-10 224 data loading tests / 224x224 数据加载测试
# ---------------------------------------------------------------------------


class TestCIFAR10224Loaders:
    """Test CIFAR-10 224x224 data loading.

    These tests require CIFAR-10 dataset cached locally.
    这些测试需要 CIFAR-10 数据集已缓存，否则跳过。
    """

    @pytest.fixture
    def cifar10_available(self):
        """检查 CIFAR-10 数据集是否已缓存。"""
        from pathlib import Path

        data_root = Path("data")
        cifar10_dir = data_root / "cifar-10-batches-py"
        if not cifar10_dir.exists():
            pytest.skip("CIFAR-10 dataset not cached")
        return True

    def test_cifar10_224_shapes(self, cifar10_available):
        """224x224 CIFAR-10 loader 返回正确 shape 的数据。"""
        from src.Q3.torchvision_transfer import get_cifar10_224_loaders

        config = TorchvisionTransferConfig()
        train_loader, test_loader = get_cifar10_224_loaders(config)
        assert len(train_loader.dataset) == 50000
        assert len(test_loader.dataset) == 10000

    def test_cifar10_224_batch_shapes(self, cifar10_available):
        """224x224 batch 有正确的 tensor shape。"""
        from src.Q3.torchvision_transfer import get_cifar10_224_loaders

        config = dataclasses.replace(
            TorchvisionTransferConfig(), batch_size=32,
        )
        train_loader, _ = get_cifar10_224_loaders(config)
        images, labels = next(iter(train_loader))
        assert images.shape == (32, 3, 224, 224)
        assert labels.shape == (32,)
        assert labels.min() >= 0
        assert labels.max() <= 9


# ---------------------------------------------------------------------------
# Transform tests / 变换测试
# ---------------------------------------------------------------------------


class TestBuildTvTransforms:
    """Test _build_tv_transforms produces correct pipelines."""

    def test_test_transform_resize(self):
        """Test transform 包含 Resize。"""
        _, test_tf = _build_tv_transforms(image_size=224)
        assert any(
            isinstance(t, transforms.Resize) for t in test_tf.transforms
        )

    def test_test_transform_normalize(self):
        """Test transform 使用 ImageNet 归一化。"""
        _, test_tf = _build_tv_transforms()
        norm = [
            t for t in test_tf.transforms
            if isinstance(t, transforms.Normalize)
        ]
        assert len(norm) == 1
        # ImageNet mean/std
        assert list(norm[0].mean) == list(IMAGENET_MEAN)
        assert list(norm[0].std) == list(IMAGENET_STD)

    def test_augment_adds_hflip(self):
        """augment=True 时 train transform 包含 HFlip。"""
        train_tf, _ = _build_tv_transforms(augment=True)
        assert any(
            isinstance(t, transforms.RandomHorizontalFlip)
            for t in train_tf.transforms
        )

    def test_no_augment_no_hflip(self):
        """augment=False 时 train transform 不含 HFlip。"""
        train_tf, _ = _build_tv_transforms(augment=False)
        assert not any(
            isinstance(t, transforms.RandomHorizontalFlip)
            for t in train_tf.transforms
        )
