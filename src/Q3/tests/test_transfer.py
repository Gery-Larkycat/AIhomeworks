"""
Tests for transfer learning: CIFAR-100 → CIFAR-10.
迁移学习测试：CIFAR-100 → CIFAR-10。

Covers: model loading, backbone freezing, TransferConfig,
optimizer filtering, _TransferNetClassifier, CIFAR-10 data loading.
覆盖：模型加载、backbone 冻结、TransferConfig、优化器过滤、
_TransferNetClassifier、CIFAR-10 数据加载。
"""

import dataclasses

import pytest
import torch
import torch.nn as nn

from src.Q3.config import TransferConfig, TrainConfig
from src.Q2.model import ResNet18
from src.Q3.transfer import (
    find_best_cifar100_checkpoint,
    freeze_backbone,
    load_pretrained_model,
    print_transfer_summary,
    _to_train_config,
)


# ---------------------------------------------------------------------------
# Load pretrained model tests / 加载预训练模型测试
# ---------------------------------------------------------------------------


class TestLoadPretrainedModel:
    """Test load_pretrained_model with synthetic checkpoint."""

    @pytest.fixture
    def checkpoint(self, tmp_path):
        """创建合成的 CIFAR-100 检查点。"""
        model = ResNet18(num_classes=100)
        path = tmp_path / "test_checkpoint.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "accuracy": 0.5,
        }, path)
        return path

    def test_fc_output_is_10(self, checkpoint):
        """新分类层输出维度为 10。"""
        model = load_pretrained_model(checkpoint)
        assert model.fc[-1].out_features == 10

    def test_fc_input_is_512(self, checkpoint):
        """原始 FC 输入维度仍为 512。"""
        model = load_pretrained_model(checkpoint)
        assert model.fc[0].in_features == 512

    def test_fc_is_sequential(self, checkpoint):
        """FC 变为 Sequential（原始 FC + 新分类层）。"""
        model = load_pretrained_model(checkpoint)
        assert isinstance(model.fc, nn.Sequential)
        assert len(model.fc) == 2

    def test_original_fc_preserved(self, checkpoint):
        """原始 FC 权重与源检查点一致。"""
        ckpt = torch.load(
            checkpoint, weights_only=False
        )
        source_state = ckpt["model_state_dict"]
        model = load_pretrained_model(checkpoint)
        assert torch.equal(
            model.fc[0].weight, source_state["fc.weight"]
        )
        assert torch.equal(
            model.fc[0].bias, source_state["fc.bias"]
        )

    def test_backbone_weights_preserved(self, checkpoint):
        """非 FC 权重与源检查点一致。"""
        ckpt = torch.load(
            checkpoint, weights_only=False
        )
        source_state = ckpt["model_state_dict"]
        model = load_pretrained_model(checkpoint)

        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                assert torch.equal(
                    param, source_state[name]
                ), f"Backbone weight mismatch: {name}"

    def test_new_classifier_random(self, checkpoint):
        """新分类层（fc[1]）权重是随机初始化的。"""
        model = load_pretrained_model(checkpoint)
        # fc[1] 是新层 Linear(100, 10)，权重不应全零
        assert model.fc[1].weight.abs().sum() > 0

    def test_custom_target_classes(self, checkpoint):
        """可自定义目标类别数。"""
        model = load_pretrained_model(
            checkpoint, target_num_classes=20,
        )
        assert model.fc[-1].out_features == 20


# ---------------------------------------------------------------------------
# Freeze backbone tests / 冻结 backbone 测试
# ---------------------------------------------------------------------------


class TestFreezeBackbone:
    """Test freeze_backbone correctly freezes non-FC params."""

    def test_fc_requires_grad_true(self):
        """FC 参数保持 requires_grad=True。"""
        model = ResNet18(num_classes=10)
        freeze_backbone(model)
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                assert param.requires_grad, f"{name} should be trainable"

    def test_non_fc_requires_grad_false(self):
        """非 FC 参数 requires_grad=False。"""
        model = ResNet18(num_classes=10)
        freeze_backbone(model)
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                assert not param.requires_grad, f"{name} should be frozen"

    def test_trainable_param_count_transfer_model(self, tmp_path):
        """迁移模型冻结后可训练参数 = 原 FC + 新分类层 = 52310。"""
        # 创建合成检查点
        model = ResNet18(num_classes=100)
        path = tmp_path / "test_checkpoint.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "accuracy": 0.5,
        }, path)
        # 加载迁移模型并冻结
        transfer_model = load_pretrained_model(path)
        freeze_backbone(transfer_model)
        trainable = sum(
            p.numel() for p in transfer_model.parameters()
            if p.requires_grad
        )
        # 原 FC: 512*100 + 100 = 51300
        # 新分类层: 100*10 + 10 = 1010
        assert trainable == 512 * 100 + 100 + 100 * 10 + 10


# ---------------------------------------------------------------------------
# TransferConfig tests / 迁移配置测试
# ---------------------------------------------------------------------------


class TestTransferConfig:
    """Test TransferConfig defaults and frozen behavior."""

    def test_default_num_classes(self):
        """默认目标类别数为 10。"""
        cfg = TransferConfig()
        assert cfg.num_classes == 10

    def test_default_learning_rate(self):
        """默认学习率 0.01（比全量训练低）。"""
        cfg = TransferConfig()
        assert cfg.learning_rate == 0.01

    def test_default_epochs(self):
        """默认 50 epochs。"""
        cfg = TransferConfig()
        assert cfg.epochs == 50

    def test_default_cifar10_stats(self):
        """默认使用 CIFAR-10 归一化统计量。"""
        from src.Q3.config import CIFAR10_MEAN, CIFAR10_STD
        cfg = TransferConfig()
        assert cfg.mean == CIFAR10_MEAN
        assert cfg.std == CIFAR10_STD

    def test_frozen(self):
        """TransferConfig 不可变。"""
        cfg = TransferConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.learning_rate = 0.1  # type: ignore[misc]

    def test_search_defaults(self):
        """搜索参数默认值合理（比 CIFAR-100 搜索短）。"""
        cfg = TransferConfig()
        assert cfg.search_epochs_min == 2
        assert cfg.search_epochs_max == 10
        assert cfg.num_trials == 30
        assert cfg.batch_size_choices == (64, 128, 256)


# ---------------------------------------------------------------------------
# _to_train_config tests / 配置转换测试
# ---------------------------------------------------------------------------


class TestToTrainConfig:
    """Test TransferConfig → TrainConfig conversion."""

    def test_produces_valid_train_config(self):
        """转换为有效的 TrainConfig。"""
        cfg = TransferConfig()
        train_cfg = _to_train_config(cfg)
        assert isinstance(train_cfg, TrainConfig)

    def test_overrides_preserved(self):
        """自定义字段值正确传递。"""
        cfg = dataclasses.replace(
            TransferConfig(),
            num_classes=10,
            learning_rate=0.005,
            epochs=20,
        )
        train_cfg = _to_train_config(cfg)
        assert train_cfg.num_classes == 10
        assert train_cfg.learning_rate == 0.005
        assert train_cfg.epochs == 20

    def test_cifar10_stats_transferred(self):
        """CIFAR-10 归一化统计量正确传递。"""
        from src.Q3.config import CIFAR10_MEAN, CIFAR10_STD
        cfg = TransferConfig()
        train_cfg = _to_train_config(cfg)
        assert train_cfg.mean == CIFAR10_MEAN
        assert train_cfg.std == CIFAR10_STD


# ---------------------------------------------------------------------------
# Optimizer filtering tests / 优化器过滤测试
# ---------------------------------------------------------------------------


class TestOptimizerFiltering:
    """Test create_optimizer only uses trainable params."""

    def test_optimizer_only_trainable_params(self):
        """optimizer 仅包含 requires_grad=True 的参数。"""
        from src.Q3.train import create_optimizer

        model = ResNet18(num_classes=10)
        freeze_backbone(model)
        config = TrainConfig()
        optimizer = create_optimizer(model, config)

        # 每个参数组的参数都应该是可训练的
        for group in optimizer.param_groups:
            for p in group["params"]:
                assert p.requires_grad

    def test_optimizer_param_count_frozen(self):
        """冻结后 optimizer 参数数 = FC 参数数（普通模型）。"""
        from src.Q3.train import create_optimizer

        model = ResNet18(num_classes=10)
        freeze_backbone(model)
        config = TrainConfig()
        optimizer = create_optimizer(model, config)

        total_params = sum(
            p.numel()
            for group in optimizer.param_groups
            for p in group["params"]
        )
        # 普通 ResNet18(10) 冻结后仅 FC 可训练
        assert total_params == 512 * 10 + 10

    def test_optimizer_param_count_transfer(self, tmp_path):
        """迁移模型冻结后 optimizer 参数数 = 原 FC + 新分类层。"""
        from src.Q3.train import create_optimizer

        # 创建合成检查点
        model = ResNet18(num_classes=100)
        path = tmp_path / "test_checkpoint.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "accuracy": 0.5,
        }, path)
        transfer_model = load_pretrained_model(path)
        freeze_backbone(transfer_model)
        config = TrainConfig()
        optimizer = create_optimizer(transfer_model, config)

        total_params = sum(
            p.numel()
            for group in optimizer.param_groups
            for p in group["params"]
        )
        # 原 FC: 512*100 + 100 = 51300
        # 新分类层: 100*10 + 10 = 1010
        assert total_params == 512 * 100 + 100 + 100 * 10 + 10

    def test_optimizer_all_params_unfrozen(self):
        """正常训练（未冻结）optimizer 包含全部参数。"""
        from src.Q3.train import create_optimizer

        model = ResNet18(num_classes=100)
        config = TrainConfig()
        optimizer = create_optimizer(model, config)

        total_params = sum(
            p.numel()
            for group in optimizer.param_groups
            for p in group["params"]
        )
        expected = sum(p.numel() for p in model.parameters())
        assert total_params == expected


# ---------------------------------------------------------------------------
# Print summary test / 打印摘要测试
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Test print_transfer_summary output."""

    def test_prints_without_error(self, tmp_path, capsys):
        """打印迁移模型摘要不报错。"""
        # 创建合成检查点
        model = ResNet18(num_classes=100)
        path = tmp_path / "test_checkpoint.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "accuracy": 0.5,
        }, path)
        transfer_model = load_pretrained_model(path)
        freeze_backbone(transfer_model)
        print_transfer_summary(transfer_model)

        captured = capsys.readouterr()
        assert "Frozen params:" in captured.out
        assert "Trainable params:" in captured.out
        # 原 FC(51300) + 新分类层(1010) = 52310
        assert "52,310" in captured.out


# ---------------------------------------------------------------------------
# CIFAR-10 data loading tests / CIFAR-10 数据加载测试
# ---------------------------------------------------------------------------


class TestCIFAR10Loaders:
    """Test CIFAR-10 data loading.

    These tests require CIFAR-10 dataset cached locally.
    They will be skipped if the dataset is not available.
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

    def test_cifar10_loaders_shapes(self, cifar10_available):
        """CIFAR-10 loader 返回正确 shape 的数据。"""
        from src.Q3.data import get_cifar10_loaders

        config = TransferConfig()
        train_loader, test_loader = get_cifar10_loaders(config)
        assert len(train_loader.dataset) == 50000
        assert len(test_loader.dataset) == 10000

    def test_cifar10_batch_shapes(self, cifar10_available):
        """CIFAR-10 batch 有正确的 tensor shape。"""
        from src.Q3.data import get_cifar10_loaders

        config = dataclasses.replace(TransferConfig(), batch_size=64)
        train_loader, _ = get_cifar10_loaders(config)
        images, labels = next(iter(train_loader))
        assert images.shape == (64, 3, 32, 32)
        assert labels.shape == (64,)
        assert labels.min() >= 0
        assert labels.max() <= 9


# ---------------------------------------------------------------------------
# Find best CIFAR-100 checkpoint tests / 自动发现最优基础模型测试
# ---------------------------------------------------------------------------


class TestFindBestCifar100Checkpoint:
    """Test find_best_cifar100_checkpoint accuracy-based selection."""

    def _make_checkpoint(
        self, directory, accuracy, filename="resnet18_cifar100_best.pth",
    ):
        """创建合成检查点文件，含 accuracy 字段。"""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        torch.save(
            {"epoch": 1, "accuracy": accuracy, "num_classes": 100},
            path,
        )
        return path

    def test_finds_in_timestamped_dir(self, tmp_path):
        """在时间戳子目录中找到检查点。"""
        ckpt = self._make_checkpoint(
            tmp_path / "2026-05-24_143022", accuracy=0.65,
        )
        result = find_best_cifar100_checkpoint(tmp_path)
        assert result == ckpt

    def test_selects_highest_accuracy(self, tmp_path):
        """选准确率最高的（而非最新的）。"""
        self._make_checkpoint(
            tmp_path / "2026-05-23_100000", accuracy=0.55,
        )
        best = self._make_checkpoint(
            tmp_path / "2026-05-24_143022", accuracy=0.72,
        )
        self._make_checkpoint(
            tmp_path / "2026-05-25_090000", accuracy=0.60,
        )
        result = find_best_cifar100_checkpoint(tmp_path)
        assert result == best

    def test_same_accuracy_picks_newest(self, tmp_path):
        """同准确率时选最新的。"""
        old = self._make_checkpoint(
            tmp_path / "2026-05-23_100000", accuracy=0.70,
        )
        new = self._make_checkpoint(
            tmp_path / "2026-05-24_143022", accuracy=0.70,
        )
        result = find_best_cifar100_checkpoint(tmp_path)
        assert result == new
        assert result != old

    def test_fallback_to_flat_dir(self, tmp_path):
        """回退到平面目录（无子目录时）。"""
        ckpt = self._make_checkpoint(
            tmp_path, accuracy=0.55,
        )
        result = find_best_cifar100_checkpoint(tmp_path)
        assert result == ckpt

    def test_returns_none_when_empty(self, tmp_path):
        """空目录返回 None。"""
        result = find_best_cifar100_checkpoint(tmp_path)
        assert result is None

    def test_returns_none_when_no_root(self, tmp_path):
        """不存在的根目录返回 None。"""
        result = find_best_cifar100_checkpoint(
            tmp_path / "nonexistent",
        )
        assert result is None

    def test_skips_dirs_without_checkpoint(self, tmp_path):
        """跳过不含检查点的目录。"""
        (tmp_path / "2026-05-24_143022").mkdir()
        good = self._make_checkpoint(
            tmp_path / "2026-05-25_090000", accuracy=0.68,
        )
        result = find_best_cifar100_checkpoint(tmp_path)
        assert result == good

    def test_flat_dir_vs_timestamped_picks_higher(self, tmp_path):
        """时间戳目录和平面目录比较准确率。"""
        self._make_checkpoint(
            tmp_path, accuracy=0.50,
        )
        best = self._make_checkpoint(
            tmp_path / "2026-05-24_143022", accuracy=0.75,
        )
        result = find_best_cifar100_checkpoint(tmp_path)
        assert result == best
