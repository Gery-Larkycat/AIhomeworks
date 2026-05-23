"""
Tests for timestamped run directories and dataset-aware filenames.
时间戳运行目录和数据集感知文件名测试。

Covers: generate_timestamp, make_run_dir, dataset_prefix,
save_best_checkpoint / save_feature_extractor with dynamic filenames.
覆盖：generate_timestamp、make_run_dir、dataset_prefix、
save_best_checkpoint / save_feature_extractor 动态文件名。
"""

import re

import pytest
import torch
from torch.optim import SGD

from src.Q3.config import (
    TrainConfig,
    generate_timestamp,
    make_run_dir,
    dataset_prefix,
)
from src.Q3.model import ResNet18


# ---------------------------------------------------------------------------
# generate_timestamp tests / 时间戳生成测试
# ---------------------------------------------------------------------------


class TestGenerateTimestamp:
    """Test generate_timestamp format."""

    def test_format_matches_pattern(self):
        """格式匹配 YYYY-MM-DD_HHMMSS。"""
        ts = generate_timestamp()
        pattern = r"^\d{4}-\d{2}-\d{2}_\d{6}$"
        assert re.match(pattern, ts), f"Bad format: {ts}"

    def test_no_colons(self):
        """Windows 安全：不含冒号。"""
        ts = generate_timestamp()
        assert ":" not in ts

    def test_length(self):
        """固定长度 17（YYYY-MM-DD_HHMMSS）。"""
        ts = generate_timestamp()
        assert len(ts) == 17


# ---------------------------------------------------------------------------
# make_run_dir tests / 运行目录构造测试
# ---------------------------------------------------------------------------


class TestMakeRunDir:
    """Test make_run_dir path construction."""

    def test_with_explicit_timestamp(self):
        """显式时间戳正确拼接。"""
        run_dir = make_run_dir(
            base=make_run_dir.__defaults__[0],
            timestamp="2026-05-24_143022",
        )
        assert run_dir.name == "2026-05-24_143022"

    def test_default_base(self):
        """默认 base 为 checkpoints。"""
        run_dir = make_run_dir(timestamp="2026-05-24_143022")
        assert run_dir.parent.name == "checkpoints"

    def test_custom_base(self, tmp_path):
        """自定义 base 路径。"""
        run_dir = make_run_dir(
            base=tmp_path, timestamp="2026-05-24_143022",
        )
        assert run_dir == tmp_path / "2026-05-24_143022"

    def test_auto_timestamp(self):
        """不传 timestamp 时自动生成。"""
        run_dir = make_run_dir()
        assert len(run_dir.name) == 17


# ---------------------------------------------------------------------------
# dataset_prefix tests / 数据集前缀测试
# ---------------------------------------------------------------------------


class TestDatasetPrefix:
    """Test dataset_prefix returns correct filename prefix."""

    def test_cifar100(self):
        """100 类 → resnet18_cifar100。"""
        assert dataset_prefix(100) == "resnet18_cifar100"

    def test_cifar10(self):
        """10 类 → resnet18_cifar10。"""
        assert dataset_prefix(10) == "resnet18_cifar10"

    def test_custom_classes(self):
        """自定义类别数 → resnet18_Ncls。"""
        assert dataset_prefix(20) == "resnet18_20cls"
        assert dataset_prefix(5) == "resnet18_5cls"


# ---------------------------------------------------------------------------
# Checkpoint filename tests / 检查点文件名测试
# ---------------------------------------------------------------------------


class TestTimestampedCheckpointFilenames:
    """Test save_best_checkpoint and save_feature_extractor
    use dataset-aware filenames."""

    def test_best_checkpoint_cifar10_name(self, tmp_path):
        """CIFAR-10 best checkpoint 文件名含 cifar10。"""
        from src.Q3.checkpoint import save_best_checkpoint

        config = TrainConfig(
            num_classes=10, checkpoint_dir=tmp_path,
        )
        model = ResNet18(num_classes=10)
        optimizer = SGD(model.parameters(), lr=0.01)

        path = save_best_checkpoint(
            model, optimizer, epoch=1, accuracy=0.5, config=config,
        )
        assert path.name == "resnet18_cifar10_best.pth"

    def test_best_checkpoint_cifar100_name(self, tmp_path):
        """CIFAR-100 best checkpoint 文件名含 cifar100。"""
        from src.Q3.checkpoint import save_best_checkpoint

        config = TrainConfig(
            num_classes=100, checkpoint_dir=tmp_path,
        )
        model = ResNet18(num_classes=100)
        optimizer = SGD(model.parameters(), lr=0.1)

        path = save_best_checkpoint(
            model, optimizer, epoch=1, accuracy=0.5, config=config,
        )
        assert path.name == "resnet18_cifar100_best.pth"

    def test_feature_extractor_cifar10_name(self, tmp_path):
        """CIFAR-10 feature extractor 文件名含 cifar10。"""
        from src.Q3.checkpoint import save_feature_extractor

        config = TrainConfig(
            num_classes=10, checkpoint_dir=tmp_path,
        )
        model = ResNet18(num_classes=10)

        path = save_feature_extractor(model, config)
        assert path.name == "resnet18_cifar10_feature_extractor.pth"

    def test_feature_extractor_custom_filename(self, tmp_path):
        """自定义 filename 参数仍生效。"""
        from src.Q3.checkpoint import save_feature_extractor

        config = TrainConfig(
            num_classes=10, checkpoint_dir=tmp_path,
        )
        model = ResNet18(num_classes=10)

        path = save_feature_extractor(
            model, config, filename="custom.pth",
        )
        assert path.name == "custom.pth"

    def test_files_in_subdirectory(self, tmp_path):
        """检查点存入时间戳子目录。"""
        from src.Q3.checkpoint import save_best_checkpoint

        run_dir = tmp_path / "2026-05-24_143022"
        config = TrainConfig(
            num_classes=100, checkpoint_dir=run_dir,
        )
        model = ResNet18(num_classes=100)
        optimizer = SGD(model.parameters(), lr=0.1)

        path = save_best_checkpoint(
            model, optimizer, epoch=1, accuracy=0.5, config=config,
        )
        assert path.parent == run_dir
        assert run_dir.exists()
