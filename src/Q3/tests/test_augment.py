"""
Unit tests for data augmentation module (augment.py).
数据增强模块单元测试。

Covers:
  - Custom tensor transforms: GaussianNoise, SaltPepperNoise,
    ProbabilisticGaussianBlur, FogEffect, RainStreaks
  - PIL-level transform: JPEGCompressionPIL
  - Batch augmentation: CutMix, Mixup, apply_batch_augmentation
  - Pipeline builder: build_train_transforms, build_test_transforms
"""

import random

import pytest
import torch
from PIL import Image

from utils.augment import (
    FogEffect,
    GaussianNoise,
    JPEGCompressionPIL,
    ProbabilisticGaussianBlur,
    RainStreaks,
    SaltPepperNoise,
    apply_batch_augmentation,
    build_test_transforms,
    build_train_transforms,
    cutmix_data,
    mixup_data,
)
from src.Q3.config import AugmentationConfig, TrainConfig, CIFAR100_MEAN, CIFAR100_STD


# ---------------------------------------------------------------------------
# Fixtures / 测试夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tensor():
    """Normalized CIFAR-shaped tensor (C=3, H=32, W=32)."""
    return torch.randn(3, 32, 32)


@pytest.fixture
def sample_pil():
    """Random PIL Image 32x32 RGB."""
    return Image.fromarray(
        torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy()
    )


@pytest.fixture
def sample_batch():
    """Batch of 8 CIFAR-shaped images + 8 integer labels."""
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 100, (8,))
    return images, labels


@pytest.fixture
def aug_config():
    """Default AugmentationConfig."""
    return AugmentationConfig()


@pytest.fixture
def train_config():
    """Default TrainConfig."""
    return TrainConfig()


# ===========================================================================
# C. Tensor-level transforms / Tensor 级变换
# ===========================================================================


class TestGaussianNoise:
    """GaussianNoise transform tests."""

    def test_shape_preserved(self, sample_tensor):
        t = GaussianNoise(std=0.02, p=1.0)(sample_tensor)
        assert t.shape == sample_tensor.shape

    def test_adds_noise_when_applied(self, sample_tensor):
        # Force apply with p=1.0, use large std for detectability
        original = sample_tensor.clone()
        t = GaussianNoise(std=1.0, p=1.0)(sample_tensor)
        assert not torch.allclose(t, original)

    def test_no_change_when_skipped(self, sample_tensor):
        original = sample_tensor.clone()
        t = GaussianNoise(std=0.02, p=0.0)(sample_tensor)
        assert torch.equal(t, original)


class TestSaltPepperNoise:
    """SaltPepperNoise transform tests."""

    def test_shape_preserved(self, sample_tensor):
        t = SaltPepperNoise(amount=0.01, p=1.0)(sample_tensor)
        assert t.shape == sample_tensor.shape

    def test_modifies_pixels_when_applied(self, sample_tensor):
        original = sample_tensor.clone()
        # High amount for visible effect
        t = SaltPepperNoise(amount=0.5, p=1.0)(sample_tensor)
        assert not torch.equal(t, original)

    def test_no_change_when_skipped(self, sample_tensor):
        original = sample_tensor.clone()
        t = SaltPepperNoise(amount=0.01, p=0.0)(sample_tensor)
        assert torch.equal(t, original)


class TestProbabilisticGaussianBlur:
    """ProbabilisticGaussianBlur transform tests."""

    def test_shape_preserved(self, sample_tensor):
        t = ProbabilisticGaussianBlur(kernel_size=3, p=1.0)(sample_tensor)
        assert t.shape == sample_tensor.shape

    def test_no_change_when_skipped(self, sample_tensor):
        original = sample_tensor.clone()
        t = ProbabilisticGaussianBlur(kernel_size=3, p=0.0)(sample_tensor)
        assert torch.equal(t, original)


class TestFogEffect:
    """FogEffect transform tests."""

    def test_shape_preserved(self, sample_tensor):
        t = FogEffect(intensity_range=(0.1, 0.3), p=1.0)(sample_tensor)
        assert t.shape == sample_tensor.shape

    def test_adds_fog_when_applied(self, sample_tensor):
        original = sample_tensor.clone()
        t = FogEffect(intensity_range=(0.5, 0.9), p=1.0)(sample_tensor)
        # Fog should make values closer to each other (higher min)
        assert not torch.equal(t, original)

    def test_no_change_when_skipped(self, sample_tensor):
        original = sample_tensor.clone()
        t = FogEffect(p=0.0)(sample_tensor)
        assert torch.equal(t, original)


class TestRainStreaks:
    """RainStreaks transform tests."""

    def test_shape_preserved(self, sample_tensor):
        t = RainStreaks(drops_range=(5, 10), p=1.0)(sample_tensor)
        assert t.shape == sample_tensor.shape

    def test_adds_streaks_when_applied(self, sample_tensor):
        original = sample_tensor.clone()
        t = RainStreaks(drops_range=(10, 20), p=1.0)(sample_tensor)
        assert not torch.equal(t, original)

    def test_no_change_when_skipped(self, sample_tensor):
        original = sample_tensor.clone()
        t = RainStreaks(p=0.0)(sample_tensor)
        assert torch.equal(t, original)


# ===========================================================================
# D. PIL-level transforms / PIL 级变换
# ===========================================================================


class TestJPEGCompressionPIL:
    """JPEGCompressionPIL transform tests."""

    def test_returns_pil(self, sample_pil):
        result = JPEGCompressionPIL(
            quality_range=(30, 70), p=1.0
        )(sample_pil)
        assert isinstance(result, Image.Image)

    def test_preserves_size(self, sample_pil):
        result = JPEGCompressionPIL(
            quality_range=(30, 70), p=1.0
        )(sample_pil)
        assert result.size == sample_pil.size

    def test_no_change_when_skipped(self, sample_pil):
        result = JPEGCompressionPIL(p=0.0)(sample_pil)
        # Same object returned (not modified)
        assert result is sample_pil


# ===========================================================================
# E. Batch augmentation / 批次级增强
# ===========================================================================


class TestCutMix:
    """CutMix tests."""

    def test_output_shapes(self, sample_batch):
        images, labels = sample_batch
        mixed_img, mixed_lbl = cutmix_data(
            images, labels, alpha=1.0, num_classes=100
        )
        assert mixed_img.shape == images.shape
        assert mixed_lbl.shape == (8, 100)

    def test_soft_labels_sum_to_one(self, sample_batch):
        images, labels = sample_batch
        _, mixed_lbl = cutmix_data(
            images, labels, alpha=1.0, num_classes=100
        )
        row_sums = mixed_lbl.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)

    def test_soft_labels_non_negative(self, sample_batch):
        images, labels = sample_batch
        _, mixed_lbl = cutmix_data(
            images, labels, alpha=1.0, num_classes=100
        )
        assert (mixed_lbl >= 0).all()

    def test_alpha_zero_returns_one_hot(self, sample_batch):
        images, labels = sample_batch
        _, mixed_lbl = cutmix_data(
            images, labels, alpha=0, num_classes=100
        )
        # Should be pure one-hot
        assert mixed_lbl.sum().item() == 8.0
        # Each row should have exactly one 1.0
        assert (mixed_lbl.argmax(dim=1) == labels).all()


class TestMixup:
    """Mixup tests."""

    def test_output_shapes(self, sample_batch):
        images, labels = sample_batch
        mixed_img, mixed_lbl = mixup_data(
            images, labels, alpha=0.2, num_classes=100
        )
        assert mixed_img.shape == images.shape
        assert mixed_lbl.shape == (8, 100)

    def test_soft_labels_sum_to_one(self, sample_batch):
        images, labels = sample_batch
        _, mixed_lbl = mixup_data(
            images, labels, alpha=0.2, num_classes=100
        )
        row_sums = mixed_lbl.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)

    def test_alpha_zero_returns_one_hot(self, sample_batch):
        images, labels = sample_batch
        _, mixed_lbl = mixup_data(
            images, labels, alpha=0, num_classes=100
        )
        assert mixed_lbl.sum().item() == 8.0
        assert (mixed_lbl.argmax(dim=1) == labels).all()


class TestApplyBatchAugmentation:
    """apply_batch_augmentation three-branch coverage."""

    def test_identity_when_both_disabled(self, sample_batch):
        images, labels = sample_batch
        cfg = AugmentationConfig(use_cutmix=False, use_mixup=False)
        out_img, out_lbl = apply_batch_augmentation(
            images, labels, cfg, 100
        )
        assert torch.equal(out_img, images)
        assert torch.equal(out_lbl, labels)

    def test_cutmix_or_mixup_or_identity(self, sample_batch):
        """Run many times to cover all three branches statistically."""
        images, labels = sample_batch
        cfg = AugmentationConfig(
            use_cutmix=True, use_mixup=True, mix_prob=0.7,
        )
        got_soft = False
        got_identity = False
        for _ in range(200):
            out_img, out_lbl = apply_batch_augmentation(
                images.clone(), labels.clone(), cfg, 100
            )
            if out_lbl.is_floating_point():
                got_soft = True
                assert out_lbl.shape == (8, 100)
            else:
                got_identity = True
                assert torch.equal(out_lbl, labels)
            if got_soft and got_identity:
                break
        assert got_soft, "Never got CutMix/Mixup in 200 tries"
        assert got_identity, "Never got identity in 200 tries"

    def test_augmentation_disabled(self, sample_batch):
        images, labels = sample_batch
        cfg = AugmentationConfig(use_augmentation=False)
        out_img, out_lbl = apply_batch_augmentation(
            images, labels, cfg, 100
        )
        assert torch.equal(out_img, images)
        assert torch.equal(out_lbl, labels)


# ===========================================================================
# Pipeline builder / 管线构建
# ===========================================================================


class TestBuildTrainTransforms:
    """build_train_transforms integration tests."""

    def test_full_pipeline_produces_tensor(self, train_config):
        transform = build_train_transforms(
            train_config.augmentation,
            train_config.mean, train_config.std,
        )
        img = Image.fromarray(
            torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy()
        )
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)

    def test_no_augmentation_pipeline(self, train_config):
        aug = AugmentationConfig(use_augmentation=False)
        transform = build_train_transforms(
            aug, train_config.mean, train_config.std,
        )
        img = Image.fromarray(
            torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy()
        )
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)

    def test_deterministic_output_with_seed(self, train_config):
        """Same seed → same output for non-random transforms."""
        aug = AugmentationConfig(
            use_augmentation=False,
        )
        transform = build_train_transforms(
            aug, train_config.mean, train_config.std,
        )
        img = Image.fromarray(
            torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy()
        )
        # Without augmentation, output is deterministic
        r1 = transform(img)
        r2 = transform(img)
        assert torch.allclose(r1, r2)


class TestBuildTestTransforms:
    """build_test_transforms tests."""

    def test_produces_normalized_tensor(self, train_config):
        transform = build_test_transforms(
            train_config.mean, train_config.std,
        )
        img = Image.fromarray(
            torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy()
        )
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)

    def test_no_augmentation_in_test(self, train_config):
        """Test pipeline should be deterministic for same input."""
        transform = build_test_transforms(
            train_config.mean, train_config.std,
        )
        img = Image.fromarray(
            torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8).numpy()
        )
        r1 = transform(img)
        r2 = transform(img)
        assert torch.allclose(r1, r2)
