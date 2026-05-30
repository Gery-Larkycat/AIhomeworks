"""
Tests for VGG-16 model implementation.
VGG-16 模型实现的测试。

Validates: output shape, parameter count, gradient flow, feature extractor,
           dropout, use_bn toggle, spatial dimensions.
"""

import torch
import torch.nn as nn

from src.Q1.model import VGG16, create_model, get_feature_extractor_state


class TestOutputShape:
    """Model output shape should be (batch_size, num_classes)."""

    def test_cifar10(self):
        model = create_model(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_cifar100(self):
        model = create_model(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 100)


class TestParameterCount:
    """VGG-16 should have ~15M parameters (CIFAR-adapted, 2-layer FC)."""

    def test_param_count_range(self):
        model = create_model(num_classes=10)
        num_params = sum(p.numel() for p in model.parameters())
        # CIFAR-adapted VGG-16 with 2-layer FC(512→512→10) ≈ 15M
        assert 10_000_000 < num_params < 25_000_000, (
            f"Unexpected param count: {num_params}"
        )


class TestGradientFlow:
    """All parameters should receive gradients after backward pass."""

    def test_backward(self):
        model = create_model(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestFeatureExtractor:
    """Feature extractor state should contain no FC layer keys."""

    def test_no_fc_keys(self):
        model = create_model(num_classes=10)
        state = get_feature_extractor_state(model)
        assert len(state) > 0
        for key in state:
            assert not key.startswith("fc1."), f"Found fc1 key: {key}"
            assert not key.startswith("fc2."), f"Found fc2 key: {key}"

    def test_loadable_with_different_classes(self):
        model_10 = create_model(num_classes=10)
        state = get_feature_extractor_state(model_10)
        model_100 = create_model(num_classes=100)
        model_100.load_state_dict(state, strict=False)
        x = torch.randn(2, 3, 32, 32)
        out = model_100(x)
        assert out.shape == (2, 100)


class TestSpatialDimensions:
    """
    Feature map through conv layers with 3 MaxPool.
    32→16→8→4→4→4→AdaptiveAvgPool→1
    """

    def test_feature_map_sizes(self):
        model = create_model(num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model.features(x)
            # 3 MaxPool: 32→16→8→4, blocks 4/5 no MaxPool → 4
            assert out.shape[2:] == (4, 4), f"After features: {out.shape}"


class TestDropout:
    """Dropout behavior tests."""

    def test_eval_mode(self):
        model = create_model(num_classes=10, dropout_rate=0.5)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_train_mode_differs(self):
        model = create_model(num_classes=10, dropout_rate=0.5)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert not torch.allclose(out1, out2), "Dropout should vary outputs"

    def test_dropout_layers_count(self):
        model = create_model(num_classes=10, dropout_rate=0.3)
        dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 2
        assert dropout_layers[0].p == 0.3


class TestUseBN:
    """use_bn parameter toggles BatchNorm layers."""

    def test_bn_enabled_has_batchnorm(self):
        model = create_model(num_classes=10, use_bn=True)
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) > 0, "Should have BN layers when use_bn=True"

    def test_bn_disabled_no_batchnorm(self):
        model = create_model(num_classes=10, use_bn=False)
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) == 0, "Should have no BN layers when use_bn=False"

    def test_bn_disabled_forward_works(self):
        model = create_model(num_classes=10, use_bn=False)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)
