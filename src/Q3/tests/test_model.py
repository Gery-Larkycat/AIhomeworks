"""
Tests for ResNet-18 model implementation.
ResNet-18 模型实现的测试。

Validates: output shape, parameter count, gradient flow, feature extractor extraction.
验证：输出形状、参数量、梯度流、特征提取器提取。
"""

import torch

from src.Q2.model import ResNet18, create_model, get_feature_extractor_state


def test_output_shape() -> None:
    """Model output shape should be (batch_size, num_classes)."""
    model = create_model(num_classes=100)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 100), f"Expected (4, 100), got {out.shape}"


def test_different_num_classes() -> None:
    """Model should work with different num_classes (e.g. CIFAR-10=10)."""
    model = create_model(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"


def test_parameter_count() -> None:
    """
    ResNet-18 should have ~11M parameters.
    ResNet-18 应有约 1100 万参数。
    """
    model = create_model(num_classes=100)
    num_params = sum(p.numel() for p in model.parameters())
    # ~11.2M for standard ResNet-18 with CIFAR stem
    assert 10_000_000 < num_params < 12_000_000, f"Unexpected param count: {num_params}"


def test_gradient_flow() -> None:
    """All parameters should receive gradients after backward pass."""
    model = create_model(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 100, (2,))
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_feature_extractor_extraction() -> None:
    """
    Feature extractor state should contain no FC layer keys.
    特征提取器 state 不应包含 FC 层的 key。
    """
    model = create_model(num_classes=100)
    feature_state = get_feature_extractor_state(model)

    assert len(feature_state) > 0, "Feature extractor state is empty"
    for key in feature_state:
        assert not key.startswith("fc."), f"Found FC key in feature extractor: {key}"


def test_feature_extractor_loadable() -> None:
    """
    Feature extractor weights should be loadable into a new model with different num_classes.
    特征提取器权重应可加载到具有不同 num_classes 的新模型中。
    """
    # Train on CIFAR-100 (100 classes) / 在 CIFAR-100（100 类）上训练
    model_100 = create_model(num_classes=100)
    feature_state = get_feature_extractor_state(model_100)

    # Transfer to CIFAR-10 (10 classes) / 迁移到 CIFAR-10（10 类）
    model_10 = create_model(num_classes=10)
    # strict=False allows FC layer to remain randomly initialized
    # strict=False 允许 FC 层保持随机初始化
    model_10.load_state_dict(feature_state, strict=False)

    x = torch.randn(2, 3, 32, 32)
    out = model_10(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"


def test_spatial_dimensions() -> None:
    """
    Verify feature map size through each layer group.
    验证每层组的特征图尺寸。

    Input: 32x32 → layer1: 32x32 → layer2: 16x16 → layer3: 8x8 → layer4: 4x4
    """
    model = create_model(num_classes=100)
    model.eval()
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        out = model.conv1(x)
        out = model.bn1(out)
        out = model.relu(out)
        assert out.shape[2:] == (32, 32), f"After stem: {out.shape}"

        out = model.layer1(out)
        assert out.shape[2:] == (32, 32), f"After layer1: {out.shape}"

        out = model.layer2(out)
        assert out.shape[2:] == (16, 16), f"After layer2: {out.shape}"

        out = model.layer3(out)
        assert out.shape[2:] == (8, 8), f"After layer3: {out.shape}"

        out = model.layer4(out)
        assert out.shape[2:] == (4, 4), f"After layer4: {out.shape}"


# ---------------------------------------------------------------------------
# Dropout tests / Dropout 测试
# ---------------------------------------------------------------------------


def test_dropout_default_is_identity() -> None:
    """
    dropout_rate=0 时 Dropout 等价于恒等变换（eval 模式下）。
    With dropout_rate=0, Dropout is identity in eval mode.
    """
    model = create_model(num_classes=100, dropout_rate=0.0)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 100)


def test_dropout_with_rate() -> None:
    """
    dropout_rate=0.5 时模型可正常前向传播。
    Model forward pass works with dropout_rate=0.5.
    """
    model = create_model(num_classes=100, dropout_rate=0.5)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 100)


def test_dropout_train_mode_actually_drops() -> None:
    """
    dropout_rate=0.5 时，训练模式下多次前向传播结果不完全相同。
    With dropout_rate=0.5, multiple forward passes in train mode differ.
    """
    model = create_model(num_classes=100, dropout_rate=0.5)
    model.train()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    # 极大概率不相等（Dropout 随机丢弃不同神经元）
    assert not torch.allclose(out1, out2), (
        "Dropout should produce different outputs in train mode"
    )


def test_dropout_module_exists() -> None:
    """
    验证模型包含 Dropout 模块且概率正确。
    Verify model contains Dropout module with correct probability.
    """
    import torch.nn as nn

    model = create_model(num_classes=100, dropout_rate=0.3)
    dropout_layers = [
        m for m in model.modules() if isinstance(m, nn.Dropout)
    ]
    assert len(dropout_layers) == 1, "Should have exactly one Dropout layer"
    assert dropout_layers[0].p == 0.3


def test_dropout_not_affecting_param_count() -> None:
    """
    Dropout 不改变参数量。
    Dropout does not change parameter count.
    """
    model_no_drop = create_model(num_classes=100, dropout_rate=0.0)
    model_with_drop = create_model(num_classes=100, dropout_rate=0.5)
    params_no = sum(p.numel() for p in model_no_drop.parameters())
    params_with = sum(p.numel() for p in model_with_drop.parameters())
    assert params_no == params_with
