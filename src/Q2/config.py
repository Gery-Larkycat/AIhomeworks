"""
Q2 ResNet-18 CIFAR-10 训练配置（向后兼容）。
Backward-compatible Q2 ResNet-18 CIFAR-10 config.

实际配置类已统一到 utils.config.TrainConfig。
Q2 的差异化默认值通过 models.registry.make_config("Q2") 获取。

用法不变：
    from Q2.config import Q2TrainConfig
    config = Q2TrainConfig()           # 自动应用 Q2 默认值
    config = Q2TrainConfig(epochs=50)  # 覆盖指定字段
"""

from utils.config import TrainConfig, AugmentationConfig  # noqa: F401


def Q2TrainConfig(**kwargs):
    """
    向后兼容工厂函数。
    返回应用了 Q2 默认值的 TrainConfig 实例。

    行为与原 Q2TrainConfig frozen dataclass 一致：
    - Q2TrainConfig() → batch_size=128, model_name="resnet18", ...
    - Q2TrainConfig(epochs=50) → 覆盖 epochs
    - dataclasses.replace(Q2TrainConfig(), ...) → 正常工作
      （因为返回值是真正的 frozen dataclass 实例）
    """
    from models.registry import make_config
    return make_config("Q2", **kwargs)
