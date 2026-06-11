"""
Q1 VGG-16 CIFAR-10 训练配置（向后兼容）。
Backward-compatible Q1 VGG-16 CIFAR-10 config.

实际配置类已统一到 utils.config.TrainConfig。
Q1 的差异化默认值通过 models.registry.make_config("Q1") 获取。

用法不变：
    from Q1.config import Q1TrainConfig
    config = Q1TrainConfig()           # 自动应用 Q1 默认值
    config = Q1TrainConfig(epochs=50)  # 覆盖指定字段
"""

from utils.config import TrainConfig, AugmentationConfig  # noqa: F401


def Q1TrainConfig(**kwargs):
    """
    向后兼容工厂函数。
    返回应用了 Q1 默认值的 TrainConfig 实例。

    行为与原 Q1TrainConfig frozen dataclass 一致：
    - Q1TrainConfig() → batch_size=256, model_name="vgg16", ...
    - Q1TrainConfig(epochs=50) → 覆盖 epochs
    - dataclasses.replace(Q1TrainConfig(), ...) → 正常工作
      （因为返回值是真正的 frozen dataclass 实例）
    """
    from models.registry import make_config
    return make_config("Q1", **kwargs)
