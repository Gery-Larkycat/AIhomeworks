"""
Q2 ResNet-18 训练管线（向后兼容重导出）。
Backward-compatible re-exports for Q2 ResNet-18 training pipeline.

实际训练实现已统一到 utils.pipeline.train_skorch。

Q2 和 Q3 共用此管线：
- Q2: CIFAR-10 训练（10 类）
- Q3: CIFAR-100 训练（100 类，save_feature_extractor=True）
"""

from models.resnet18 import ResNet18
from utils.pipeline import train_skorch


def train_resnet(config, train_dataset, test_dataset,
                 save_feature_extractor=False):
    """
    ResNet-18 skorch 训练管线。
    委托到 utils.pipeline.train_skorch，固定 model_class=ResNet18。

    Args / Returns: 与 utils.pipeline.train_skorch 相同。
    """
    return train_skorch(
        config, train_dataset, test_dataset,
        model_class=ResNet18,
        save_feature_extractor=save_feature_extractor,
    )
