"""
Q1 VGG-16 训练管线（向后兼容重导出）。
Backward-compatible re-exports for Q1 VGG-16 training pipeline.

实际训练实现已统一到 utils.pipeline.train_skorch。
"""

from models.vgg16 import VGG16
from utils.pipeline import train_skorch


def train_vgg(config, train_dataset, test_dataset):
    """
    VGG-16 skorch 训练管线。
    委托到 utils.pipeline.train_skorch，固定 model_class=VGG16。

    Args / Returns: 与 utils.pipeline.train_skorch 相同。
    """
    return train_skorch(
        config, train_dataset, test_dataset,
        model_class=VGG16,
    )
