"""
Q2 ResNet-18 模型定义（向后兼容重导出）。
Backward-compatible re-exports for Q2 ResNet-18 model.

实际模型定义已移至 models.resnet18。
"""

from models.resnet18 import (  # noqa: F401
    ResNet18,
    create_model,
    get_feature_extractor_state,
)
