"""
Q1 VGG-16 模型定义（向后兼容重导出）。
Backward-compatible re-exports for Q1 VGG-16 model.

实际模型定义已移至 models.vgg16。
"""

from models.vgg16 import (  # noqa: F401
    VGG16,
    create_model,
    get_feature_extractor_state,
)
