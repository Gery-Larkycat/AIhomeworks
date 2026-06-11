"""
模型定义包：VGG-16、ResNet-18 及任务注册表。
Model definitions: VGG-16, ResNet-18, and task registry.
"""

from .vgg16 import VGG16, create_model, get_feature_extractor_state
from .resnet18 import ResNet18
from .registry import TaskSpec, register, get_spec, make_config

__all__ = [
    "VGG16",
    "ResNet18",
    "TaskSpec",
    "register",
    "get_spec",
    "make_config",
]
