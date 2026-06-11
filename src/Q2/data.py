"""
Q2 CIFAR-10 数据加载（向后兼容重导出）。
Backward-compatible re-exports for Q2 CIFAR-10 data loading.

实际实现已统一到 utils.data。
"""

from utils.data import get_datasets as get_cifar10_datasets  # noqa: F401
from utils.data import get_test_only as get_cifar10_test_only  # noqa: F401
from utils.data import get_loaders as get_cifar10_loaders  # noqa: F401
