# Experience & Lessons Learned / 经验与教训

## Architecture / 架构

- **CIFAR 适配 stem**: 标准 ResNet-18 的 7x7 conv stride=2 + maxpool 会把 32x32 图像压缩到 8x8，后续层进一步压缩到 1x1。改为 3x3 conv stride=1 + 去掉 maxpool 后，特征图变化为 32→32→16→8→4，保留了足够的空间信息。
- **迁移学习准备**: 只需保存去掉 `fc.` 前缀的 state_dict，迁移时 `load_state_dict(..., strict=False)` 即可，FC 层保持随机初始化。

## Windows Notes / Windows 注意事项

- DataLoader `num_workers > 0` 必须在 `if __name__ == "__main__"` 守卫内使用，否则会触发无限递归。
- PyTorch CUDA 需要通过 `[[tool.uv.index]]` 配置 wheel index，默认 PyPI 提供 CPU-only 版本。
