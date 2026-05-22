# Q3: ResNet-18 on CIFAR-100 + Transfer Learning Preparation

## Context / 背景

在 `src/Q3/` 下从零实现 ResNet-18，在 CIFAR-100 上训练评估，并为后续迁移到 CIFAR-10 做好准备（保存特征提取器权重）。

---

## Architecture Decisions / 架构决策

### ResNet-18 结构

- **从零实现**（不调用 torchvision.models.resnet18）
- **仅 stem 做 CIFAR 适配**：3x3 conv stride=1 + 去掉 maxpool（防止 32x32 图像特征图过早缩到 1x1）
- **所有残差块完全保持标准结构**：4 组 [2,2,2,2] BasicBlock，通道 64→128→256→512
- **FC 层适配不同数据集**：CIFAR-100 → Linear(512, 100)；CIFAR-10 → Linear(512, 10)

### Data / 数据

- **不做额外增强**，仅 Normalize（CIFAR-100 stats）
- CIFAR-100 归一化：mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]

### Training / 训练

- SGD (momentum=0.9, weight_decay=5e-4) + CosineAnnealingLR
- Label smoothing=0.1
- Batch size=128, Epochs=200, LR=0.1
- 预期 ~77-80% top-1 accuracy

### Transfer Learning / 迁移学习准备

- 训练完成后保存特征提取器（去掉 FC 层的 state_dict）
- CIFAR-10 迁移时只需加载权重 + 新建 Linear(512, 10) 头部

---

## File Structure / 文件结构

```
src/Q3/
  __init__.py
  config.py         # 冻结 dataclass，所有超参数
  model.py          # 从零实现 ResNet-18（CIFAR 适配 stem）
  data.py           # CIFAR-100 数据加载
  train.py          # 训练循环
  evaluate.py       # 评估指标
  checkpoint.py     # 模型保存（含特征提取器导出）
  visualize.py      # 可视化
  main.py           # 入口
  tests/
    __init__.py
    test_model.py
    test_data.py
```

根目录：`UPDATE.md`, `EXPERIENCE.md`

---

## Git Commits / 提交计划

1. `feat(Q3): 初始化项目结构和依赖 / Initialize project structure and dependencies`
2. `test(Q3): 添加测试 / Add tests`
3. `feat(Q3): 实现ResNet-18模型 / Implement ResNet-18 model`
4. `feat(Q3): 实现数据加载 / Implement data loading`
5. `feat(Q3): 实现训练和评估 / Implement training and evaluation`
6. `feat(Q3): 实现检查点和可视化 / Implement checkpoint and visualization`
7. `feat(Q3): 实现主入口 / Implement main entry point`
8. `docs(Q3): 添加文档 / Add documentation`

---

## Verification / 验证

1. `uv run pytest src/Q3/tests/` 通过
2. `uv run python src/Q3/main.py` 训练无报错
3. 特征提取器 .pth 文件生成且可加载
4. GPU 正常使用（nvidia-smi 确认）
