# Update Log / 更新日志

## 2026-05-21: Q3 ResNet-18 CIFAR-100 项目初始化

### Added / 新增

- `PLAN.md`: 项目计划文档
- `src/Q3/config.py`: 训练配置（冻结 dataclass）
- `src/Q3/model.py`: 从零实现 ResNet-18（CIFAR 适配 stem）
  - BasicBlock: 标准 ResNet 残差块
  - ResNet18: stem=3x3 conv stride=1（无 maxpool）+ 标准 [2,2,2,2] 残差层
  - `get_feature_extractor_state()`: 提取特征提取器权重（去掉 FC）
- `src/Q3/data.py`: CIFAR-100 数据加载（仅 Normalize，无增强）
- `src/Q3/train.py`: 训练循环（SGD + Cosine Annealing + Label Smoothing）
- `src/Q3/evaluate.py`: 评估指标（准确率、每类准确率、混淆矩阵）
- `src/Q3/checkpoint.py`: 检查点管理 + 特征提取器导出
- `src/Q3/visualize.py`: 可视化（训练曲线、混淆矩阵、学习率曲线）
- `src/Q3/main.py`: 主入口（支持命令行参数覆盖）
- `src/Q3/tests/test_model.py`: 模型测试（shape、参数量、梯度流、迁移学习验证）
- `src/Q3/tests/test_data.py`: 数据测试（shape、label 范围、归一化）

### Configuration / 配置

- 依赖: torch (CUDA 12.4), torchvision, matplotlib, numpy, pytest
- 超参数: batch=128, lr=0.1, epochs=200, SGD(momentum=0.9, wd=5e-4)
