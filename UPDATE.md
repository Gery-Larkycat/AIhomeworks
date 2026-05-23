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

---

## 2026-05-21: Q3 性能优化与早停

### Changed / 变更

- `src/Q3/train.py`: GPU 利用率优化（non_blocking, set_to_none, GPU 累积指标避免 per-batch sync）
- `src/Q3/train.py`: AMP 混合精度训练支持（`--amp`），RTX 4060 实测 1.90x 加速
- `src/Q3/train.py`: 早停策略（patience=5, min_delta=1e-4），防止过拟合
- `src/Q3/config.py`: patience 从 20 改为 5；新增 `use_amp` 字段
- `src/Q3/evaluate.py`: GPU 张量累积，evaluate 支持 AMP autocast
- `src/Q3/tests/test_perf.py`: GPU 吞吐量基准测试
- Windows 下 `num_workers=0`（spawn 开销 > 并行收益）

---

## 2026-05-21: Q3 超参数搜索（第一版：演化/随机/网格）

### Added / 新增

- `src/Q3/search.py`: 手写超参数搜索（演化/随机/网格三种策略）
  - 演化：(μ+λ) 演化策略，锦标赛选择、BLX-α 交叉、高斯变异
  - 随机：均匀随机采样
  - 网格：笛卡尔积穷举
- `src/Q3/config.py`: `HyperparamRange`, `SearchConfig` 配置
- `src/Q3/main.py`: `--search`, `--search-only`, `--search-strategy`, `--ignore-search` 命令行参数
- `src/Q3/tests/test_search.py`: 搜索相关测试（94 项）

### Changed / 变更

- `src/Q3/visualize.py`: 修复中文字体渲染问题
- `src/Q3/config.py`: 超参数默认值调整（batch=1024, epochs=100, lr=0.1）

---

## 2026-05-22: Q3 19 种数据增强

### Added / 新增

- `src/Q3/augment.py`: 19 种数据增强技术，分 5 大类
  - A. 几何变换: RandomCrop, HFlip, Affine, Perspective
  - B. 颜色变换: ColorJitter, Grayscale, AutoContrast, Equalize, Posterize, Solarize
  - C. 噪声与降质: GaussianNoise, SaltPepper, GaussianBlur, RandomErasing
  - D. 天气与压缩: JPEGCompression, Fog, Rain
  - E. 批次级混合: CutMix, Mixup
- `src/Q3/config.py`: `AugmentationConfig`（约 40 个字段，全部有默认值）
- `src/Q3/train.py`: batch 级增强（apply_batch_augmentation），CutMix/Mixup 激活时自动禁用 label_smoothing
- `src/Q3/main.py`: `--no-augmentation` 命令行参数
- `src/Q3/tests/test_augment.py`: 增强相关测试（32 项）

---

## 2026-05-22: Q3 搜索策略扩展

### Changed / 变更

- `src/Q3/config.py`: SearchConfig 默认 strategy 改为 `"random"`，num_trials 改为 10
- `src/Q3/main.py`: `--search-strategy` 支持 random / grid
- `src/Q3/tests/test_search.py`: 新增随机搜索和网格搜索测试

---

## 2026-05-23: Q3 超参数搜索改为验证集评估（已废弃）

> 此方案在实测中发现搜索效果不如默认值，已被下一版 skorch 重写替代。

### Changed / 变更

- `src/Q3/config.py`: TrainConfig 新增 `val_ratio=0.1`（后续已删除）
- `src/Q3/data.py`: 新增 `get_cifar100_search_loaders()`（后续已删除）
- `src/Q3/search.py`: fitness 改为 `val_acc - penalty * overfit_gap`（后续已重写）
- `src/Q3/main.py`: `run_search` 不再传 `test_loader`（后续进一步简化）

---

## 2026-05-23: Q3 超参数搜索用 skorch + sklearn 重写

### Changed / 变更

- `pyproject.toml`: 新增 `skorch`, `scikit-learn`, `scipy` 依赖
- `src/Q3/config.py`: 删除 `HyperparamRange` + `val_ratio`；SearchConfig 重写为 sklearn 兼容配置
  - strategy: `"halving-random"` / `"random"` / `"grid"`
  - 新增: `search_epochs_min/max`, `halving_factor`, `cv`, `batch_size_choices`
- `src/Q3/search.py`: **全部重写** — skorch `NeuralNetClassifier` 包装 ResNet-18 → sklearn 搜索工具
  - `_prepare_search_data()`: CIFAR-100 转 numpy（仅 Normalize，无增强）
  - `_create_search_net()`: skorch 网络定义（固定 SGD, train_split=False）
  - `run_search()`: 委托 sklearn `HalvingRandomSearchCV` / `RandomizedSearchCV` / `GridSearchCV`
  - 搜索结果自动映射为 TrainConfig 字段名
- `src/Q3/data.py`: 删除 `get_cifar100_search_loaders`
- `src/Q3/main.py`: `run_search` 不再传任何 DataLoader；`--search-strategy` 改为 halving-random/random/grid
- `src/Q3/tests/test_search.py`: **全部重写**（17 项新测试）
- `src/Q3/DOC.md`: 更新超参数搜索章节（Section 3.1, 7, 8）
