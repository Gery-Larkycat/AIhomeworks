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

---

## 2026-05-24: Q3 搜索结果过滤修复

### Fixed / 修复

- `src/Q3/search.py`: `HalvingRandomSearchCV` 的 `resource="max_epochs"` 会将 `max_epochs` 纳入 `best_params_`，但这是 successive halving 的资源预算参数，不是训练超参数。传入 `TrainConfig` 会触发 `TypeError`。
  - 新增 `_VALID_TRAIN_FIELDS` 集合（从 `dataclasses.fields(TrainConfig)` 派生）
  - `run_search()` 和 `load_best_search_params()` 返回前均过滤：`{k: v for k, v in mapped.items() if k in _VALID_TRAIN_FIELDS}`
- `src/Q3/tests/test_search.py`: 新增 `test_load_filters_invalid_fields` 测试

---

## 2026-05-24: Q3 CIFAR-100 → CIFAR-10 迁移学习 + 超参搜索

### Added / 新增

- `src/Q3/transfer.py`: **新建** — 迁移学习核心模块
  - `load_pretrained_model()`: 加载 CIFAR-100 完整检查点 → 替换 FC 为目标类别数
  - `freeze_backbone()`: 冻结除 FC 外所有参数（`requires_grad=False`）
  - `print_transfer_summary()`: 打印冻结/可训练参数统计
  - `_TransferNetClassifier(NeuralNetClassifier)`: skorch 包装器，覆写 `initialize_module()` 使每次 CV fold 自动加载预训练权重并冻结 backbone
  - `run_transfer_search()`: 迁移超参搜索（HalvingRandomSearchCV，CIFAR-10 数据）
  - `_to_train_config()`: TransferConfig → TrainConfig 字段映射
  - `run_transfer()`: 迁移学习主流程（可选搜索 → 加载 → 冻结 → 训练 → 保存）
- `src/Q3/config.py`: `TransferConfig` 冻结 dataclass（CIFAR-10 归一化统计量、FC-only 微调超参数、迁移搜索配置）
- `src/Q3/config.py`: `CIFAR10_MEAN` / `CIFAR10_STD` 常量
- `src/Q3/data.py`: `get_cifar10_loaders()` — CIFAR-10 数据加载（鸭子类型接受 TransferConfig 或 TrainConfig）
- `src/Q3/main.py`: `--transfer` 和 `--transfer-checkpoint` 命令行参数 + `_run_transfer()` 迁移学习入口

### Changed / 变更

- `src/Q3/train.py`: `create_optimizer()` 改为 `[p for p in model.parameters() if p.requires_grad]`，冻结后仅优化 FC 层
- `src/Q3/main.py`: `--search` / `--search-only` 在 `--transfer` 模式下同样生效

### Tests / 测试

- `src/Q3/tests/test_transfer.py`: **新建** — 23 项测试
  - TestLoadPretrainedModel: FC 维度、backbone 权重保留、FC 随机初始化
  - TestFreezeBackbone: requires_grad 验证、可训练参数计数（5,130）
  - TestTransferConfig: 默认值、frozen 行为
  - TestToTrainConfig: 配置转换
  - TestOptimizerFiltering: 冻结/未冻结优化器参数验证
  - TestCIFAR10Loaders: 数据集 shape、batch shape（需本地缓存）

---

## 2026-05-24: Q3 训练运行隔离 + 迁移自动选最优基础模型

### Added / 新增

- `src/Q3/config.py`: 3 个辅助函数
  - `generate_timestamp()`: 生成 `YYYY-MM-DD_HHMMSS` 时间戳（可排序、Windows 安全无冒号）
  - `make_run_dir()`: 构造 `checkpoints/<timestamp>` 运行目录路径
  - `dataset_prefix(num_classes)`: 根据 num_classes 返回检查点文件名前缀（100→`resnet18_cifar100`，10→`resnet18_cifar10`）
- `src/Q3/transfer.py`: `find_best_cifar100_checkpoint()` — 自动发现最优基础模型
  - 扫描 `checkpoints/` 下所有子目录，读取每个检查点的 `accuracy` 字段
  - 按准确率降序选择，同准确率取最新
  - 回退兼容旧平面目录结构
- `src/Q3/tests/test_run_dirs.py`: **新建** — 15 项测试（时间戳格式、目录构造、数据集前缀、动态文件名）

### Changed / 变更

- `src/Q3/config.py`: 删除 `TrainConfig` 中重复的 `scheduler_t_max` 字段
- `src/Q3/checkpoint.py`: `save_best_checkpoint` 和 `save_feature_extractor` 改用 `dataset_prefix()` 动态生成文件名（CIFAR-10 产出 `resnet18_cifar10_*.pth`）
- `src/Q3/main.py`:
  - `build_config()` 新增 `checkpoint_dir` 参数
  - `main()` 和 `_run_transfer()` 启动时生成时间戳运行目录，所有产物存入 `checkpoints/<timestamp>/`
  - `_run_transfer()` 无 `--transfer-checkpoint` 时自动调用 `find_best_cifar100_checkpoint()`
  - 修正评估阶段硬编码的检查点路径
- `src/Q3/tests/test_transfer.py`: 新增 `TestFindBestCifar100Checkpoint` — 8 项测试（准确率选优、同分取新、回退兼容）

### Behavior Change / 行为变更

- 每次训练运行存入独立的 `checkpoints/YYYY-MM-DD_HHMMSS/` 目录，不再互相覆盖
- `--transfer` 无显式 `--transfer-checkpoint` 时，自动从所有历史运行中选 CIFAR-100 准确率最高的模型
- 检查点文件名根据数据集动态生成（`resnet18_cifar100_*.pth` vs `resnet18_cifar10_*.pth`）

---

## 2026-05-24: Q3 迁移学习架构修正（保留原 FC + 追加新分类层）

### Changed / 变更

- `src/Q3/transfer.py`:
  - `load_pretrained_model()`: 不再替换 FC 层，改为保留原始 FC（512→100）并追加新分类层（100→10）。模型 forward 路径：backbone → 原 FC（微调）→ 新分类层（训练）
  - `_TransferNetClassifier.initialize_module()`: 加载完整预训练权重（含 FC），然后包装 FC 为 `nn.Sequential(原 FC, 新分类层)`，backbone 冻结后两层 FC 均可训练
  - `run_transfer_search()`: `module__num_classes` 改为 `source_num_classes`（100），新增 `module__target_num_classes`
- `src/Q3/tests/test_transfer.py`: 适配新 FC 结构
  - TestLoadPretrainedModel: 验证 FC 是 Sequential、原 FC 权重保留、新分类层随机初始化
  - TestFreezeBackbone: 迁移模型可训练参数 = 52,310（原 FC 51,300 + 新分类层 1,010）
  - TestOptimizerFiltering: 新增 `test_optimizer_param_count_transfer`
  - TestPrintSummary: 使用迁移模型验证输出

---

## 2026-05-24: Q3 PyTorch 预训练 ResNet-18 迁移学习

### Added / 新增

- `src/Q3/torchvision_transfer.py`: **新建** — torchvision 预训练模型迁移学习
  - `load_torchvision_pretrained()`: 加载 torchvision ImageNet 预训练 ResNet-18，替换 FC 为目标类别
  - `freeze_backbone_tv()`: 冻结 backbone（仅 FC 可训练，5,130 参数）
  - `_build_tv_transforms()`: 构建 224x224 变换管线（Resize + ImageNet 归一化，可选增强）
  - `get_cifar10_224_loaders()`: CIFAR-10 数据加载（32x32 → 224x224 上采样）
  - `_to_train_config()`: TorchvisionTransferConfig → TrainConfig 转换
  - `run_torchvision_transfer()`: 迁移学习主流程（加载 → 冻结 → 训练 FC → 保存）
- `src/Q3/config.py`: 新增 `IMAGENET_MEAN/STD` 常量 + `TorchvisionTransferConfig` 冻结 dataclass
- `src/Q3/main.py`: `--tv-transfer` 命令行参数 + `_run_torchvision_transfer()` 入口
- `src/Q3/tests/test_torchvision_transfer.py`: **新建** — 26 项测试
  - TestLoadTorchvisionPretrained: FC 维度、backbone 权重加载、前向传播
  - TestFreezeBackboneTv: requires_grad 验证、可训练参数计数（5,130）
  - TestTorchvisionTransferConfig: 默认值、frozen 行为
  - TestToTrainConfig: 配置转换
  - TestOptimizerFilteringTv: 优化器参数过滤
  - TestCIFAR10224Loaders: 224x224 数据 shape
  - TestBuildTvTransforms: 变换管线验证

### Configuration / 配置

- `TorchvisionTransferConfig` 默认值: batch=64, lr=0.01, epochs=30, ImageNet 归一化, 224x224 输入

---

## 2026-05-24: Q3 ResNet-18 Dropout 正则化

### Added / 新增

- `src/Q3/model.py`: `ResNet18` 新增 `dropout_rate` 参数（默认 0.5），在 `AdaptiveAvgPool2d` 后、FC 前插入 `nn.Dropout`。`rate=0` 时等同于无 Dropout。
- `src/Q3/config.py`: `TrainConfig` 新增 `dropout_rate: float = 0.5` 字段
- `src/Q3/main.py`: 新增 `--dropout` CLI 参数，所有 `create_model` 调用均传递 `dropout_rate`
- `src/Q3/search.py`: `module__dropout_rate` 加入 `PARAM_MAP`（→ `dropout_rate`）+ 搜索空间
  - 随机搜索: `uniform(0.0, 0.5)`
  - 网格搜索: `[0.0, 0.1, 0.3, 0.5]`
- `src/Q3/tests/test_model.py`: 新增 5 项 Dropout 测试（默认行为、指定比率、训练模式随机性、模块存在性、参数量不变）

### Behavior Change / 行为变更

- 训练默认启用 Dropout（rate=0.5）；可通过 `--dropout 0` 禁用
- 超参数搜索自动探索 dropout_rate，搜索结果会覆盖默认值
- 迁移学习不受影响（`ResNet18` 默认 `dropout_rate=0`，预训练权重加载不涉及 Dropout 参数）
