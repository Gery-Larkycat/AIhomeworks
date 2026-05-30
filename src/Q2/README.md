# Q2 — ResNet-18 CIFAR-10 分类训练管线

基于从零实现的 ResNet-18，在 CIFAR-10 上进行图像分类。使用 skorch `NeuralNetClassifier` 包装训练流程，内置计时、早停、检查点管理。

Q3 引用此模块的 `model.py`（ResNet-18 定义）和 `training.py`（训练管线）完成 CIFAR-100 训练和迁移学习。

## 目录

```
Q2/
├── main.py       # CLI 入口（训练/搜索/评估/可视化）
├── model.py      # ResNet-18 模型定义（BasicBlock + ResNet18）
├── config.py     # Q2TrainConfig（CIFAR-10 默认值）
├── data.py       # CIFAR-10 数据加载（Dataset + DataLoader）
├── training.py   # train_resnet() — skorch 训练管线
├── search.py     # Q2 超参搜索（委托 utils.search）
├── README.md
└── tests/
    ├── __init__.py
    ├── test_config.py   # 配置测试（默认值、不可变性、覆盖）
    ├── test_data.py     # 数据加载测试（形状、标签范围、归一化）
    └── test_search.py   # 搜索测试（参数映射、过滤、加载）

outputs/Q2/
├── checkpoints/                          # 训练检查点 / Training checkpoints
│   └── <timestamp>/                      # 每次运行的独立目录 / Per-run directory
│       ├── resnet18_cifar10_best.pth     # 最优模型权重 / Best model weights
│       ├── training_history.json         # 训练历史 / Training history
│       └── plots/                        # 可视化图表 / Visualization plots
└── search_results/                       # 超参搜索结果 / HP search results
    └── <timestamp>_cifar10_hp_search.json
```

---

## 模块详解

### `main.py` — CLI 入口

命令行训练入口，编排完整流程：数据加载 → 超参搜索（可选）→ 训练 → 评估 → 可视化。

**CLI 参数**：

| 参数 | 说明 |
|---|---|
| `--epochs` | 覆盖训练轮数 |
| `--batch-size` | 覆盖批大小 |
| `--lr` | 覆盖学习率 |
| `--dropout` | FC 前 Dropout 比率（0 = 禁用） |
| `--data-root` | 覆盖数据根目录 |
| `--amp` | 启用混合精度（FP16）训练 |
| `--no-augmentation` | 禁用数据增强 |
| `--search` | 先搜索再用最优配置训练 |
| `--search-only` | 仅运行搜索并退出 |
| `--search-strategy` | 搜索策略：`halving-random`（默认）、`random`、`grid` |
| `--ignore-search` | 忽略已有搜索结果，使用默认/CLI 参数 |
| `--search-results` | 指定搜索结果文件路径 |
| `--eval-only` | 仅评估（需要已有检查点） |

**流程**：
1. 解析参数 → 生成时间戳运行目录 → 构建配置
2. 加载 CIFAR-10 数据集
3. 超参搜索（`--search`）或自动加载已有搜索结果
4. 调用 `train_resnet()` 训练（skorch 管线）
5. 评估：test loss/acc、每类准确率（top-5 / bottom-5）、混淆矩阵
6. 可视化：训练曲线、混淆矩阵热力图、学习率曲线

---

### `model.py` — ResNet-18 模型

从零实现的 ResNet-18，适配 CIFAR（32×32）图像。

**与标准 ImageNet ResNet-18 的差异**：
- **Stem**：`Conv2d(3→64, 3×3, stride=1, padding=1)`（标准为 7×7 stride=2 + maxpool）
- 去掉 maxpool，保留 32×32 图像的空间信息
- 特征图路径：32→32→16→8→4（标准为 224→56→28→14→7）

**`BasicBlock(nn.Module)`**：
- 两个 3×3 卷积层 + 残差跳跃连接
- `expansion=1`（ResNet-18 不使用 bottleneck）
- shortcut：维度匹配时 `Identity`，否则 `1×1 conv + BN`

**`ResNet18(nn.Module)`**：
- 构建参数：`num_classes`（默认 10）、`dropout_rate`（默认 0）
- 结构：stem → 4 组残差层 [2,2,2,2] → AdaptiveAvgPool2d → Dropout → FC
- Kaiming 正态初始化（`fan_out`, `relu`）
- 约 11.17M 参数（100 类时）

**辅助函数**：
- `create_model(num_classes, dropout_rate)` — 工厂函数
- `get_feature_extractor_state(model)` — 返回去掉 `fc.` 前缀的 state_dict（迁移学习用）

---

### `config.py` — Q2 训练配置

**`Q2TrainConfig`**（frozen dataclass），CIFAR-10 默认值：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `data_root` | `Path("data")` | 数据集根目录 / Dataset root |
| `checkpoint_dir` | `Path("outputs/Q2/checkpoints")` | 检查点保存目录 / Checkpoint save dir |
| `num_classes` | `10` | CIFAR-10 类别数 |
| `dropout_rate` | `0.5` | FC 前 Dropout |
| `batch_size` | `128` | 批大小 |
| `epochs` | `200` | 训练轮数 |
| `learning_rate` | `0.1` | 初始学习率 |
| `momentum` | `0.9` | SGD 动量 |
| `weight_decay` | `5e-4` | L2 正则化 |
| `label_smoothing` | `0.1` | 标签平滑 |
| `scheduler_t_max` | `200` | 余弦退火周期（= epochs） |
| `patience` | `10` | 早停等待轮数 |
| `mean` / `std` | CIFAR-10 统计量 | 归一化参数 |

从 `utils.config` 导入 `AugmentationConfig`、`SearchConfig`、常量、辅助函数。

---

### `data.py` — CIFAR-10 数据加载

- `get_cifar10_datasets(config)` → `(train_dataset, test_dataset)`：训练集带增强，测试集仅归一化。返回 Dataset，由 skorch 训练器创建 DataLoader。
- `get_cifar10_test_only(config)` → `test_dataset`：仅加载测试集（评估用）。
- `get_cifar10_loaders(config)` → `(train_loader, test_loader)`：返回 DataLoader，供评估和可视化使用。

训练集变换通过 `utils.augment.build_train_transforms(config.augmentation, config.mean, config.std)` 构建，测试集通过 `build_test_transforms(config.mean, config.std)`。

---

### `training.py` — skorch 训练管线

**`train_resnet(config, train_dataset, test_dataset, save_feature_extractor=False)`** → `(net, history_dict)`

完整 ResNet-18 训练管线，Q2 和 Q3 共用：
- Q2：CIFAR-10 训练（10 类）
- Q3：CIFAR-100 训练（100 类，`save_feature_extractor=True`）

**流程**：
1. 启用 cuDNN benchmark（CIFAR 输入尺寸固定）
2. 调用 `create_classifier_net(ResNet18, config, ...)` 创建 skorch 训练器
3. `net.fit(train_dataset, y=None)` 开始训练
4. `extract_history(net)` 提取标准历史 dict

**返回的 `history_dict`**：
- `train_loss` / `test_loss` — 每轮损失
- `train_acc` / `test_acc` — 每轮准确率
- `lr` — 每轮学习率
- `dur` — 每轮耗时（秒）

skorch 自动管理的功能：epoch 循环、EarlyStopping（patience=10）、CosineAnnealingLR 调度、CustomCheckpoint（最优模型保存）、CutMix/Mixup 批次增强。

---

### `search.py` — Q2 超参搜索

Q2 的 CIFAR-10 超参搜索，委托 `utils.search` 通用搜索模块。

- `run_q2_search(config, search_cfg)` — 加载 CIFAR-10 训练集（仅 Normalize），调用 `run_search(ResNet18, ...)`，结果保存到 `outputs/Q2/search_results/<timestamp>_cifar10_hp_search.json`
- `load_q2_best_params(specific_file)` → `dict | None` — 默认扫描 `outputs/Q2/search_results/*_cifar10_hp_search.json`，加载搜索结果，过滤为 `Q2TrainConfig` 有效字段

搜索空间与 `utils.search` 一致（lr, momentum, weight_decay, batch_size, dropout_rate）。

---

## 测试

| 测试文件 | 覆盖内容 |
|---|---|
| `test_config.py` | 默认值（17 项）、不可变性（2 项）、构造覆盖（3 项）、默认路径（3 项）、dataclasses.replace（2 项） |
| `test_data.py` | 数据集大小、批次形状、标签范围、归一化范围、test_only（7 项） |
| `test_search.py` | 搜索结果加载/映射/过滤、SearchConfig 默认值（7 项） |

运行：`uv run pytest src/Q2/tests/ -v`

注意：模型测试在 Q3 的 `test_model.py` 中（因为 model.py 在 Q2 但 Q3 最先建立测试）。

---

## 使用方法

**CLI 训练**：
```bash
# 完整训练（200 轮）
uv run python src/Q2/main.py

# 自定义参数
uv run python src/Q2/main.py --epochs 50 --lr 0.01 --dropout 0.3

# 仅超参搜索
uv run python src/Q2/main.py --search-only

# 搜索后训练
uv run python src/Q2/main.py --search

# 指定搜索结果文件
uv run python src/Q2/main.py --search-results outputs/Q2/search_results/xxx_cifar10_hp_search.json

# 禁用增强
uv run python src/Q2/main.py --no-augmentation
```

**编程接口**：
```python
from Q2.config import Q2TrainConfig
from Q2.data import get_cifar10_datasets
from Q2.training import train_resnet

config = Q2TrainConfig()
train_ds, test_ds = get_cifar10_datasets(config)
net, history = train_resnet(config, train_ds, test_ds)

# history 含每轮 dur（秒）
print(f"Total training time: {sum(history['dur']):.1f}s")
print(f"Best accuracy: {max(history['test_acc']):.4f}")
```
