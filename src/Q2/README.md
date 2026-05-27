# Q2 — ResNet-18 CIFAR-10 分类训练管线

基于从零实现的 ResNet-18，在 CIFAR-10 上进行图像分类。使用 skorch `NeuralNetClassifier` 包装训练流程，内置计时、早停、检查点管理。

Q3 引用此模块的 `model.py`（ResNet-18 定义）和 `training.py`（训练管线）完成 CIFAR-100 训练和迁移学习。

## 目录

```
Q2/
├── model.py      # ResNet-18 模型定义（BasicBlock + ResNet18）
├── config.py     # Q2TrainConfig（CIFAR-10 默认值）
├── data.py       # CIFAR-10 数据加载（返回 Dataset）
├── training.py   # train_resnet() — skorch 训练管线
├── search.py     # Q2 超参搜索（委托 utils.search）
└── tests/
    └── __init__.py
```

---

## 模块详解

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

返回 **Dataset**（不是 DataLoader），由 skorch 训练器负责创建 DataLoader。

- `get_cifar10_datasets(config)` → `(train_dataset, test_dataset)`：训练集带增强，测试集仅归一化
- `get_cifar10_test_only(config)` → `test_dataset`：仅加载测试集（评估用）

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

- `run_q2_search(config, search_cfg)` — 加载 CIFAR-10 训练集（仅 Normalize），调用 `run_search(ResNet18, ...)`
- `load_q2_best_params(config)` → `dict | None` — 加载搜索结果，过滤为 `Q2TrainConfig` 有效字段

搜索空间与 `utils.search` 一致（lr, momentum, weight_decay, batch_size, dropout_rate）。

---

## 使用方法

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
