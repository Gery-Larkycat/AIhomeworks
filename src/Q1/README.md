# Q1 — VGG-16 CIFAR-10 分类训练管线

基于从零实现的 VGG-16，在 CIFAR-10 上进行图像分类。使用 skorch `NeuralNetClassifier` 包装训练流程，内置计时、早停、检查点管理。

## 目录

```
Q1/
├── main.py       # CLI 入口（训练/搜索/评估/可视化）
├── model.py      # VGG-16 模型定义（CIFAR 适配）
├── config.py     # Q1TrainConfig（CIFAR-10 默认值）
├── data.py       # CIFAR-10 数据加载（Dataset + DataLoader）
├── training.py   # train_vgg() — skorch 训练管线
├── search.py     # Q1 超参搜索（委托 utils.search）
├── README.md
└── tests/
    ├── __init__.py
    ├── test_model.py   # 模型测试（形状、参数量、Dropout、use_bn）
    ├── test_config.py  # 配置测试（默认值、不可变性）
    ├── test_data.py    # 数据加载测试
    └── test_search.py  # 搜索测试

outputs/Q1/
├── checkpoints/                          # 训练检查点
│   └── <timestamp>/
│       ├── vgg16_cifar10_best.pth
│       ├── training_history.json
│       └── plots/
└── search_results/
    └── <timestamp>_vgg16_cifar10_hp_search.json
```

---

## VGG-16 CIFAR-10 适配

标准 VGG-16（ImageNet 224×224）有 5 个 MaxPool，特征图 224→112→56→28→14→7。

**CIFAR-10 适配**（32×32）：
- 保留全部 5 组卷积结构，仅 Block 1/2/3 各跟 1 个 MaxPool（3 个）
- 特征图路径：32→16→8→4→4→4 → AdaptiveAvgPool2d(1,1)
- FC 简化为 2 层：`Dropout → Linear(512, 512) → ReLU → Dropout → Linear(512, num_classes)`
- BatchNorm 可通过 `use_bn` 参数启用/禁用

---

## 模块详解

### `model.py` — VGG-16 模型

**`VGG16(nn.Module)`**：
- 构造参数：`num_classes=10`, `dropout_rate=0.0`, `use_bn=True`
- 5 组卷积层 + 3 个 MaxPool
- 2 层 FC 分类器
- Kaiming 正态初始化

**辅助函数**：
- `create_model(num_classes, dropout_rate, use_bn)` — 工厂函数
- `get_feature_extractor_state(model)` — 去掉 FC 前缀的 state_dict

### `config.py` — Q1TrainConfig

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_classes` | `10` | CIFAR-10 类别数 |
| `dropout_rate` | `0.5` | FC 前 Dropout |
| `use_bn` | `True` | 卷积后 BatchNorm |
| `batch_size` | `256` | 批大小 |
| `epochs` | `200` | 训练轮数 |
| `learning_rate` | `0.1` | 初始学习率 |
| `patience` | `10` | 早停等待轮数 |

### `main.py` — CLI 入口

| 参数 | 说明 |
|---|---|
| `--epochs`, `--batch-size`, `--lr`, `--dropout` | 训练参数覆盖 |
| `--no-bn` | 禁用 BatchNorm |
| `--no-augmentation` | 禁用数据增强 |
| `--search` / `--search-only` | 超参搜索 |
| `--amp` | 混合精度训练 |

---

## 使用方法

```bash
# 完整训练
uv run python src/Q1/main.py

# 自定义参数
uv run python src/Q1/main.py --epochs 50 --lr 0.01 --dropout 0.3

# 禁用 BN
uv run python src/Q1/main.py --no-bn

# 超参搜索
uv run python src/Q1/main.py --search-only
```

运行测试：`uv run pytest src/Q1/tests/ -v`
