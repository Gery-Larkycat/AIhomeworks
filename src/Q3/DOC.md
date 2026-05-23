# Q3: ResNet-18 on CIFAR-100 — 实现文档

## 目录

1. [项目概览](#1-项目概览)
2. [网络架构](#2-网络架构)
3. [模块详解](#3-模块详解)
4. [数据增强](#4-数据增强)
5. [训练流程](#5-训练流程)
6. [迁移学习准备](#6-迁移学习准备)
7. [超参数搜索](#7-超参数搜索)
8. [使用方法](#8-使用方法)

---

## 1. 项目概览

本实现从零构建 ResNet-18 网络，在 CIFAR-100（100 类，32×32 彩色图像）上进行训练和评估，并导出特征提取器权重供后续迁移到 CIFAR-10 使用。

**文件结构**：

```
src/Q3/
├── config.py         # 训练配置 + 搜索配置 + 数据增强配置（AugmentationConfig）
├── model.py          # 从零实现的 ResNet-18
├── data.py           # CIFAR-100 数据加载（委托 augment.py 构建变换管线）
├── augment.py        # 数据增强（19 种技术，5 大类 + CutMix/Mixup）
├── train.py          # 训练循环 + 优化器/调度器工厂函数 + batch 级增强
├── evaluate.py       # 评估指标
├── search.py         # 超参数搜索（skorch + sklearn，halving-random/random/grid）
├── checkpoint.py     # 检查点管理与特征提取器导出
├── visualize.py      # 可视化（训练曲线、混淆矩阵）
├── main.py           # 主入口（含 --search / --no-augmentation / --amp 等）
├── scripts/          # 探索性分析脚本
│   └── analyze_class_distribution.py  # 类别分布分析
└── tests/
    ├── test_model.py   # 模型验证
    ├── test_data.py    # 数据验证
    ├── test_augment.py # 增强验证（32 项测试）
    ├── test_search.py  # 搜索验证
    └── test_perf.py    # GPU 性能基准测试
```

**设计原则**：每个文件单一职责（SRP），模块间通过 `TrainConfig` dataclass 和函数参数传递依赖，避免全局状态。

---

## 2. 网络架构

### 2.1 整体结构

```
输入 (B, 3, 32, 32)
    │
    ▼
┌─────────────────────────┐
│  Stem                    │  Conv2d(3→64, 3×3, stride=1, padding=1, bias=False)
│                          │  BatchNorm2d(64)
│                          │  ReLU
└─────────────────────────┘
    │  → (B, 64, 32, 32)
    ▼
┌─────────────────────────┐
│  Layer1: 2× BasicBlock   │  通道 64→64, stride=1
└─────────────────────────┘
    │  → (B, 64, 32, 32)
    ▼
┌─────────────────────────┐
│  Layer2: 2× BasicBlock   │  通道 64→128, stride=2
└─────────────────────────┘
    │  → (B, 128, 16, 16)
    ▼
┌─────────────────────────┐
│  Layer3: 2× BasicBlock   │  通道 128→256, stride=2
└─────────────────────────┘
    │  → (B, 256, 8, 8)
    ▼
┌─────────────────────────┐
│  Layer4: 2× BasicBlock   │  通道 256→512, stride=2
└─────────────────────────┘
    │  → (B, 512, 4, 4)
    ▼
┌─────────────────────────┐
│  AdaptiveAvgPool2d(1,1)  │  → (B, 512, 1, 1)
│  Flatten                 │  → (B, 512)
│  Linear(512→num_classes) │  → (B, 100)
└─────────────────────────┘
```

### 2.2 与标准 ImageNet ResNet-18 的差异

唯一改动在 **stem** 部分：

| | 标准 ImageNet ResNet-18 | 本实现（CIFAR 适配） |
|---|---|---|
| **conv1** | Conv2d(3→64, 7×7, stride=2, padding=3) | Conv2d(3→64, 3×3, stride=1, padding=1) |
| **maxpool** | MaxPool2d(3×3, stride=2, padding=1) | **无** |

**为什么改 stem**：标准 stem 的 7×7 conv stride=2 会将 32×32 图像下采样到 16×16，maxpool 再压到 8×8。后续 4 组残差层继续减半，最终特征图只有 1×1，丢失了几乎全部空间信息。改为 3×3 conv stride=1 + 去掉 maxpool 后，特征图变化路径为 32→32→16→8→4，在全局平均池化前保留了 4×4 的空间分辨率。

**所有残差块完全保持标准结构**，包括 BasicBlock 内部的 3×3 卷积、BatchNorm、shortcut 连接方式，以及 4 组 [2,2,2,2] 的层数配置。

### 2.3 BasicBlock 残差块

每个 BasicBlock 包含两个 3×3 卷积层和一个残差跳跃连接：

```
输入 x
  ├─→ conv1 (3×3, stride) → BN → ReLU → conv2 (3×3, stride=1) → BN ──┐
  │                                                                      │
  └─→ shortcut (维度匹配时为 Identity；不匹配时为 1×1 conv + BN) ────────┤
                                                                         │
                                                    + ←──────────────────┘
                                                    │
                                                  ReLU
                                                    │
                                                  输出
```

**shortcut 分支**：
- 当 `stride=1` 且 `in_channels == out_channels` 时：`nn.Identity()`（直连）
- 否则：`Conv2d(in→out, 1×1, stride) + BatchNorm2d`（对齐维度）

### 2.4 权重初始化

- 卷积层：Kaiming 正态初始化（`mode="fan_out"`, `nonlinearity="relu"`），适配 ReLU 激活函数
- BatchNorm：权重=1，偏置=0

### 2.5 参数量

约 11.17M 参数（取决于 `num_classes`）。各部分分布：

| 组件 | 参数量（约） |
|---|---|
| Stem (conv1 + bn1) | 1,920 |
| Layer1 (4 个 BasicBlock) | 148,032 |
| Layer2 (4 个 BasicBlock) | 526,464 |
| Layer3 (4 个 BasicBlock) | 2,100,032 |
| Layer4 (4 个 BasicBlock) | 8,394,752 |
| FC (512→100) | 51,300 |
| **总计** | **~11.22M** |

---

## 3. 模块详解

### 3.1 `config.py` — 训练配置、搜索配置与增强配置

使用 `@dataclass(frozen=True)` 定义不可变配置对象，所有超参数集中管理：

**TrainConfig** — 训练配置：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `data_root` | `"data"` | CIFAR-100 数据存放路径 |
| `checkpoint_dir` | `"checkpoints"` | 检查点保存路径 |
| `num_classes` | `100` | 分类数（CIFAR-100=100） |
| `batch_size` | `1024` | 批大小 |
| `epochs` | `100` | 训练轮数 |
| `learning_rate` | `0.1` | 初始学习率 |
| `momentum` | `0.9` | SGD 动量 |
| `weight_decay` | `5e-4` | L2 正则化系数 |
| `label_smoothing` | `0.1` | 标签平滑系数 |
| `optimizer_type` | `"sgd"` | 优化器类型（sgd/adam/adamw/rmsprop/nadam） |
| `scheduler_type` | `"cosine"` | 调度器类型（cosine/constant/step） |
| `patience` | `10` | 早停等待轮数 |
| `min_delta` | `1e-4` | 视为改善的最小准确率增量 |
| `scheduler_t_max` | `100` | 余弦退火周期（= epochs） |
| `mean` | `(0.5071, 0.4867, 0.4408)` | CIFAR-100 均值 |
| `std` | `(0.2675, 0.2565, 0.2761)` | CIFAR-100 标准差 |
| `augmentation` | `AugmentationConfig()` | 数据增强配置（独立管理） |

**AugmentationConfig** — 数据增强配置（详见第 5 节）：

**SearchConfig** — 超参数搜索配置（skorch + sklearn）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `strategy` | `"halving-random"` | 搜索策略：`halving-random` / `random` / `grid` |
| `search_epochs_min` | `2` | successive halving 起始轮数 |
| `search_epochs_max` | `20` | successive halving 最大轮数 |
| `halving_factor` | `3` | 每轮保留前 1/factor 的候选 |
| `num_trials` | `50` | 随机采样候选数 |
| `cv` | `3` | 交叉验证折数 |
| `batch_size_choices` | `(128, 256, 512)` | batch_size 候选值 |
| `scoring` | `"accuracy"` | sklearn 评分指标 |

搜索空间（`scipy.stats` 分布，硬编码在 `search.py`）：
- `lr`: loguniform(1e-4, 1.0)
- `optimizer__momentum`: uniform(0.85, 0.14) → [0.85, 0.99]
- `optimizer__weight_decay`: loguniform(1e-6, 1e-2)
- `batch_size`: 离散候选列表

`frozen=True` 保证训练过程中配置不被意外修改。搜索固定使用 SGD（ResNet-18 标准优化器）。

### 3.2 `model.py` — 网络模型

核心类和函数：

- **`BasicBlock(nn.Module)`**：标准残差块。第一个 conv 接受 `stride` 参数用于下采样，第二个 conv 始终 stride=1。`expansion=1`（ResNet-18 不使用 bottleneck）。
- **`ResNet18(nn.Module)`**：
  - `__init__`：构建 stem → 4 层残差组 → 分类头。`_make_layer` 递增 `self.in_channels` 供下一层使用。
  - `_make_layer(out_channels, num_blocks, stride)`：构建一组 BasicBlock。第一个块使用传入的 stride（可能下采样），后续块 stride=1。
  - `_initialize_weights()`：Kaiming 初始化卷积，常数初始化 BN。
  - `forward(x)`：stem → layer1-4 → avgpool → flatten → fc。
- **`create_model(num_classes=100)`**：工厂函数，创建模型实例。
- **`get_feature_extractor_state(model)`**：返回去掉 `fc.` 前缀的 state_dict，供迁移学习使用。

### 3.3 `data.py` — 数据加载

训练集和测试集使用独立的变换管线：

- **训练集**：委托 `augment.py` 的 `build_train_transforms()` 构建（19 种增强 + 归一化）
- **测试集**：仅 `ToTensor + Normalize`（无增强）

`get_cifar100_loaders(config)` 返回 `(train_loader, test_loader)`，首次运行自动下载 CIFAR-100 到 `data_root`。

超参数搜索的数据准备由 `search.py` 的 `_prepare_search_data()` 处理（转为 numpy 数组，仅 Normalize），不经过此模块。

### 3.4 `train.py` — 训练循环

**工厂函数**：
- `create_optimizer(model, config)`：根据 `config.optimizer_type` 创建优化器（SGD/Adam/AdamW/RMSprop/NAdam）
- `create_scheduler(optimizer, config)`：根据 `config.scheduler_type` 创建调度器（CosineAnnealingLR/StepLR/constant=None）
- `create_criterion(config)`：创建 `CrossEntropyLoss(label_smoothing=...)`

**`train_one_epoch`**：
- 标准训练步骤：`zero_grad → batch augmentation → forward → loss → backward → step`
- 每个批次前随机应用 CutMix / Mixup / 不增强（概率控制）
- Soft labels 由 `CrossEntropyLoss` 原生处理
- 准确率使用 dominant label 计算（soft labels 时用 `argmax`）
- 累积 loss 和正确数，每 100 个 batch 打印进度
- 返回 `(avg_loss, accuracy)`

**`train`**（完整训练循环）：
- 通过工厂函数创建优化器、调度器和损失函数
- CutMix/Mixup 激活时自动禁用 `label_smoothing`（soft labels 已提供正则化）
- 每 epoch：训练 → 评估 → 调整学习率 → 记录历史 → 保存最佳模型 → 早停检查
- 当测试准确率创新高时，同时保存完整检查点和特征提取器
- **早停策略**：连续 `patience` 轮测试准确率没有超过 `min_delta` 的改善则提前终止
- 返回训练历史字典（每轮的 train_loss, train_acc, test_loss, test_acc, lr）

### 3.5 `evaluate.py` — 评估

三个纯函数，全部使用 `@torch.no_grad()` 装饰器：

- **`evaluate(model, loader, device)`** → `(loss, accuracy)`：全局 top-1 评估
- **`per_class_accuracy(model, loader, device, num_classes)`** → `dict[int, float]`：每个类别的准确率，用于分析模型在细粒度类别上的表现差异
- **`confusion_matrix(model, loader, device, num_classes)`** → `Tensor(100, 100)`：混淆矩阵（行=真实，列=预测）

### 3.6 `checkpoint.py` — 检查点管理

保存三种产物：

| 文件 | 内容 | 用途 |
|---|---|---|
| `resnet18_cifar100_best.pth` | epoch, model_state_dict, optimizer_state_dict, accuracy, num_classes | 恢复训练、完整推理 |
| `resnet18_cifar100_feature_extractor.pth` | 去掉 `fc.` 的 state_dict | **迁移学习** |
| `training_history.json` | 每 epoch 的 loss/acc/lr 列表 | 训练分析 |

特征提取器的提取逻辑：遍历 model 的 `state_dict()`，过滤掉所有以 `"fc."` 开头的 key。这样得到的权重只包含 stem + 4 组残差层的参数（特征提取部分），不包含分类头。

### 3.7 `visualize.py` — 可视化

生成三张图到 `checkpoints/plots/` 目录：

- **`training_curves.png`**：左图 loss 曲线（train/test），右图 accuracy 曲线（train/test）
- **`confusion_matrix.png`**：100×100 热力图（蓝色色阶）
- **`lr_schedule.png`**：学习率随 epoch 的变化（余弦曲线）

### 3.8 `main.py` — 主入口

编排完整流程：

```
解析命令行参数 → 构建配置 → 创建模型 → 加载数据
→ [可选] 超参数搜索（--search / --search-only）
→ [可选] 自动加载已有搜索结果
→ 训练 → 保存训练历史
→ 最终评估（top-1 accuracy + per-class accuracy + 混淆矩阵）
→ 生成可视化 → 验证迁移学习就绪
```

支持命令行覆盖配置：`--epochs`, `--batch-size`, `--lr`, `--data-root`, `--eval-only`。

搜索相关参数：`--search`（搜索+训练）、`--search-only`（仅搜索）、`--ignore-search`（忽略已有结果）。

---

## 4. 数据增强

### 4.1 概述

共 **19 种增强技术**，分为 **5 大类**。每张图片平均被施加 5~8 种增强（各自独立概率），保证多样性而不至于单张图片过度扭曲。

CIFAR-100 类别完全均衡（每类 500 训练 / 100 测试，CV=0.00%），**不需要类别平衡算法**。

### 4.2 增强技术一览

#### A. 几何变换 Geometric（PIL 级）

| # | 方法 | 参数 | 概率 | 作用 |
|---|---|---|---|---|
| 1 | `RandomCrop` | `32, padding=4, reflect` | 100% | 位置不变性，CIFAR 金标准 |
| 2 | `RandomHorizontalFlip` | `p=0.5` | 50% | 水平对称性 |
| 3 | `RandomAffine` | `degrees=15, translate=0.1, scale=0.9~1.1, shear=5` | 100% | 旋转+平移+缩放+剪切一体化 |
| 4 | `RandomPerspective` | `distortion=0.2` | 30% | 透视变形，模拟不同观察角度 |

#### B. 颜色变换 Color（PIL 级）

| # | 方法 | 参数 | 概率 | 作用 |
|---|---|---|---|---|
| 5 | `ColorJitter` | `brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15` | 100% | 亮度/对比度/饱和度/色调抖动 |
| 6 | `RandomGrayscale` | `p=0.1` | 10% | 灰度化，增强颜色不敏感特征 |
| 7 | `RandomAutocontrast` | `p=0.2` | 20% | 自动对比度增强 |
| 8 | `RandomEqualize` | `p=0.1` | 10% | 直方图均衡化 |
| 9 | `RandomPosterize` | `bits=4` | 10% | 色彩量化，模拟低色深 |
| 10 | `RandomSolarize` | `threshold=128` | 10% | 高亮像素反转 |

#### C. 噪声与降质 Noise & Degradation（Tensor 级）

| # | 方法 | 参数 | 概率 | 作用 |
|---|---|---|---|---|
| 11 | `GaussianNoise` | `std=0.02` | 50% | 高斯加性噪声，模拟传感器噪声 |
| 12 | `SaltPepperNoise` | `amount=0.01` | 20% | 椒盐噪声，模拟坏像素 |
| 13 | `ProbabilisticGaussianBlur` | `kernel_size=3` | 20% | 高斯模糊，模拟失焦 |
| 14 | `RandomErasing` | `p=0.25, scale=(0.02,0.2)` | 25% | 随机擦除矩形区域 |

#### D. 天气与压缩模拟 Weather & Compression

| # | 方法 | 参数 | 概率 | 作用 |
|---|---|---|---|---|
| 15 | `JPEGCompressionPIL` | `quality=30~70` | 20% | JPEG 压缩伪影（PIL 级） |
| 16 | `FogEffect` | `intensity=0.05~0.2` | 15% | 雾化效果，模拟大气散射 |
| 17 | `RainStreaks` | `drops=3~10, angle=±30°` | 15% | 雨滴条纹，模拟雨天遮挡 |

#### E. 批次级混合 Batch Mixing（训练循环内）

| # | 方法 | 参数 | 概率 | 作用 |
|---|---|---|---|---|
| 18 | `CutMix` | `alpha=1.0` | 40% | 区域裁剪混合 + 标签混合 |
| 19 | `Mixup` | `alpha=0.2` | 30% | 全图线性插值 + 标签混合 |

### 4.3 Transform 管线顺序

```
PIL Image (32×32)
│
├── [A] RandomCrop → RandomHorizontalFlip → RandomAffine → RandomPerspective
├── [B] ColorJitter → RandomGrayscale → RandomAutocontrast
│     → RandomEqualize → RandomPosterize → RandomSolarize
├── [D] JPEGCompressionPIL
│
├── ToTensor() → Normalize(mean, std)
│
├── [C] GaussianNoise → SaltPepperNoise → GaussianBlur
│     → FogEffect → RainStreaks → RandomErasing
│
└── → DataLoader → batch
     │
     └── train_one_epoch 内: CutMix / Mixup / identity (batch 级)
```

### 4.4 Soft labels 处理

CutMix/Mixup 产生 soft labels（float tensor `(B, num_classes)`）:
- `CrossEntropyLoss` 天然支持 float targets，无需修改 loss 计算
- accuracy 改用 dominant label 计算（`labels.argmax(1)`）
- CutMix/Mixup 激活时自动禁用 `label_smoothing`（soft labels 已提供类似正则化）

### 4.5 配置管理

增强参数独立为 `AugmentationConfig` frozen dataclass（约 40 个字段，全部有默认值），通过 `TrainConfig.augmentation` 引用。

- 总开关: `use_augmentation=True/False`
- 所有参数可通过 `dataclasses.replace()` 覆盖，预留超参数搜索可能性
- 设 `use_augmentation=False` 时，管线退化为仅 `ToTensor + Normalize`

### 4.6 `augment.py` 模块结构

```
augment.py
├── PIL 级自定义变换
│   └── JPEGCompressionPIL     — JPEG 压缩伪影模拟
│
├── Tensor 级自定义变换 (nn.Module)
│   ├── GaussianNoise           — 高斯加性噪声
│   ├── SaltPepperNoise         — 椒盐噪声
│   ├── ProbabilisticGaussianBlur — 带概率控制的高斯模糊
│   ├── FogEffect               — 雾化效果
│   └── RainStreaks             — 雨滴条纹
│
├── 批次级增强 (函数, 在 train_one_epoch 中调用)
│   ├── cutmix_data()           — CutMix 区域混合
│   ├── mixup_data()            — Mixup 线性插值
│   └── apply_batch_augmentation() — 随机选择: cutmix / mixup / identity
│
└── 管线构建
    ├── build_train_transforms() — 完整训练变换管线
    └── build_test_transforms()  — 仅归一化的测试管线
```

---

## 5. 训练流程

### 5.1 优化策略

| 组件 | 选择 | 原因 |
|---|---|---|
| **优化器** | 可配置（默认 SGD, momentum=0.9） | 通过 `create_optimizer()` 工厂函数支持 SGD/Adam/AdamW/RMSprop/NAdam |
| **学习率调度** | 可配置（默认 CosineAnnealingLR） | 通过 `create_scheduler()` 工厂函数支持 cosine/step/constant |
| **损失函数** | CrossEntropyLoss (label_smoothing=0.1) | 100 类分类中，标签平滑防止模型过度自信，提升泛化能力 |
| **正则化** | weight_decay=5e-4 + BN | L2 正则化配合 BatchNorm 是 ResNet 的标准配置 |
| **早停** | patience=20, min_delta=1e-4 | 防止过拟合，避免无效训练 |

### 5.2 学习率变化

余弦退火将学习率从 0.1 平滑降到接近 0：

```
lr(t) = 0.5 * 0.1 * (1 + cos(π * t / 200))
```

- Epoch 1: lr ≈ 0.1（快速学习）
- Epoch 100: lr ≈ 0.05（中期减速）
- Epoch 200: lr ≈ 0（精细收敛）

### 5.3 预期性能

在 RTX 4060 上，batch_size=128：
- 单 epoch 约 5-8 秒
- 200 epochs 约 15-25 分钟
- 预期 top-1 accuracy：~77-80%

---

## 6. 迁移学习准备

### 6.1 导出内容

训练过程中，每当测试准确率创新高时，自动保存特征提取器权重：

```python
# checkpoint.py 中 save_feature_extractor 的核心逻辑
feature_state = OrderedDict(
    (k, v) for k, v in model.state_dict().items()
    if not k.startswith("fc.")  # 过滤掉分类头
)
torch.save(feature_state, path)
```

导出的 `.pth` 文件包含 stem + layer1-4 的所有参数（约 11.17M - 51.3K ≈ 11.12M），不包含 FC 层。

### 6.2 CIFAR-10 迁移用法

```python
from src.Q3.model import create_model

# 1. 创建 10 类的模型（FC 层随机初始化）
model = create_model(num_classes=10)

# 2. 加载 CIFAR-100 训练的特征提取器权重
state = torch.load("checkpoints/resnet18_cifar100_feature_extractor.pth")
model.load_state_dict(state, strict=False)  # strict=False 允许 FC 缺失

# 3. 冻结特征提取器，只训练 FC 头（或全部微调）
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False  # 冻结

# 4. 在 CIFAR-10 上训练
# ... 使用类似的训练循环
```

### 6.3 迁移策略选择

| 策略 | 做法 | 适用场景 |
|---|---|---|
| **冻结 + 只训练 FC** | 冻结 stem+layer1-4，只更新 fc 层 | CIFAR-10 数据较少、快速迁移 |
| **全量微调** | 加载权重后全部参数都可训练 | CIFAR-10 数据充足、追求最佳性能 |
| **部分解冻** | 冻结 stem+layer1-2，只训练 layer3-4+fc | 折中方案，平衡速度和性能 |

---

## 7. 超参数搜索

使用 **skorch** 将 ResNet-18 包装为 sklearn 兼容估计器，然后用 **sklearn.model_selection** 的搜索工具做超参数搜索。sklearn 的搜索框架自带交叉验证、successive halving、标准评分指标，避免手写 fitness 函数的各种陷阱。

支持三种策略，通过 `SearchConfig.strategy` 或 `--search-strategy` 选择：

| 策略 | CLI 参数 | 特点 | 适用场景 |
|---|---|---|---|
| Halving-Random (默认) | `halving-random` | successive halving + 随机采样，最高效 | 大搜索空间，GPU 资源有限 |
| 随机 | `random` | 均匀随机采样 + 交叉验证 | 简单基线 |
| 网格 | `grid` | 笛卡尔积穷举 | 小搜索空间 |

### 7.1 设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 框架 | skorch + sklearn | 久经考验的 CV 框架，无手写 fitness bug |
| 优化器 | 固定 SGD | ResNet-18 标准；避免 Adam 不接受 momentum 等参数不兼容问题 |
| 增强搜索阶段 | 无增强（仅 Normalize） | 搜索目的是找 optimizer 参数，干净信号更可靠 |
| 数据格式 | numpy 数组 | skorch/sklearn 原生支持，CV 用索引切分 |
| 调度器 | 无（搜索阶段） | 搜索轮数短（2-20），scheduler 不适用 |
| label_smoothing | 无（搜索阶段） | 无 CutMix/Mixup 时意义不大 |
| CV | 3-fold 交叉验证 | sklearn 自带 train/val split，不碰 test set |

### 7.2 Successive Halving（HalvingRandomSearchCV）

默认策略的工作流程：

1. **第 1 轮**：50 个随机候选 × 2 epochs → 按 accuracy 保留前 1/3（~17 个）
2. **第 2 轮**：17 个候选 × 6 epochs → 保留前 1/3（~6 个）
3. **第 3 轮**：6 个候选 × 18 epochs → 最终最优

总训练量：每轮 × 候选数 × epochs，远小于网格搜索的穷举量。

### 7.3 搜索空间

| 超参数 | skorch 参数名 | 类型 | 范围 | TrainConfig 映射 |
|---|---|---|---|---|
| 学习率 | `lr` | loguniform | [1e-4, 1.0] | `learning_rate` |
| 动量 | `optimizer__momentum` | uniform | [0.85, 0.99] | `momentum` |
| 权重衰减 | `optimizer__weight_decay` | loguniform | [1e-6, 1e-2] | `weight_decay` |
| 批大小 | `batch_size` | 离散 | [128, 256, 512] | `batch_size` |

连续参数使用 `scipy.stats` 分布（loguniform / uniform），离散参数使用候选列表。

### 7.4 数据准备

`_prepare_search_data()` 将 CIFAR-100 训练集转为 numpy 数组：

```python
X, y = _prepare_search_data(config)
# X: (50000, 3, 32, 32) float32, 已归一化
# y: (50000,) int64
```

归一化使用与正式训练相同的 CIFAR-100 统计量（mean/std）。不使用任何数据增强。

### 7.5 输出文件

搜索结果保存为 `checkpoints/hp_search_results.json`：

```json
{
  "search_config": {
    "strategy": "halving-random",
    "total_candidates": 50,
    "cv": 3
  },
  "best": {
    "params": {
      "lr": 0.032,
      "optimizer__momentum": 0.93,
      "optimizer__weight_decay": 2.1e-4,
      "batch_size": 256
    },
    "mean_test_score": 0.352
  },
  "all_candidates": [
    { "params": {...}, "mean_test_score": 0.352, "rank": 1 },
    ...
  ]
}
```

### 7.6 参数映射与训练集成

`run_search()` 返回已映射为 TrainConfig 字段名的参数字典：

```python
# skorch 参数名 → TrainConfig 字段名
PARAM_MAP = {
    "lr": "learning_rate",
    "optimizer__momentum": "momentum",
    "optimizer__weight_decay": "weight_decay",
    "batch_size": "batch_size",
}
```

`main.py` 直接用 `dataclasses.replace(config, **best_params)` 应用搜索结果。

### 7.7 训练时自动加载

`main.py` 默认检测 `hp_search_results.json`：
- 文件存在 → 自动应用最优参数
- `--ignore-search` → 忽略搜索结果

---

## 8. 使用方法

### 超参数搜索

```bash
# 仅运行搜索（推荐：先搜索一次，默认 halving-random）
uv run python src/Q3/main.py --search-only

# 搜索 + 用最优配置训练
uv run python src/Q3/main.py --search

# 指定搜索策略
uv run python src/Q3/main.py --search-only --search-strategy random
uv run python src/Q3/main.py --search-only --search-strategy grid
```

### 自动使用搜索结果

```bash
# 搜索完成后，正常训练会自动加载最优参数
uv run python src/Q3/main.py

# 忽略搜索结果，使用默认/CLI参数
uv run python src/Q3/main.py --ignore-search
```

### 运行测试

```bash
cd AIhomeworks
uv sync  # 安装依赖（torch, torchvision, matplotlib, pytest）
```

### 完整训练（200 epochs）

```bash
uv run python src/Q3/main.py

# 禁用数据增强
uv run python src/Q3/main.py --no-augmentation
```

### 快速测试（2 epochs）

```bash
uv run python src/Q3/main.py --epochs 2
```

### 自定义参数

```bash
uv run python src/Q3/main.py --epochs 100 --batch-size 64 --lr 0.05
```

### 运行测试

```bash
uv run pytest src/Q3/tests/ -v
```

### 训练产物

训练完成后，`checkpoints/` 目录下会生成：

```
checkpoints/
├── resnet18_cifar100_best.pth                # 最佳完整模型
├── resnet18_cifar100_feature_extractor.pth   # 特征提取器（迁移学习用）
├── training_history.json                     # 训练历史
├── hp_search_results.json                    # 超参数搜索结果
└── plots/
    ├── training_curves.png                   # Loss/Accuracy 曲线
    ├── confusion_matrix.png                  # 混淆矩阵
    └── lr_schedule.png                       # 学习率曲线
```
