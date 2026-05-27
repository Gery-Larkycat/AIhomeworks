# utils — 跨作业共享基础设施

Q1（VGG-16 CIFAR-10）、Q2（ResNet-18 CIFAR-10）、Q3（ResNet-18 CIFAR-100 + 迁移学习）共用的基础模块。不依赖任何作业特定的代码，仅依赖 PyTorch / skorch / sklearn / matplotlib 等第三方库。

## 目录

```
utils/
├── config.py       # 配置类型、常量、辅助函数
├── augment.py      # 19 种增强技术 + 管线构建
├── net.py          # skorch 通用分类训练器
├── callbacks.py    # 自定义 skorch 回调
├── evaluate.py     # 评估指标
├── visualize.py    # 训练曲线、混淆矩阵、学习率图
└── search.py       # 通用超参搜索
```

---

## 模块详解

### `config.py` — 共享配置组件

定义跨作业复用的配置类型、数据集常量和辅助函数。

**数据集归一化常量**：

| 常量 | 值 | 用途 |
|---|---|---|
| `CIFAR10_MEAN` / `CIFAR10_STD` | `(0.4914, 0.4822, 0.4465)` / `(0.2470, 0.2435, 0.2616)` | Q2/Q3 迁移 CIFAR-10 归一化 |
| `CIFAR100_MEAN` / `CIFAR100_STD` | `(0.5071, 0.4867, 0.4408)` / `(0.2675, 0.2565, 0.2761)` | Q3 CIFAR-100 归一化 |
| `IMAGENET_MEAN` / `IMAGENET_STD` | `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` | Q3 torchvision 迁移 ImageNet 归一化 |

**`AugmentationConfig`**（frozen dataclass）：19 种增强技术的完整配置，约 40 个字段，全部有默认值。分为 5 大类：几何变换（A）、颜色变换（B）、噪声与降质（C）、天气与压缩（D）、批次级混合（E）。总开关 `use_augmentation`。

**`SearchConfig`**（frozen dataclass）：超参搜索策略配置（halving-random / random / grid），含 successive halving 参数、候选数、CV 折数、batch_size 候选列表。

**辅助函数**：

- `generate_timestamp()` → `"YYYY-MM-DD_HHMMSS"` 格式时间戳（Windows 安全无冒号）
- `make_run_dir(base, timestamp)` → 构造 `checkpoints/<timestamp>` 运行目录路径
- `dataset_prefix(num_classes)` → 检查点文件名前缀（100 → `resnet18_cifar100`，10 → `resnet18_cifar10`）

---

### `augment.py` — 数据增强

19 种增强技术 + 变换管线构建器。

**自定义变换类**（PIL 级）：`JPEGCompressionPIL`

**自定义变换类**（Tensor 级，`nn.Module`）：`GaussianNoise`、`SaltPepperNoise`、`ProbabilisticGaussianBlur`、`FogEffect`、`RainStreaks`

**批次级增强**（函数，在训练循环中调用）：
- `cutmix_data(X, y, ...)` — CutMix 区域裁剪混合
- `mixup_data(X, y, ...)` — Mixup 线性插值
- `apply_batch_augmentation(X, y, aug_config, num_classes)` — 随机选择 CutMix / Mixup / identity

**管线构建**：
- `build_train_transforms(aug_config, mean, std)` — 完整训练变换管线（19 种增强 + 归一化）
- `build_test_transforms(mean, std)` — 测试管线（仅 ToTensor + Normalize）

签名接受 `(mean, std)` 参数而非 config 对象，消除对 `TrainConfig` 的依赖。

**管线顺序**：PIL 级几何变换 → PIL 级颜色变换 → PIL 级天气模拟 → ToTensor → Normalize → Tensor 级噪声/天气 → RandomErasing → DataLoader → 批次级 CutMix/Mixup

---

### `net.py` — skorch 通用分类训练器

用 skorch `NeuralNetClassifier` 包装任意 PyTorch 模型的通用训练器。Q2 传入 `ResNet18`，Q1 可传入 `VGG16`。

**`ClassifierNet(NeuralNetClassifier)`**：
- 接受 `aug_config` 和 `train_num_classes` 参数
- 覆写 `train_step_single()` 以在前向传播前应用 CutMix/Mixup 批次级增强
- Soft labels（float tensor）由 `CrossEntropyLoss` 原生支持

**`make_fixed_split(test_dataset)`**：
- 返回 `train_split` 闭包，将独立测试集作为 skorch 验证集
- 签名 `split(dataset, *args, **kwargs)` 兼容不同 skorch 版本

**`create_classifier_net(model_class, config, train_dataset, test_dataset, save_feature_extractor=False)`**：
- 工厂函数：从鸭子类型配置创建完整配置的 skorch 分类器
- 自动配置回调：EarlyStopping、LRScheduler(CosineAnnealingLR)、CustomCheckpoint、EpochScoring(train_acc)、LRRecorder、TrainingHistory
- 自动处理 CutMix/Mixup + label_smoothing 互斥（激活时 label_smoothing=0）
- 设置 `classes=list(range(num_classes))` 以支持 `y=None` 拟合
- 配置字段通过鸭子类型访问，任何具有所需属性的对象均可传入

**配置要求的字段**（鸭子类型）：`optimizer_type`、`learning_rate`、`momentum`、`weight_decay`、`scheduler_type`、`epochs`、`scheduler_t_max`、`label_smoothing`、`use_amp`、`augmentation`、`num_classes`、`patience`、`min_delta`、`dropout_rate`、`checkpoint_dir`、`batch_size`、`num_workers`、`pin_memory`

---

### `callbacks.py` — 自定义 skorch 回调

skorch 内置 `Checkpoint` 不保存 accuracy/epoch/num_classes 元数据，迁移学习需要这些字段，因此用自定义回调替代。

**`CustomCheckpoint`**：监控 `valid_acc_best`，仅在新最优时保存检查点。格式：`{epoch, model_state_dict, optimizer_state_dict, accuracy, num_classes}`。

**`FeatureExtractorCheckpoint`**：监控 `valid_acc_best`，保存去掉 FC 层的特征提取器权重（过滤 `fc.` 前缀键）。用于迁移学习。

**`LRRecorder`**：每个 epoch 记录当前学习率到 `history`（`history.record("lr", lr)`）。

**`TrainingHistory`**：训练结束后调用 `extract_history()` 导出标准 dict 并保存为 `training_history.json`。

**`extract_history(net)`**：将 skorch `history_` 转换为标准 dict 格式。字段映射：`train_loss → train_loss`、`valid_loss → test_loss`、`train_acc → train_acc`、`valid_acc → test_acc`、`dur → dur`、`lr → lr`。

---

### `evaluate.py` — 评估指标

三个纯函数，全部使用 `@torch.no_grad()` 装饰器。

**`evaluate(model, loader, device, use_amp=False)`** → `(loss, accuracy)`：
- 全局 top-1 评估，GPU 张量累积避免每 batch 同步
- 保留给旧训练循环（`train.py`）兼容，skorch 管线的评估由内置评分处理

**`per_class_accuracy(model, loader, device, num_classes)`** → `dict[int, float]`：
- 每个类别的准确率

**`confusion_matrix(model, loader, device, num_classes)`** → `Tensor(N, N)`：
- 混淆矩阵（行=真实标签，列=预测标签）

---

### `visualize.py` — 可视化

生成三张图到 `checkpoints/plots/` 目录（或指定 `save_dir`）。

- `plot_training_curves(history, save_dir)` — 左图 loss 曲线、右图 accuracy 曲线
- `plot_confusion_matrix(cm, save_dir, title, max_labels)` — 热力图（蓝色色阶）
- `plot_lr_schedule(history, save_dir)` — 学习率随 epoch 变化

Windows CJK 字体兼容（SimHei / Microsoft YaHei）。

---

### `search.py` — 通用超参搜索

基于 skorch + sklearn 的通用搜索模块。每个作业只需准备 (X, y) numpy 数据，传入模型类即可。

**公共 API**：

- `prepare_search_data(dataset, mean, std)` → `(X, y)`：torchvision Dataset 转归一化 numpy 数组
- `run_search(X, y, model_class, model_kwargs, search_cfg, checkpoint_dir, num_workers)` → `dict`：执行搜索并返回映射后的最优参数
- `load_best_search_params(checkpoint_dir, valid_fields)` → `dict | None`：从 JSON 加载最优参数

**搜索空间**（硬编码，适用于 ResNet-18）：
- `lr`: loguniform(1e-4, 1.0)
- `optimizer__momentum`: uniform(0.85, 0.99)
- `optimizer__weight_decay`: loguniform(1e-6, 1e-2)
- `batch_size`: 离散候选列表
- `module__dropout_rate`: uniform(0.0, 0.5)

**参数映射**（`PARAM_MAP`）：skorch 参数名 → 配置字段名（如 `lr → learning_rate`），返回结果可直接用于 `dataclasses.replace()`。

**设计决策**：
- 固定 SGD（ResNet-18 标准），避免不同 optimizer 参数不兼容
- 搜索阶段无增强、无 scheduler、无 label_smoothing（干净信号）
- `train_split=False`（sklearn CV 负责数据划分）
- 支持 `halving-random`（默认）、`random`、`grid` 三种策略
