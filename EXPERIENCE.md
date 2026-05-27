# Experience & Lessons Learned / 经验与教训

## Architecture / 架构

- **CIFAR 适配 stem**: 标准 ResNet-18 的 7x7 conv stride=2 + maxpool 会把 32x32 图像压缩到 8x8，后续层进一步压缩到 1x1。改为 3x3 conv stride=1 + 去掉 maxpool 后，特征图变化为 32→32→16→8→4，保留了足够的空间信息。
- **迁移学习准备**: 只需保存去掉 `fc.` 前缀的 state_dict，迁移时 `load_state_dict(..., strict=False)` 即可，FC 层保持随机初始化。

## Windows Notes / Windows 注意事项

- DataLoader `num_workers > 0` 必须在 `if __name__ == "__main__"` 守卫内使用，否则会触发无限递归。
- PyTorch CUDA 需要通过 `[[tool.uv.index]]` 配置 wheel index，默认 PyPI 提供 CPU-only 版本。

## Hyperparameter Search / 超参数搜索

- **手写 fitness 函数是坑**：第一版用 `10 * AIR + LDR`（改善速率），偏好学得快的配置。第二版改为 `val_acc - penalty * overfit_gap`，但 5 epoch 探针下 train/val gap 噪声大，penalty=1.0 过激，搜索选出的参数还不如默认值。最终用 skorch + sklearn 框架替代，让 sklearn 处理 CV、scoring、successive halving。
- **CutMix/Mixup 下 train_acc 不可靠**：`train_one_epoch` 用 soft labels 的 dominant class 算准确率，但 `mix_prob=0.7` 意味着 70% batch 的 train_acc 在错误基准上。如果 fitness 依赖 train_acc（如 overfit gap），必须禁用 batch augmentation 或用 `evaluate()` 单独评估。
- **搜索阶段不需要增强**：搜索目的是找 optimizer 参数（lr, momentum, weight_decay），不是评估增强效果。不用增强使信号更干净，搜索更可靠。
- **固定 SGD 搜索更安全**：不同 optimizer 接受不同的参数（Adam 不接受 momentum），混搜会导致 TypeError。固定 SGD 是 ResNet-18 的标准选择。
- **sklearn HalvingRandomSearchCV 很高效**：50 个候选 × successive halving（2→6→18 epochs），总训练量远小于穷举，自动分配更多资源给有希望的配置。
- **skorch `train_split=False`**：使用 sklearn 的 CV 做数据划分时，必须禁用 skorch 内部的 train_split，否则会双重 split。

## Training / 训练

- **batch_size=1024 对小数据集有效**：CIFAR-100 训练集 50k 样本，batch=1024 时每 epoch 仅 ~49 步更新。配合 lr=0.1 + cosine annealing + label smoothing=0.1，收敛稳定。但搜索空间中应包含较小 batch（128/256）作为候选。
- **CutMix/Mixup 激活时禁用 label_smoothing**：soft labels 已提供类似正则化效果，同时用 label_smoothing 是双重正则化，`train.py` 中已自动处理。

## Transfer Learning / 迁移学习

- **sklearn CV 会 clone 估计器**：sklearn 交叉验证每次 fold 都会 clone 估计器，skorch 随之重新创建模块。迁移学习中必须确保每次重建后都加载预训练权重并冻结 backbone。方案：子类化 `NeuralNetClassifier`，覆写 `initialize_module()` 在模块创建后自动注入预训练权重。
- **`module__` 前缀传递参数**：skorch 通过 `module__param` 前缀将参数传递给底层模块。`source_checkpoint` 通过 `module__source_checkpoint` 传入，避免在 `__init__` 中作为构造参数（防止 sklearn clone 序列化问题）。
- **保留原 FC + 追加新分类层优于替换 FC**：不替换原始 FC（512→100），而是保留它作为特征投影层并在其后追加新分类层（100→10）。原 FC 从 CIFAR-100 训练中学到的 100 维语义空间对新任务有用，微调比从零训练更好。可训练参数 = 52,310（原 FC 51,300 + 新分类层 1,010），而非之前方案的 5,130。
- **冻结参数必须过滤优化器**：`create_optimizer` 改为 `[p for p in model.parameters() if p.requires_grad]`，否则优化器仍会为冻结参数分配动量/梯度状态，浪费内存且 `step()` 计算无意义。
- **TransferConfig 与 TrainConfig 分开**：迁移学习的超参数默认值与全量训练差异大（lr 0.01 vs 0.1，epochs 30 vs 100，batch 256 vs 1024），不应共用默认值。用 `_to_train_config()` 桥接两者（取同名字段）。

## Run Isolation / 运行隔离

- **时间戳目录避免覆盖**：检查点文件名硬编码（如 `resnet18_cifar100_best.pth`）导致新训练覆盖旧结果。用 `YYYY-MM-DD_HHMMSS` 格式子目录隔离每次运行，格式不含冒号（Windows 目录名安全）。
- **frozen dataclass 的时间戳注入**：`TrainConfig` / `TransferConfig` 是 frozen dataclass，不能在构造后修改 `checkpoint_dir`。必须在 `main.py` 构造前生成时间戳，通过 `dataclasses.replace()` 注入。
- **自动选基础模型按准确率而非最新**：迁移学习选源模型时，应比较各运行的 `accuracy` 字段（存在检查点中），而非简单选最新的。同准确率时才按时间戳取最新。
- **数据集感知文件名**：同一时间戳目录内，CIFAR-100 和 CIFAR-10 的检查点通过 `dataset_prefix(num_classes)` 生成不同前缀（`resnet18_cifar100_*` vs `resnet18_cifar10_*`），避免混淆。

## Torchvision Pretrained Transfer / PyTorch 预训练迁移学习

- **torchvision ResNet-18 输入 224x224**：官方预训练模型的 stem 是 7x7 conv stride=2 + maxpool，设计用于 224x224 输入。CIFAR-10 的 32x32 图像需要通过 `Resize(224)` 上采样，这增加了计算量但保留了预训练特征的完整空间结构。
- **ImageNet 归一化必须匹配**：使用 torchvision 预训练模型时，必须用 ImageNet 的归一化统计量（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]），而非 CIFAR-10 的统计量。否则预训练特征无法正确激活。
- **FC 参数少（5,130）**：torchvision ResNet-18 的 FC 是 Linear(512, 10)，冻结 backbone 后仅 5,130 可训练参数。相比 CIFAR-100 自训练迁移（52,310 参数），训练更快但对数据量要求更低。
- **`train()` 函数通用性**：已有的 `train()` 函数完全不感知模型来源，只通过 `requires_grad` 过滤可训练参数，因此 torchvision 模型可以直接复用同一训练循环、检查点保存、早停逻辑。

## Dropout / Dropout 正则化

- **Dropout 位置选择**：放在 `AdaptiveAvgPool2d` 之后、FC 之前是 ResNet 的标准做法。不在 BasicBlock 内部加 Dropout，因为残差连接 + BN 已提供足够正则化，Block 内 Dropout 反而可能破坏 shortcut 的信息流。
- **Dropout 不增加参数量**：`nn.Dropout` 没有可学习参数，不影响 `state_dict` 结构，也不会影响预训练权重加载（迁移学习无需调整）。
- **默认启用 + 搜索覆盖**：默认 `dropout_rate=0.5`，超参数搜索空间包含 `module__dropout_rate`。搜索过就用搜索结果，否则用配置默认值。

## skorch 训练包装 / skorch Training Wrapper

- **skorch 替代手写训练循环**：`ClassifierNet(NeuralNetClassifier)` 替代了手写 `train()` 函数。skorch 自动提供 epoch 计时（`history['dur']`）、train/valid loss/acc、early stopping、checkpoint。不再需要手动管理 epoch 循环和指标累积。
- **CutMix/Mixup 通过覆写 `train_step_single()` 实现**：在 `train_step_single()` 中对 batch 应用批次级增强，soft labels 由 `CrossEntropyLoss` 原生支持。不需要修改 skorch 的整体训练流程。
- **测试集作为验证集**：skorch 默认用 `train_split` 划分训练数据。我们用 `make_fixed_split(test_dataset)` 闭包，将独立的测试集作为验证集传给 skorch，保持与原有一致的 train/test 分离。
- **skorch 的 `EarlyStopping` 通过 `KeyboardInterrupt` 终止训练**：`fit_loop` 捕获该异常并优雅退出。设置 `load_best=True` 确保训练结束后模型是最优权重。
- **自定义 Callback 替代 skorch 内置 Checkpoint**：skorch 的 `Checkpoint` 不保存 accuracy/epoch/num_classes 元数据，迁移学习需要这些字段。用 `CustomCheckpoint` 回调保存自定义格式。
- **Label smoothing + CutMix/Mixup 互斥**：在 `create_classifier_net()` 工厂中检测批次增强是否激活，激活时自动将 `label_smoothing` 设为 0。

## 代码提取与包组织 / Code Extraction & Package Organization

- **三层架构：utils → Q2 → Q3**：`utils/` 存放跨作业共享基础设施，`Q2/` 存放 ResNet-18 训练管线，`Q3/` 仅保留 CIFAR-100 特有配置和迁移学习代码。Q1（VGG-16）只引用 utils。
- **`build_train_transforms` 签名解耦**：从 `(aug_config, config)` 改为 `(aug_config, mean, std)`，消除对 `TrainConfig` 的依赖，不同作业可以传入不同的归一化统计量。
- **搜索模块分离数据准备和搜索逻辑**：`utils/search.run_search()` 接受预处理好的 numpy 数组，不关心数据来源。每个作业只需准备自己的数据然后调用通用搜索。
- **`train.py` 保留给迁移学习**：迁移学习的训练流程（加载预训练权重→冻结 backbone→仅训练 FC）与标准训练不同，暂时保留旧的 `train()` 函数。后续可统一为 skorch 管线。
- **pytest `pythonpath = ["src"]`**：将 `src/` 加入 Python 路径后，`Q2` 和 `utils` 可以直接 import，不再依赖 `sys.path.insert()`。
