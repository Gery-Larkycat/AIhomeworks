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
