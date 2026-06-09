"""
通用 skorch 分类训练器。
Generic skorch NeuralNetClassifier for training any classification model.

设计思路:
- 接受任意 PyTorch 模型类（ResNet-18, VGG-16 等）
- 内置计时（EpochTimer → history['dur']）
- 覆写 train_step_single() 支持 CutMix/Mixup 批次增强
- 通过 make_fixed_split() 将测试集作为验证集
- create_classifier_net() 工厂自动配置 callbacks

Q2 传入 ResNet18，Q1 可传入 VGG16。
"""

import dataclasses

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler

from .augment import apply_batch_augmentation
from .callbacks import (
    CustomCheckpoint,
    FeatureExtractorCheckpoint,
    LRRecorder,
    TrainingHistory,
)


def _serialize_config(config) -> dict:
    """将鸭子类型配置转为 JSON-safe 字典，供 CustomCheckpoint 保存。"""
    from .config import config_to_dict
    # 配置是 frozen dataclass 时直接序列化
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return config_to_dict(config)
    # 非 dataclass（不应出现）：返回空 dict
    return {}


class ClassifierNet(NeuralNetClassifier):
    """
    通用 skorch 分类训练器。
    Generic skorch classifier with CutMix/Mixup augmentation support.

    覆写 train_step_single() 以在前向传播前应用批次级增强。
    CutMix/Mixup 产生的 soft labels（float tensor）由 CrossEntropyLoss 原生支持。

    Required config fields (duck-typed, any object with these attributes):
      optimizer_type, learning_rate, momentum, weight_decay, scheduler_type,
      epochs, scheduler_t_max, label_smoothing, use_amp, augmentation,
      num_classes, patience, min_delta, dropout_rate, checkpoint_dir,
      batch_size, num_workers, pin_memory
    """

    def __init__(
        self,
        module,
        *,
        aug_config=None,
        train_num_classes: int = 10,
        **kwargs,
    ):
        """
        Args:
            module:           PyTorch 模型类（如 ResNet18）
            aug_config:       AugmentationConfig，为 None 时不增强
            train_num_classes: 用于 CutMix/Mixup 的类别数
            **kwargs:         传递给 NeuralNetClassifier 的参数
        """
        super().__init__(module, **kwargs)
        self.aug_config = aug_config
        self.train_num_classes = train_num_classes

    def train_step_single(self, batch, **fit_params):
        """
        覆写：在前向传播前应用 CutMix/Mixup 批次级增强。
        Override: apply CutMix/Mixup batch augmentation before forward pass.
        """
        Xi, yi = batch
        # Batch-level augmentation (CutMix/Mixup)
        if self.aug_config and self.aug_config.use_augmentation:
            Xi, yi = apply_batch_augmentation(
                Xi, yi, self.aug_config, self.train_num_classes
            )
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        return {"loss": loss, "y_pred": y_pred}


def make_fixed_split(test_dataset):
    """
    创建 train_split 闭包：训练集原样返回，测试集作为验证集。
    Create a train_split closure that uses a separate test set as validation.

    skorch 的 train_split 签名因版本而异，使用 *args/**kwargs 兼容。

    Usage:
        net = ClassifierNet(
            ...,
            train_split=make_fixed_split(test_dataset),
        )
    """

    def split(dataset, *args, **kwargs):
        # skorch 不同版本传递 (dataset, y, net) 或 (dataset, **fit_params)
        return dataset, test_dataset

    return split


def create_classifier_net(
    model_class,
    config,
    train_dataset,
    test_dataset,
    save_feature_extractor: bool = False,
):
    """
    工厂：从配置创建完整配置的 skorch 分类器。
    Factory: create a fully configured skorch classifier from config.

    自动配置:
    - EarlyStopping（监控 valid_acc）
    - CustomCheckpoint（自定义格式保存最优模型）
    - FeatureExtractorCheckpoint（可选，保存特征提取器）
    - LRScheduler（CosineAnnealingLR）
    - EpochScoring（train_acc）
    - LRRecorder（记录学习率）
    - TrainingHistory（训练结束保存 JSON）

    Args:
        model_class:           模型类（如 ResNet18）
        config:                训练配置（duck-typed）
        train_dataset:         训练集 Dataset
        test_dataset:          测试集 Dataset（作为验证集）
        save_feature_extractor: 是否额外保存特征提取器（迁移学习需要）

    Returns:
        配置好的 ClassifierNet 实例，调用 .fit(train_dataset, y=None) 开始训练
    """
    aug_config = config.augmentation

    # CutMix/Mixup 激活时禁用 label_smoothing（soft labels 已提供正则化）
    batch_mix_active = (
        aug_config.use_augmentation
        and (aug_config.use_cutmix or aug_config.use_mixup)
        and aug_config.mix_prob > 0
    )
    label_smoothing = 0.0 if batch_mix_active else config.label_smoothing

    # 构建 callbacks
    # 非条件化的 callbacks（始终启用）
    callbacks = [
        # 训练准确率（默认只有 valid_acc）
        (
            "train_acc",
            EpochScoring(
                "accuracy",
                name="train_acc",
                on_train=True,
                lower_is_better=False,
            ),
        ),
        # 学习率记录
        ("lr_recorder", LRRecorder()),
    ]

    # 学习率调度（可条件化关闭）
    # getattr 保证缺少字段时默认 True（后向兼容 Q1/Q3 旧配置）
    if getattr(config, "use_scheduler", True):
        callbacks.append((
            "lr_scheduler",
            LRScheduler(
                policy=torch.optim.lr_scheduler.CosineAnnealingLR,
                T_max=config.scheduler_t_max,
            ),
        ))

    # 早停（可条件化关闭）
    if getattr(config, "use_early_stopping", True):
        callbacks.append((
            "early_stop",
            EarlyStopping(
                monitor="valid_acc",
                patience=config.patience,
                threshold=config.min_delta,
                lower_is_better=False,
                load_best=True,
            ),
        ))

    # 始终启用的 callbacks
    callbacks.extend([
        # 自定义格式检查点
        (
            "custom_checkpoint",
            CustomCheckpoint(
                checkpoint_dir=config.checkpoint_dir,
                num_classes=config.num_classes,
                task_tag=getattr(config, "task_tag", ""),
                model_name=getattr(config, "model_name", "resnet18"),
                config_dict=_serialize_config(config),
            ),
        ),
        # 训练历史 JSON 保存
        (
            "training_history",
            TrainingHistory(checkpoint_dir=config.checkpoint_dir),
        ),
    ])

    # 可选：特征提取器保存（迁移学习需要）
    if save_feature_extractor:
        callbacks.append(
            (
                "feature_extractor",
                FeatureExtractorCheckpoint(
                    checkpoint_dir=config.checkpoint_dir,
                    num_classes=config.num_classes,
                    task_tag=getattr(config, "task_tag", ""),
                    model_name=getattr(config, "model_name", "resnet18"),
                ),
            )
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = ClassifierNet(
        model_class,
        module__num_classes=config.num_classes,
        module__dropout_rate=config.dropout_rate,
        module__use_bn=getattr(config, "use_bn", True),
        criterion=nn.CrossEntropyLoss,
        criterion__label_smoothing=label_smoothing,
        optimizer=torch.optim.SGD,
        lr=config.learning_rate,
        optimizer__momentum=config.momentum,
        optimizer__weight_decay=config.weight_decay,
        max_epochs=config.epochs,
        batch_size=config.batch_size,
        iterator_train__shuffle=True,
        iterator_train__num_workers=config.num_workers,
        iterator_train__pin_memory=config.pin_memory,
        iterator_valid__num_workers=config.num_workers,
        iterator_valid__pin_memory=config.pin_memory,
        train_split=make_fixed_split(test_dataset),
        classes=list(range(config.num_classes)),
        device=device,
        callbacks=callbacks,
        verbose=1,
        # Custom params / 自定义参数
        aug_config=aug_config,
        train_num_classes=config.num_classes,
    )

    return net
