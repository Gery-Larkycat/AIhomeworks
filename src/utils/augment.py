"""
数据增强模块: 19 种增强技术，5 大类。
Data augmentation module: 19 techniques across 5 categories.

Custom transforms for PIL-level and tensor-level augmentation,
batch-level CutMix/Mixup, and a pipeline builder.

Categories / 大类:
  A. Geometric (几何变换): RandomCrop, HFlip, Affine, Perspective
  B. Color (颜色变换): ColorJitter, Grayscale, AutoContrast,
     Equalize, Posterize, Solarize
  C. Noise & Degradation (噪声与降质): GaussianNoise, SaltPepper,
     GaussianBlur, RandomErasing
  D. Weather & Compression (天气与压缩): JPEGCompression, Fog, Rain
  E. Batch Mixing (批次级混合): CutMix, Mixup
"""

import io
import math
import random
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .config import AugmentationConfig


# ===========================================================================
# D. PIL-level custom transforms (JPEG Compression)
#    PIL 级自定义变换（JPEG 压缩）
# ===========================================================================


class JPEGCompressionPIL:
    """
    JPEG 压缩伪影模拟（PIL 级）。
    JPEG compression artifact simulation at PIL level.

    Takes PIL Image, returns PIL Image (possibly JPEG-compressed).

    Args:
        quality_range: JPEG 质量范围 (1-95) / Quality range
        p:             应用概率 / Application probability
    """

    def __init__(
        self,
        quality_range: tuple[int, int] = (30, 70),
        p: float = 0.2,
    ):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img

        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(quality_range={self.quality_range}, p={self.p})"
        )


# ===========================================================================
# C. Custom tensor-level transforms (nn.Module)
#    自定义 tensor 级变换
# ===========================================================================


class GaussianNoise(nn.Module):
    """
    高斯加性噪声，模拟传感器噪声。
    Additive Gaussian noise to simulate sensor noise.

    Args:
        std: 噪声标准差 / Noise standard deviation
        p:   应用概率 / Application probability
    """

    def __init__(self, std: float = 0.02, p: float = 0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise


class SaltPepperNoise(nn.Module):
    """
    椒盐噪声，模拟坏像素。
    Salt-and-pepper noise to simulate dead pixels.

    Args:
        amount: 被污染像素比例 / Fraction of pixels affected
        p:      应用概率 / Application probability
    """

    def __init__(self, amount: float = 0.01, p: float = 0.2):
        super().__init__()
        self.amount = amount
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        result = tensor.clone()
        _, h, w = result.shape

        num_salt = max(1, int(self.amount * h * w * 0.5))
        num_pepper = max(1, int(self.amount * h * w * 0.5))

        for _ in range(num_salt):
            i = random.randint(0, h - 1)
            j = random.randint(0, w - 1)
            result[:, i, j] = result.max()

        for _ in range(num_pepper):
            i = random.randint(0, h - 1)
            j = random.randint(0, w - 1)
            result[:, i, j] = result.min()

        return result


class ProbabilisticGaussianBlur(nn.Module):
    """
    带概率控制的高斯模糊。
    Gaussian blur with probability control.

    torchvision.transforms.GaussianBlur has no `p` parameter,
    so we wrap it with random chance.

    Args:
        kernel_size: 高斯核大小 / Gaussian kernel size
        p:           应用概率 / Application probability
    """

    def __init__(self, kernel_size: int = 3, p: float = 0.2):
        super().__init__()
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size)
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        return self.blur(tensor)


class FogEffect(nn.Module):
    """
    雾化效果，模拟大气散射。
    Fog effect to simulate atmospheric scattering.

    Blends image toward a per-channel mean value.

    Args:
        intensity_range: 雾强度范围 / Fog intensity range
        p:               应用概率 / Application probability
    """

    def __init__(
        self,
        intensity_range: tuple[float, float] = (0.05, 0.2),
        p: float = 0.15,
    ):
        super().__init__()
        self.intensity_range = intensity_range
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        intensity = random.uniform(*self.intensity_range)
        fog = tensor.mean(dim=(-2, -1), keepdim=True).clamp(min=0.5)
        return tensor * (1 - intensity) + fog * intensity


class RainStreaks(nn.Module):
    """
    雨滴条纹，模拟雨天遮挡。
    Rain streak simulation to simulate rainy weather occlusion.

    Draws slanted bright lines at random positions.

    Args:
        drops_range: 雨滴数量范围 / Number of rain drops range
        angle_range: 倾斜角度范围（度）/ Slant angle range (degrees)
        p:           应用概率 / Application probability
    """

    def __init__(
        self,
        drops_range: tuple[int, int] = (3, 10),
        angle_range: tuple[float, float] = (-30.0, 30.0),
        p: float = 0.15,
    ):
        super().__init__()
        self.drops_range = drops_range
        self.angle_range = angle_range
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor

        result = tensor.clone()
        _, h, w = result.shape
        num_drops = random.randint(*self.drops_range)
        angle_deg = random.uniform(*self.angle_range)
        dx = math.sin(math.radians(angle_deg))
        dy = math.cos(math.radians(angle_deg))

        for _ in range(num_drops):
            x0 = random.randint(0, w - 1)
            y0 = random.randint(0, h - 1)
            length = random.randint(
                max(1, int(h * 0.3)), max(2, int(h * 0.7))
            )

            for step in range(length):
                x = int(x0 + dx * step)
                y = int(y0 + dy * step)
                if 0 <= x < w and 0 <= y < h:
                    result[:, y, x] = result[:, y, x] * 0.3 + 0.7

        return result


# ===========================================================================
# E. Batch-level augmentation: CutMix, Mixup
#    批次级增强: CutMix, Mixup
# ===========================================================================


def _one_hot(
    labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Convert integer labels to one-hot float tensor.
    将整数标签转为 one-hot 浮点张量。
    """
    return torch.zeros(
        labels.size(0), num_classes,
        device=labels.device, dtype=torch.float32,
    ).scatter_(1, labels.unsqueeze(1).long(), 1.0)


def cutmix_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix: crop a random region and paste onto another image,
    mix labels proportionally to the cropped area.
    CutMix：裁剪随机区域粘贴到另一张图，按面积比例混合标签。

    Reference: Yun et al., "CutMix: Regularization Strategy..." (ICCV 2019)
    """
    if alpha <= 0:
        return images, _one_hot(labels, num_classes)

    lam = random.betavariate(alpha, alpha)
    lam = max(lam, 1 - lam)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    _, _, h, w = images.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = max(1, int(h * cut_ratio))
    cut_w = max(1, int(w * cut_ratio))

    cy = random.randint(0, h - 1)
    cx = random.randint(0, w - 1)

    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    actual_area = (y2 - y1) * (x2 - x1) / (h * w)
    lam_adjusted = 1.0 - actual_area

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[
        index, :, y1:y2, x1:x2
    ]

    labels_a = _one_hot(labels, num_classes)
    labels_b = _one_hot(labels[index], num_classes)
    mixed_labels = (
        labels_a * lam_adjusted + labels_b * (1 - lam_adjusted)
    )

    return mixed_images, mixed_labels


def mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mixup: linearly interpolate between two images and their labels.
    Mixup：两张图像及其标签的线性插值。

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization"
    (ICLR 2018)
    """
    if alpha <= 0:
        return images, _one_hot(labels, num_classes)

    lam = random.betavariate(alpha, alpha)
    lam = max(lam, 1 - lam)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = images * lam + images[index] * (1 - lam)

    labels_a = _one_hot(labels, num_classes)
    labels_b = _one_hot(labels[index], num_classes)
    mixed_labels = labels_a * lam + labels_b * (1 - lam)

    return mixed_images, mixed_labels


def apply_batch_augmentation(
    images: torch.Tensor,
    labels: torch.Tensor,
    aug_config: AugmentationConfig,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly apply CutMix / Mixup / identity per batch.
    每个批次随机选择 CutMix / Mixup / 不增强。

    Distribution within mix_prob: CutMix 4/7, Mixup 3/7.
    Residual (1 - mix_prob): identity.
    """
    if not aug_config.use_augmentation:
        return images, labels
    # 分类主开关：批次混合类别禁用时直接返回
    if not getattr(aug_config, "use_mixing_aug", True):
        return images, labels
    if not aug_config.use_cutmix and not aug_config.use_mixup:
        return images, labels

    r = random.random()

    cutmix_threshold = aug_config.mix_prob * (4 / 7)
    mixup_threshold = aug_config.mix_prob

    if aug_config.use_cutmix and r < cutmix_threshold:
        return cutmix_data(
            images, labels, aug_config.cutmix_alpha, num_classes
        )
    elif aug_config.use_mixup and r < mixup_threshold:
        return mixup_data(
            images, labels, aug_config.mixup_alpha, num_classes
        )
    else:
        return images, labels


# ===========================================================================
# Pipeline builder / 管线构建
# ===========================================================================


def build_train_transforms(
    aug_config: AugmentationConfig,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> transforms.Compose:
    """
    Build the full training transform pipeline from config.
    根据配置构建完整的训练变换管线。

    Pipeline order / 管线顺序:
      PIL 级:  Crop → Flip → Affine → Perspective → ColorJitter
               → Grayscale → AutoContrast → Equalize → Posterize
               → Solarize → JPEGCompression
      Tensor 级: ToTensor → Normalize → GaussianNoise → SaltPepper
               → GaussianBlur → FogEffect → RainStreaks → RandomErasing

    Args:
        aug_config: augmentation parameters
        mean:       归一化均值 / normalization mean
        std:        归一化标准差 / normalization std
    """
    if not aug_config.use_augmentation:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    pipeline: list[Callable] = []

    # ---- A. Geometric / 几何变换 (PIL 级) ----
    # getattr 保证缺少字段时默认 True（后向兼容）
    if getattr(aug_config, "use_geom_aug", True):
        pipeline.append(transforms.RandomCrop(
            32, padding=aug_config.random_crop_padding,
            padding_mode="reflect",
        ))
        pipeline.append(transforms.RandomHorizontalFlip(
            p=aug_config.hflip_prob,
        ))
        pipeline.append(transforms.RandomAffine(
            degrees=aug_config.affine_degrees,
            translate=(aug_config.affine_translate, aug_config.affine_translate),
            scale=aug_config.affine_scale,
            shear=aug_config.affine_shear,
        ))
        pipeline.append(transforms.RandomPerspective(
            distortion_scale=aug_config.perspective_distortion,
            p=aug_config.perspective_prob,
        ))

    # ---- B. Color / 颜色变换 (PIL 级) ----
    if getattr(aug_config, "use_color_aug", True):
        pipeline.append(transforms.ColorJitter(
            brightness=aug_config.cj_brightness,
            contrast=aug_config.cj_contrast,
            saturation=aug_config.cj_saturation,
            hue=aug_config.cj_hue,
        ))
        pipeline.append(transforms.RandomGrayscale(
            p=aug_config.grayscale_prob,
        ))
        pipeline.append(transforms.RandomAutocontrast(
            p=aug_config.auto_contrast_prob,
        ))
        pipeline.append(transforms.RandomEqualize(
            p=aug_config.equalize_prob,
        ))
        pipeline.append(transforms.RandomPosterize(
            bits=aug_config.posterize_bits,
            p=aug_config.posterize_prob,
        ))
        pipeline.append(transforms.RandomSolarize(
            threshold=aug_config.solarize_threshold,
            p=aug_config.solarize_prob,
        ))

    # ---- D. JPEG Compression / JPEG 压缩 (PIL 级) ----
    if getattr(aug_config, "use_weather_aug", True):
        pipeline.append(JPEGCompressionPIL(
            quality_range=aug_config.jpeg_quality,
            p=aug_config.jpeg_prob,
        ))

    # ---- ToTensor + Normalize / 转张量 + 归一化（始终包含）----
    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(mean=mean, std=std))

    # ---- C. Noise & Degradation / 噪声与降质 (Tensor 级) ----
    if getattr(aug_config, "use_noise_aug", True):
        pipeline.append(GaussianNoise(
            std=aug_config.gaussian_noise_std,
            p=aug_config.gaussian_noise_prob,
        ))
        pipeline.append(SaltPepperNoise(
            amount=aug_config.salt_pepper_amount,
            p=aug_config.salt_pepper_prob,
        ))
        pipeline.append(ProbabilisticGaussianBlur(
            kernel_size=aug_config.gaussian_blur_kernel,
            p=aug_config.gaussian_blur_prob,
        ))
        pipeline.append(transforms.RandomErasing(
            p=aug_config.erasing_prob,
            scale=aug_config.erasing_scale,
        ))

    # ---- D. Weather / 天气效果 (Tensor 级) ----
    if getattr(aug_config, "use_weather_aug", True):
        pipeline.append(FogEffect(
            intensity_range=aug_config.fog_intensity,
            p=aug_config.fog_prob,
        ))
        pipeline.append(RainStreaks(
            drops_range=aug_config.rain_drops,
            angle_range=aug_config.rain_angle,
            p=aug_config.rain_prob,
        ))

    return transforms.Compose(pipeline)


def build_test_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> transforms.Compose:
    """
    Build test transform pipeline (no augmentation, only normalize).
    构建测试变换管线（无增强，仅归一化）。
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
