"""
GPU training pipeline: optimization verification, profiling, and benchmarks.
GPU 训练流水线：优化验证、性能分析和基准测试。

Sections:
  TestOptimizationsActive  — 静态检查优化已生效（快速，无需 GPU）
  TestGPUThroughput         — 合成数据吞吐量对比（需 GPU）
  TestPipelineBreakdown     — 逐阶段计时定位瓶颈（需 GPU）
"""

import dataclasses
import inspect
import time

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.Q3.config import TrainConfig
from src.Q3.data import get_cifar100_loaders
from src.Q2.model import create_model
from src.Q3.train import train_one_epoch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_synthetic(batch_size: int = 256, n: int = 2048):
    """Create synthetic CIFAR-shaped DataLoaders."""
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 100, (n,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, pin_memory=True)


def _train_naive_one_epoch(model, loader, optimizer, criterion, device):
    """Un-optimized baseline: .item() on every batch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


def _gpu_only(cls):
    """Skip entire class when no CUDA."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA"
    )(cls)


# ===========================================================================
# 1. Static checks — fast, no GPU needed
# ===========================================================================


class TestOptimizationsActive:
    """Verify optimizations are present in source code."""

    def test_non_blocking(self):
        src = inspect.getsource(train_one_epoch)
        assert "non_blocking=True" in src

    def test_set_to_none(self):
        src = inspect.getsource(train_one_epoch)
        assert "set_to_none=True" in src

    def test_amp_autocast_in_train(self):
        src = inspect.getsource(train_one_epoch)
        assert "autocast" in src
        assert "scaler.scale(" in src

    def test_amp_autocast_in_evaluate(self):
        from utils.evaluate import evaluate
        src = inspect.getsource(evaluate)
        assert "autocast" in src

    def test_evaluate_accumulates_on_gpu(self):
        """evaluate should not call .item() inside the loop."""
        from utils.evaluate import evaluate
        src = inspect.getsource(evaluate)
        in_loop = False
        for line in src.splitlines():
            if "for " in line and "loader" in line:
                in_loop = True
                continue
            if in_loop and line.strip() and not line.startswith(" " * 8):
                in_loop = False
            if in_loop and ".item()" in line:
                pytest.fail("evaluate() calls .item() inside the loop")


# ===========================================================================
# 2. GPU benchmarks — synthetic data
# ===========================================================================


@_gpu_only
class TestGPUThroughput:
    """Throughput comparison: optimized vs naive pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda")
        self.model = create_model(100).to(self.device)
        self.optimizer = SGD(
            self.model.parameters(), lr=0.01, momentum=0.9
        )
        self.criterion = nn.CrossEntropyLoss()
        self.loader = _make_synthetic(batch_size=256, n=2048)
        torch.backends.cudnn.benchmark = True
        torch.cuda.synchronize()

    def _measure(self, fn, epochs=5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(epochs):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n = len(self.loader.dataset) * epochs
        return n / elapsed, elapsed

    def test_optimized_vs_naive(self):
        # Naive / 朴素
        naive_tput, naive_t = self._measure(
            lambda: _train_naive_one_epoch(
                self.model, self.loader,
                self.optimizer, self.criterion, self.device,
            )
        )
        # Optimized + AMP / 优化 + AMP
        scaler = torch.amp.GradScaler("cuda")

        def opt_fn():
            train_one_epoch(
                self.model, self.loader, self.optimizer,
                self.criterion, self.device, epoch=1, scaler=scaler,
            )

        opt_tput, opt_t = self._measure(opt_fn)
        speedup = opt_tput / naive_tput
        print(
            f"\n  Naive FP32:     {naive_tput:.0f} s/s ({naive_t:.2f}s)"
        )
        print(
            f"  Optimized+AMP:  {opt_tput:.0f} s/s ({opt_t:.2f}s)"
        )
        print(f"  Speedup:        {speedup:.2f}x")
        assert speedup >= 1.5

    def test_absolute_throughput(self):
        scaler = torch.amp.GradScaler("cuda")
        tput, _ = self._measure(
            lambda: train_one_epoch(
                self.model, self.loader, self.optimizer,
                self.criterion, self.device, epoch=1, scaler=scaler,
            ),
            epochs=3,
        )
        print(f"\n  Throughput: {tput:.0f} samples/sec")
        assert tput > 1500


# ===========================================================================
# 3. Pipeline breakdown — per-phase timing with real data
# ===========================================================================


@_gpu_only
class TestPipelineBreakdown:
    """
    Per-phase timing and bottleneck identification.
    Uses real CIFAR-100 to capture actual data loading cost.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda")
        self.config = TrainConfig(num_workers=0, batch_size=256)
        self.train_loader, _ = get_cifar100_loaders(self.config)
        self.model = create_model(100).to(self.device)
        self.optimizer = SGD(
            self.model.parameters(), lr=0.01, momentum=0.9
        )
        self.criterion = nn.CrossEntropyLoss()
        torch.backends.cudnn.benchmark = True
        self.model.train()
        torch.cuda.synchronize()

    def test_phase_timing(self):
        """
        Break down: data_load → h2d → forward → backward → opt_step.
        逐阶段分解计时。
        """
        device = self.device
        loader = self.train_loader

        # Warmup
        for i, (img, lbl) in enumerate(loader):
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.criterion(self.model(img), lbl)
            loss.backward()
            self.optimizer.step()
            if i >= 2:
                break
        torch.cuda.synchronize()

        # Measure
        phases = {
            "data_load": 0.0,
            "h2d": 0.0,
            "forward": 0.0,
            "backward": 0.0,
            "opt_step": 0.0,
        }
        prev_end = time.perf_counter()

        for images, labels in loader:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            loss.backward()
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            self.optimizer.step()
            torch.cuda.synchronize()
            t4 = time.perf_counter()

            phases["data_load"] += t0 - prev_end
            phases["h2d"] += t1 - t0
            phases["forward"] += t2 - t1
            phases["backward"] += t3 - t2
            phases["opt_step"] += t4 - t3
            prev_end = t4

        total = sum(phases.values())
        bottleneck = max(phases, key=phases.get)

        print(f"\n  === Phase timing ({len(loader)} batches) ===")
        for name, t in phases.items():
            pct = t / total * 100
            print(f"  {name:12s}: {t:.3f}s ({pct:.1f}%)")
        print(f"  {'total':12s}: {total:.3f}s")
        print(f"  Bottleneck: {bottleneck}")
        print(
            f"  Throughput: "
            f"{len(loader.dataset) / total:.0f} samples/sec"
        )

    def test_num_workers_effect(self):
        """num_workers=0 vs 2 vs 4 with real data."""
        for nw in [0, 2, 4]:
            cfg = dataclasses.replace(self.config, num_workers=nw)
            loader, _ = get_cifar100_loaders(cfg)
            model = create_model(100).to(self.device)
            opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
            crit = nn.CrossEntropyLoss()
            model.train()

            # Warmup 1 batch
            for img, lbl in loader:
                img = img.to(self.device, non_blocking=True)
                lbl = lbl.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                crit(model(img), lbl).backward()
                opt.step()
                break
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_one_epoch(model, loader, opt, crit, self.device, 1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            tput = len(loader.dataset) / elapsed
            print(
                f"  num_workers={nw}: "
                f"{tput:.0f} s/s ({elapsed:.2f}s)"
            )
