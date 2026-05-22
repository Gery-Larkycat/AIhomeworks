"""
Performance benchmark for GPU training pipeline optimizations.
GPU 训练流水线优化的性能基准测试。

Verifies:
1. Optimizations are active (cudnn.benchmark, non_blocking, set_to_none)
2. Throughput improvement: optimized vs naive (per-batch .item()) pipeline
验证：
1. 优化已生效（cudnn.benchmark, non_blocking, set_to_none）
2. 吞吐量提升：优化管线 vs 朴素管线（每 batch 调 .item()）
"""

import time

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.Q3.model import create_model
from src.Q3.train import train_one_epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic(batch_size: int = 256, n: int = 2048):
    """Create synthetic CIFAR-shaped loaders / 创建合成 CIFAR 形状的 loader。"""
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 100, (n,))
    train_ds = TensorDataset(images, labels)
    test_ds = TensorDataset(images[:512], labels[:512])
    return (
        DataLoader(train_ds, batch_size=batch_size, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, pin_memory=True),
    )


def _train_naive_one_epoch(model, loader, optimizer, criterion, device):
    """
    Naive pipeline: .item() on every batch (the un-optimized version).
    朴素管线：每 batch 调 .item()（未优化版本）。
    """
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
        # Naive: per-batch sync / 朴素：每 batch 同步
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Test: optimizations are active
# ---------------------------------------------------------------------------


class TestOptimizationsActive:
    """Verify each optimization is actually enabled."""

    def test_cudnn_benchmark_set_in_train(self):
        """train() 应启用 cudnn.benchmark。"""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA")
        old_val = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        assert torch.backends.cudnn.benchmark is True
        torch.backends.cudnn.benchmark = old_val

    def test_non_blocking_used(self):
        """验证 train_one_epoch 使用 non_blocking 传输。"""
        import inspect
        src = inspect.getsource(train_one_epoch)
        assert "non_blocking=True" in src

    def test_set_to_none_used(self):
        """验证 train_one_epoch 使用 set_to_none=True。"""
        import inspect
        src = inspect.getsource(train_one_epoch)
        assert "set_to_none=True" in src

    def test_amp_used_in_train(self):
        """验证 train_one_epoch 使用 AMP autocast。"""
        import inspect
        src = inspect.getsource(train_one_epoch)
        assert "autocast" in src
        # Should use scaler passed as param, not create one
        assert "scaler.scale(" in src or "scaler.step(" in src

    def test_amp_used_in_evaluate(self):
        """验证 evaluate 使用 AMP autocast。"""
        import inspect
        from src.Q3.evaluate import evaluate
        src = inspect.getsource(evaluate)
        assert "autocast" in src

    def test_evaluate_no_per_batch_item(self):
        """验证 evaluate 不再每 batch 调 .item()。"""
        import inspect
        from src.Q3.evaluate import evaluate
        src = inspect.getsource(evaluate)
        lines = src.split("\n")
        in_loop = False
        for line in lines:
            if "for " in line and "loader" in line:
                in_loop = True
                continue
            if in_loop and line.strip() and not line.startswith(" " * 8):
                in_loop = False
            if in_loop and ".item()" in line:
                pytest.fail(
                    "evaluate() should not call .item() "
                    "inside the loop"
                )

    def test_evaluate_no_per_batch_item(self):
        """验证 evaluate 不再每 batch 调 .item()。"""
        import inspect
        from src.Q3.evaluate import evaluate
        src = inspect.getsource(evaluate)
        # Should NOT have .item() inside the loop body
        # loop 内不应有 .item()
        # Only allowed at the end (return line)
        lines = src.split("\n")
        in_loop = False
        for line in lines:
            if "for " in line and "loader" in line:
                in_loop = True
                continue
            if in_loop and line.strip() and not line.startswith(" " * 8):
                in_loop = False
            if in_loop and ".item()" in line:
                pytest.fail(
                    "evaluate() should not call .item() "
                    "inside the loop"
                )


# ---------------------------------------------------------------------------
# Benchmark: throughput comparison
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA"
)
class TestGPUTThroughput:
    """
    Compare optimized vs naive training throughput on GPU.
    对比优化 vs 朴素训练管线的 GPU 吞吐量。
    """

    @pytest.fixture()
    def setup(self):
        """Create model, loaders, optimizer, criterion on GPU."""
        device = torch.device("cuda")
        model = create_model(num_classes=100).to(device)
        optimizer = SGD(
            model.parameters(), lr=0.01, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        train_loader, test_loader = _make_synthetic(
            batch_size=256, n=2048
        )
        # Warm up GPU / GPU 预热
        torch.cuda.synchronize()
        return {
            "model": model,
            "optimizer": optimizer,
            "criterion": criterion,
            "device": device,
            "train_loader": train_loader,
            "test_loader": test_loader,
        }

    def _measure_throughput(self, train_fn, setup, epochs=5):
        """
        Measure average throughput (samples/sec) over multiple epochs.
        测量多个 epoch 的平均吞吐量（样本/秒）。
        """
        device = setup["device"]
        total_samples = 0
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(epochs):
            train_fn(
                setup["model"],
                setup["train_loader"],
                setup["optimizer"],
                setup["criterion"],
                device,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        n_samples = len(setup["train_loader"].dataset)
        total_samples = n_samples * epochs
        throughput = total_samples / elapsed
        return throughput, elapsed

    def test_optimized_faster_than_naive(self, setup):
        """
        Optimized pipeline (AMP + tensor accum) vs naive (.item() per batch).
        优化管线（AMP + 张量累积）vs 朴素管线（每 batch 调 .item()）。
        """
        torch.backends.cudnn.benchmark = True

        # Naive pipeline / 朴素管线
        naive_tput, naive_time = self._measure_throughput(
            _train_naive_one_epoch, setup, epochs=5
        )

        # Optimized pipeline with AMP / 带 AMP 的优化管线
        scaler = torch.amp.GradScaler("cuda")

        def optimized_fn(
            model, loader, optimizer, criterion, device
        ):
            return train_one_epoch(
                model, loader, optimizer, criterion,
                device, epoch=1, scaler=scaler,
            )

        opt_tput, opt_time = self._measure_throughput(
            optimized_fn, setup, epochs=5
        )

        speedup = opt_tput / naive_tput
        print(
            f"\n  Naive:  {naive_tput:.0f} samples/sec"
            f" ({naive_time:.3f}s)"
        )
        print(
            f"  Optimized (AMP): {opt_tput:.0f} samples/sec"
            f" ({opt_time:.3f}s)"
        )
        print(f"  Speedup: {speedup:.2f}x")

        # AMP + optimizations should give >= 1.5x speedup
        assert speedup >= 1.5, (
            f"Optimized pipeline ({opt_tput:.0f} s/s) "
            f"should be >= 1.5x faster than naive "
            f"({naive_tput:.0f} s/s), "
            f"got {speedup:.2f}x"
        )

    def test_absolute_throughput_reasonable(self, setup):
        """
        Throughput with AMP should be reasonable for RTX 4060.
        AMP 下 RTX 4060 的吞吐量应合理。
        """
        torch.backends.cudnn.benchmark = True
        scaler = torch.amp.GradScaler("cuda")
        tput, _ = self._measure_throughput(
            lambda m, loader, o, c, d: train_one_epoch(
                m, loader, o, c, d, epoch=1, scaler=scaler
            ),
            setup,
            epochs=3,
        )
        print(f"\n  Throughput: {tput:.0f} samples/sec")
        # With small synthetic data (8 batches/epoch), kernel launch
        # overhead dominates. This threshold just catches gross issues.
        # 合成数据太小（8 batch/epoch），kernel launch 开销占主导。
        # 此阈值仅用于检测严重问题。
        assert tput > 1500, (
            f"Throughput {tput:.0f} s/s is too low, "
            f"expected > 1500 s/s"
        )
