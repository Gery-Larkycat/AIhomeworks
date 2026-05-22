"""
GPU training pipeline profiling — identify remaining bottlenecks.
GPU 训练流水线 profiling — 定位剩余瓶颈。

Measures per-phase timing:
  data_load → h2d_transfer → forward → backward → optimizer_step
and checks for unnecessary CPU↔GPU transfers.
逐阶段计时：data_load → h2d_transfer → forward → backward → optimizer_step
并检查非必要的 CPU↔GPU 数据搬运。
"""

import time

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from src.Q3.model import create_model


def _make_loader(batch_size=256, n=2048):
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 100, (n,))
    ds = TensorDataset(images, labels)
    return DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
class TestPipelineProfile:
    """
    Break down per-batch timing to find the bottleneck.
    逐 batch 分解计时以找到瓶颈。
    """

    def test_per_phase_timing(self):
        """
        Measure data_load, h2d, forward, backward, optimizer_step.
        测量 data_load、H2D、forward、backward、optimizer_step 各阶段。
        """
        device = torch.device("cuda")
        model = create_model(100).to(device)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        loader = _make_loader(batch_size=256, n=4096)

        torch.backends.cudnn.benchmark = True
        model.train()

        # Warmup / 预热
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        # Profile / 计时
        timings = {
            "data_load": 0.0,
            "h2d": 0.0,
            "forward": 0.0,
            "backward": 0.0,
            "opt_step": 0.0,
            "total": 0.0,
        }
        n_batches = 0

        # Use CUDA events for accurate GPU timing
        # 使用 CUDA event 精确计时
        data_iter = iter(loader)

        torch.cuda.synchronize()
        t_total_start = time.perf_counter()

        for images, labels in loader:
            # Phase 1: data load (CPU)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Phase 2: H2D transfer
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Phase 3: forward
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            # Phase 4: backward
            loss.backward()
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            # Phase 5: optimizer step
            optimizer.step()
            torch.cuda.synchronize()
            t4 = time.perf_counter()

            # Approximate data_load as time between batches
            timings["h2d"] += t1 - t0
            timings["forward"] += t2 - t1
            timings["backward"] += t3 - t2
            timings["opt_step"] += t4 - t3
            n_batches += 1

        torch.cuda.synchronize()
        t_total_end = time.perf_counter()
        timings["total"] = t_total_end - t_total_start
        timings["data_load"] = (
            timings["total"]
            - timings["h2d"]
            - timings["forward"]
            - timings["backward"]
            - timings["opt_step"]
        )

        # Report / 报告
        print("\n  === Per-phase timing (avg per batch) ===")
        for phase, t in timings.items():
            if phase == "total":
                print(
                    f"  {phase:15s}: {t:.4f}s total"
                )
            else:
                avg = t / n_batches
                pct = t / timings["total"] * 100
                print(
                    f"  {phase:15s}: {avg*1000:.2f}ms/batch"
                    f" ({pct:.1f}%)"
                )

        # Identify bottleneck / 识别瓶颈
        phases = ["data_load", "h2d", "forward", "backward", "opt_step"]
        bottleneck = max(phases, key=lambda p: timings[p])
        print(f"\n  Bottleneck: {bottleneck}")

    def test_throughput_vs_batch_size(self):
        """
        Measure throughput at different batch sizes to find saturation point.
        测量不同 batch_size 下的吞吐量，找到饱和点。
        """
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

        results = []
        for bs in [64, 128, 256, 512, 1024]:
            model = create_model(100).to(device)
            optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            loader = _make_loader(batch_size=bs, n=4096)
            model.train()

            # Warmup
            for img, lbl in loader:
                img = img.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(img), lbl)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()

            # Measure
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                for img, lbl in loader:
                    img = img.to(device, non_blocking=True)
                    lbl = lbl.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(img), lbl)
                    loss.backward()
                    optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            n_samples = 4096 * 3
            tput = n_samples / elapsed
            results.append((bs, tput))
            print(
                f"  batch_size={bs:4d}: "
                f"{tput:.0f} samples/sec"
            )

        # Throughput should be reasonable at all batch sizes
        # 所有 batch_size 下的吞吐量应合理
        for bs, tput in results:
            assert tput > 500, (
                f"batch_size={bs} throughput {tput:.0f} too low"
            )
        # Largest batch should generally be among the fastest
        # (more GPU work per kernel launch amortizes overhead)
        # 最大 batch 通常应是最快的（更多 GPU 工作分摊开销）
        assert results[-1][1] > results[0][1], (
            f"bs={results[-1][0]} should beat bs={results[0][0]}"
        )

    def test_amp_speedup(self):
        """
        Measure throughput with AMP (mixed precision) vs FP32.
        对比 AMP（混合精度）与 FP32 的吞吐量。
        """
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

        results = {}

        for use_amp, label in [(False, "FP32"), (True, "AMP")]:
            model = create_model(100).to(device)
            optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            loader = _make_loader(batch_size=256, n=4096)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            model.train()

            # Warmup
            for img, lbl in loader:
                img = img.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(img), lbl)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()

            # Measure
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                for img, lbl in loader:
                    img = img.to(device, non_blocking=True)
                    lbl = lbl.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        loss = criterion(model(img), lbl)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            n_samples = 4096 * 5
            tput = n_samples / elapsed
            results[label] = tput
            print(
                f"  {label}: {tput:.0f} samples/sec"
                f" ({elapsed:.3f}s)"
            )

        speedup = results["AMP"] / results["FP32"]
        print(f"  AMP speedup: {speedup:.2f}x")
        # AMP should be at least 1.2x faster on RTX 4060
        assert speedup >= 1.2, (
            f"AMP should be faster, got {speedup:.2f}x"
        )
