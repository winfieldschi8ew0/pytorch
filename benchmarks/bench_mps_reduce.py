#!/usr/bin/env python3
"""Benchmark Metal reduce ops vs MPSGraph baseline.

Measures: prod, var, std, var_mean, std_mean, argmax, argmin, max(dim), min(dim)
Tests fixed-shape throughput, variable-shape cache swell, and large reductions.
"""

import torch
import time

WARMUP = 20
ITERS = 200


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds


def fixed_shape_benchmark():
    print("\n=== Fixed-shape throughput (us/call) ===")
    configs = [
        ("prod dim=0", (128, 256), torch.float32, lambda x: torch.prod(x, dim=0)),
        ("prod dim=1", (128, 256), torch.float32, lambda x: torch.prod(x, dim=1)),
        ("prod scalar", (64, 64), torch.float32, lambda x: torch.prod(x)),
        ("prod int64 dim=0", (64, 128), torch.int64, lambda x: torch.prod(x, dim=0)),
        ("var dim=1", (128, 256), torch.float32, lambda x: torch.var(x, dim=1)),
        ("var dim=0", (128, 256), torch.float32, lambda x: torch.var(x, dim=0)),
        ("std dim=1", (128, 256), torch.float32, lambda x: torch.std(x, dim=1)),
        (
            "var_mean dim=1",
            (128, 256),
            torch.float32,
            lambda x: torch.var_mean(x, dim=1),
        ),
        (
            "std_mean dim=1",
            (128, 256),
            torch.float32,
            lambda x: torch.std_mean(x, dim=1),
        ),
        ("var scalar", (64, 64), torch.float32, lambda x: torch.var(x)),
        ("argmax dim=0", (128, 256), torch.float32, lambda x: torch.argmax(x, dim=0)),
        ("argmax dim=1", (128, 256), torch.float32, lambda x: torch.argmax(x, dim=1)),
        ("argmin dim=1", (128, 256), torch.float32, lambda x: torch.argmin(x, dim=1)),
        ("max dim=0", (128, 256), torch.float32, lambda x: torch.max(x, dim=0)),
        ("max dim=1", (128, 256), torch.float32, lambda x: torch.max(x, dim=1)),
        ("min dim=0", (128, 256), torch.float32, lambda x: torch.min(x, dim=0)),
        (
            "argmax fp16 dim=1",
            (256, 512),
            torch.float16,
            lambda x: torch.argmax(x, dim=1),
        ),
        ("var bf16 dim=1", (128, 256), torch.bfloat16, lambda x: torch.var(x, dim=1)),
    ]

    print(f"{'Op':<25} {'Shape':<15} {'dtype':<10} {'Time (us)':<12}")
    print("-" * 65)
    for name, shape, dtype, fn in configs:
        if dtype == torch.int64:
            x = torch.randint(1, 10, shape, device="mps", dtype=dtype)
        else:
            x = torch.randn(shape, device="mps", dtype=dtype)
        t = bench(lambda: fn(x))
        print(f"{name:<25} {str(shape):<15} {str(dtype):<10} {t:>10.1f}")


def variable_shape_benchmark():
    print("\n=== Variable-shape cache swell test (us/call, 20 shapes) ===")
    shapes = [(n, 64) for n in range(10, 210, 10)]

    ops = [
        ("prod dim=0", lambda x: torch.prod(x, dim=0)),
        ("var dim=0", lambda x: torch.var(x, dim=0)),
        ("argmax dim=0", lambda x: torch.argmax(x, dim=0)),
        ("max dim=0", lambda x: torch.max(x, dim=0)),
    ]

    for op_name, op_fn in ops:
        tensors = [torch.randn(s, device="mps") for s in shapes]
        # Warmup
        for _ in range(5):
            for t in tensors:
                op_fn(t)
        torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(50):
            for t in tensors:
                op_fn(t)
        torch.mps.synchronize()
        elapsed = (time.perf_counter() - t0) / (50 * len(shapes)) * 1e6
        print(
            f"  {op_name:<25} avg {elapsed:>8.1f} us/call across {len(shapes)} shapes"
        )


def large_reduction_benchmark():
    print("\n=== Large reduction benchmark (us/call) ===")
    configs = [
        (
            "prod dim=1 1Kx4K",
            (1024, 4096),
            torch.float32,
            lambda x: torch.prod(x, dim=1),
        ),
        ("var dim=1 1Kx4K", (1024, 4096), torch.float32, lambda x: torch.var(x, dim=1)),
        (
            "argmax dim=1 1Kx4K",
            (1024, 4096),
            torch.float32,
            lambda x: torch.argmax(x, dim=1),
        ),
        ("max dim=1 1Kx4K", (1024, 4096), torch.float32, lambda x: torch.max(x, dim=1)),
        (
            "prod dim=0 4Kx1K",
            (4096, 1024),
            torch.float32,
            lambda x: torch.prod(x, dim=0),
        ),
        ("var dim=0 4Kx1K", (4096, 1024), torch.float32, lambda x: torch.var(x, dim=0)),
    ]

    print(f"{'Op':<30} {'Time (us)':<12}")
    print("-" * 45)
    for name, shape, dtype, fn in configs:
        x = torch.randn(shape, device="mps", dtype=dtype)
        t = bench(lambda: fn(x), warmup=10, iters=100)
        print(f"{name:<30} {t:>10.1f}")


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print(f"MPS device: {torch.backends.mps.is_available()}")
    fixed_shape_benchmark()
    variable_shape_benchmark()
    large_reduction_benchmark()
