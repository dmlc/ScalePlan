"""Benchmark the per-kernel CPU DISPATCH GAP for MoE-training idle modeling.

Motivation
----------
The analytical model over-predicts MFU on small-tile / low-MFU benchmark configs
because it does not model per-step IDLE: the GPU stalling because the CPU (eager
PyTorch dispatch + autograd + routing host ops) cannot enqueue the next kernel
before the current one finishes. The 18B CPU-timing analysis
(profile_parse/trace_summaries/18B_CPU_ANALYSIS_SUMMARY.md) shows idle is host-side
work (aten::copy_, masked_select/nonzero routing, per-op launch) — NOT comm launch
(<350ms). Backing the measured idle out of a per-kernel model gave a ~560us median
gap with 6x spread; this script MEASURES that gap directly so the model uses a
calibrated number, not a back-out.

Physics being measured
-----------------------
For a chain of independent eager kernels, steady-state throughput is bounded by the
SLOWER of GPU execution and CPU enqueue:

    wall_per_kernel(M) = max( gpu_kernel_time(M), cpu_dispatch_gap )

Sweeping the GEMM size M traces this curve:
  * small M  -> gpu_kernel_time ~ 0, so wall_per_kernel PLATEAUS at cpu_dispatch_gap
               (the number the idle model needs).
  * large M  -> GPU-bound, wall_per_kernel = gpu_kernel_time (crosschecks gemm_util).
The crossover M (where the two meet) is where dispatch stops being hidden by
compute — i.e. exactly the boundary below which the model must add idle. This
validates the max(compute, gap) form AND calibrates the gap in one sweep.

Modes (--mode):
  * gemm      : chain of eager torch.matmul (the dominant kernel). Default.
  * gemm_relu : matmul + relu per step (2 kernels) — more host ops per GPU op,
                closer to a real transformer block's launch density.

Usage (inside the training image on a B200 sleeper pod):
    python dispatch_gap_benchmark.py --device b200 --output dispatch_gap_b200.parquet
    python dispatch_gap_benchmark.py --only-target   # fast: representative shape only

Emits parquet+csv keyed by (device, gpu_name, mode, M, N, K, dtype) with
wall_us_per_kernel, gpu_active_us_per_kernel, idle_us_per_kernel, and the derived
dispatch_gap_us (= wall - gpu_active, the stall the model must add).
"""

from __future__ import annotations

import argparse
import itertools
import platform
import sys
from dataclasses import asdict, dataclass

import pandas as pd
import torch

# GEMM M (row count) sweep — from launch-bound (tiny) to compute-bound (large).
DEFAULT_M = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# (N, K) = (out, in) per kernel. 2048 ~ a representative hidden; the gap is
# ~size-independent (it's CPU-side) so one shape suffices, but allow a couple.
DEFAULT_SHAPES = [(2048, 2048)]
DEFAULT_DTYPES = ["bf16"]
DEFAULT_MODES = ["gemm"]

# Representative point for --only-target: a small expert-tile GEMM (M ~ dropless cap).
TARGET_POINT = dict(mode="gemm", M=192, N=2048, K=2048, dtype="bf16")

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DEVICE_AUTODETECT = [("B200", "b200"), ("H200", "h200"), ("H100", "h100"), ("A100", "a100")]


@dataclass
class BenchRow:
    device: str
    gpu_name: str
    mode: str
    M: int
    N: int
    K: int
    dtype: str
    chain_len: int
    wall_us_per_kernel: float
    gpu_active_us_per_kernel: float
    idle_us_per_kernel: float  # wall - gpu_active == the stall the model must add
    gpu_frac: float  # gpu_active / wall  (1.0 => compute-bound, ~0 => dispatch-bound)
    torch_version: str
    status: str


def autodetect_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; cannot benchmark dispatch gap on GPU")
    name = torch.cuda.get_device_name(0)
    for needle, label in DEVICE_AUTODETECT:
        if needle in name:
            return label
    raise RuntimeError(f"Could not autodetect device label from '{name}'. Pass --device.")


def _step_fn(mode: str):
    """One eager 'step' = the op(s) whose launch cadence we're measuring.

    Returns a callable(x, w) -> tensor. Kept as plain eager ops (no CUDA graph, no
    compile) so each call pays the real per-kernel CPU dispatch cost that eager
    training pays.
    """
    if mode == "gemm":
        return lambda x, w: torch.matmul(x, w)
    if mode == "gemm_relu":
        return lambda x, w: torch.relu(torch.matmul(x, w))
    raise ValueError(f"unknown mode {mode}")


def bench_chain_wall(step, x, w, chain_len: int, n_warmup: int, n_iter: int) -> float:
    """Wall time per kernel over a chain of `chain_len` eager ops (CUDA events).

    No synchronization inside the chain: the CPU enqueues as fast as it can and the
    GPU drains the queue, so steady-state wall/chain_len == max(gpu_time, cpu_gap).
    A large chain_len makes it steady-state (amortizes queue fill/drain at the ends).
    """
    for _ in range(n_warmup):
        for _ in range(chain_len):
            step(x, w)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        for _ in range(chain_len):
            step(x, w)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return (total_ms * 1000.0) / (n_iter * chain_len)  # us per kernel


def bench_chain_gpu_active(step, x, w, chain_len: int, n_warmup: int) -> float:
    """GPU-active time per kernel via profiler (sum of CUDA self-time / n_kernels).

    Isolates GPU execution from the dispatch gap: idle = wall - gpu_active.
    """
    for _ in range(n_warmup):
        for _ in range(chain_len):
            step(x, w)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        for _ in range(chain_len):
            step(x, w)
        torch.cuda.synchronize()

    total_us = 0.0
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            total_us += getattr(
                evt, "self_device_time_total", getattr(evt, "self_cuda_time_total", 0)
            )
    return total_us / chain_len


def benchmark_point(
    device_label, gpu_name, mode, M, N, K, dtype, chain_len, n_warmup, n_iter, torch_version
):
    torch_dtype = DTYPE_MAP[dtype]
    step = _step_fn(mode)
    try:
        x = torch.randn(M, K, dtype=torch_dtype, device="cuda")
        w = torch.randn(K, N, dtype=torch_dtype, device="cuda")
        wall = bench_chain_wall(step, x, w, chain_len, n_warmup, n_iter)
        gpu = bench_chain_gpu_active(step, x, w, chain_len, n_warmup)
        idle = max(wall - gpu, 0.0)
        status = "OK"
    except Exception as e:  # noqa: BLE001
        return BenchRow(device_label, gpu_name, mode, M, N, K, dtype, chain_len,
                        float("nan"), float("nan"), float("nan"), 0.0, torch_version,
                        f"FAILED: {type(e).__name__}: {e}")
    return BenchRow(
        device=device_label, gpu_name=gpu_name, mode=mode, M=M, N=N, K=K, dtype=dtype,
        chain_len=chain_len, wall_us_per_kernel=wall, gpu_active_us_per_kernel=gpu,
        idle_us_per_kernel=idle, gpu_frac=(gpu / wall if wall > 0 else 0.0),
        torch_version=torch_version, status=status,
    )


def build_grid(args) -> list[dict]:
    if args.only_target:
        return [dict(TARGET_POINT)]
    return [
        dict(mode=mode, M=m, N=n, K=k, dtype=dt)
        for mode, m, (n, k), dt in itertools.product(
            args.modes, args.ms, DEFAULT_SHAPES, args.dtypes
        )
    ]


def summarize(df: pd.DataFrame) -> None:
    ok = df[df["status"] == "OK"]
    print("\n=== Summary ===")
    print(f"Device: {df['device'].iloc[0]} ({df['gpu_name'].iloc[0]})  torch={df['torch_version'].iloc[0]}")
    print(f"Rows: {len(df)} ({len(ok)} OK)")
    if ok.empty:
        return
    for mode in sorted(ok["mode"].unique()):
        sub = ok[ok["mode"] == mode].sort_values("M")
        print(f"\nmode={mode}  (N,K from {sorted(set(zip(sub.N, sub.K)))}):")
        print(f"  {'M':>7}{'wall_us':>10}{'gpu_us':>10}{'idle_us':>10}{'gpu_frac':>10}")
        for _, r in sub.iterrows():
            print(f"  {int(r.M):>7}{r.wall_us_per_kernel:>10.2f}{r.gpu_active_us_per_kernel:>10.2f}"
                  f"{r.idle_us_per_kernel:>10.2f}{r.gpu_frac:>10.2f}")
        # The dispatch gap = wall_per_kernel in the launch-bound (small-M) plateau:
        # the median wall over the M's where gpu_frac < 0.25 (GPU clearly not the bottleneck).
        lb = sub[sub["gpu_frac"] < 0.25]
        if len(lb):
            gap = lb["wall_us_per_kernel"].median()
            print(f"  -> DISPATCH GAP (median wall where gpu_frac<0.25): {gap:.2f} us/kernel")
        # Crossover: smallest M where GPU becomes the bottleneck (gpu_frac >= 0.5).
        cb = sub[sub["gpu_frac"] >= 0.5]
        if len(cb):
            print(f"  -> crossover: GPU-bound (gpu_frac>=0.5) at M>={int(cb.M.min())} "
                  f"(below this, model must add idle)")


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark per-kernel CPU dispatch gap (MoE idle model)")
    p.add_argument("--device", choices=["a100", "h100", "h200", "b200"], default=None)
    p.add_argument("--output", type=str, default="dispatch_gap_timings.parquet")
    p.add_argument("--ms", type=int, nargs="+", default=DEFAULT_M)
    p.add_argument("--dtypes", type=str, nargs="+", default=DEFAULT_DTYPES)
    p.add_argument("--modes", type=str, nargs="+", default=DEFAULT_MODES,
                   choices=["gemm", "gemm_relu"])
    p.add_argument("--chain-len", type=int, default=512,
                   help="kernels per timed chain (steady-state; large amortizes queue ends)")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--only-target", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available on this host.", file=sys.stderr)
        return 2

    device_label = args.device or autodetect_device()
    gpu_name = torch.cuda.get_device_name(0)
    tv = torch.__version__
    print(f"GPU: {gpu_name}\nDevice label: {device_label}\nTorch: {tv}, CUDA: {torch.version.cuda}")
    print(f"Host: {platform.node()}")

    grid = build_grid(args)
    print(f"Grid size: {len(grid)} (chain_len={args.chain_len})")

    rows = []
    for i, cfg in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] mode={cfg['mode']} M={cfg['M']:>6} N={cfg['N']} K={cfg['K']} "
              f"{cfg['dtype']}", end=" ... ", flush=True)
        row = benchmark_point(
            device_label=device_label, gpu_name=gpu_name, chain_len=args.chain_len,
            n_warmup=args.n_warmup, n_iter=args.n_iter, torch_version=tv, **cfg,
        )
        rows.append(row)
        if row.status == "OK":
            print(f"wall={row.wall_us_per_kernel:.2f}us gpu={row.gpu_active_us_per_kernel:.2f}us "
                  f"idle={row.idle_us_per_kernel:.2f}us frac={row.gpu_frac:.2f}")
        else:
            print(row.status)

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_parquet(args.output, index=False)
    df.to_csv(args.output.rsplit(".", 1)[0] + ".csv", index=False)
    print(f"\nSaved: {args.output}")
    summarize(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
