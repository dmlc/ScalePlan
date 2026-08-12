"""Benchmark torch.topk (PyTorch ATen sbtopk::gatherTopK) for MoE routing.

Motivation:
    dlcalc/utils/moe_router_util.py hardcodes a per-GPU TOPK_THROUGHPUT for the
    FORWARD, and dlcalc/utils/backward.py:compute_topk_bwd_time_s hardcodes a
    SEPARATE, uncalibrated TOPK_THROUGHPUT for the BACKWARD that is ~120x too slow
    (1.5e8 vs the forward's measured 1.8e10 elem/s on B200), producing a ~7ms
    phantom TopK-backward per MoE block (Routing predicted ~30x the measured
    bucket; see perf_scratch/COMPONENT_COMPARE.md). This script measures BOTH the
    forward and the autograd backward across shapes/dtypes/GPUs so both constants
    can be recalibrated from data, not guesses.

Kernels of interest:
    Forward:  at::native::sbtopk::gatherTopK<...> - PyTorch's small-bucket TopK
              kernel (aten/.../TensorTopK.cu). Dispatched when the reduction-dim
              slice fits the small-bucket threshold (experts axis, N <= ~2048).
    Backward: the topk backward scatters grad_values back into a zeros tensor of
              the full [num_tokens, num_experts] logits shape (a scatter/index
              kernel) -- same memory volume as the forward selection, which is why
              compute_topk_bwd_time_s models it at the forward's element count.

Usage on a sleeper pod (one SKU at a time):
    # A100 / H100 / H200 / B200, auto-detect device label:
    python topk_benchmark.py --output topk_timings.parquet

    # Override device label (useful when torch.cuda.get_device_name is ambiguous):
    python topk_benchmark.py --device b200 --output topk_b200.parquet

    # Target shape only (the 5p3b config point):
    python topk_benchmark.py --only-target
"""

from __future__ import annotations

import argparse
import itertools
import platform
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd
import torch


# Canonical shapes to sweep. The 5p3b config point (8192, 128, k=3, fp32) is
# always included regardless of sweep overrides.
DEFAULT_NUM_TOKENS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_NUM_EXPERTS = [32, 64, 128, 256]
DEFAULT_K = [1, 2, 3, 4, 8]
DEFAULT_DTYPES = ["fp32", "bf16"]

# Target point: fp32 logits (moe_router_dtype: fp32), seq=8192, experts=128, k=3
TARGET_POINT = dict(num_tokens=8192, num_experts=128, k=3, dtype="fp32")

DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# Map torch.cuda.get_device_name() substrings to canonical labels.
DEVICE_AUTODETECT = [
    ("B200", "b200"),
    ("H200", "h200"),
    ("H100", "h100"),
    ("A100", "a100"),
]


@dataclass
class BenchRow:
    device: str
    gpu_name: str
    num_tokens: int
    num_experts: int
    k: int
    dtype: str
    n_elements: int
    # Forward (torch.topk selection).
    event_time_ms: float
    profiler_kernel_ms: float
    kernel_name: str
    elements_per_sec_event: float
    elements_per_sec_kernel: float
    # Backward (autograd scatter of grad_values into the full logits tensor).
    bwd_event_time_ms: float
    bwd_elements_per_sec_event: float


def autodetect_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; cannot benchmark torch.topk on GPU")
    name = torch.cuda.get_device_name(0)
    for needle, label in DEVICE_AUTODETECT:
        if needle in name:
            return label
    raise RuntimeError(
        f"Could not autodetect device label from '{name}'. Pass --device explicitly."
    )


def make_input(num_tokens: int, num_experts: int, dtype: torch.dtype) -> torch.Tensor:
    # Use randn so logits have a realistic spread; fp32 sigmoid + expert_bias in the
    # real router adds a small offset but does not change TopK kernel behavior.
    return torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")


def bench_bwd_event(x: torch.Tensor, k: int, n_warmup: int, n_iter: int) -> float:
    """Wall-clock of the topk BACKWARD over n_iter calls (CUDA events, ms per call).

    Isolates the backward by re-running only ``.backward()`` on a retained graph:
    build the graph once (topk on a leaf requiring grad), then time repeated
    backward passes with a fixed grad seed. ``retain_graph=True`` keeps the graph
    alive across iters; we zero the leaf grad each iter so it doesn't accumulate
    (accumulation is a cheap add and would otherwise grow unbounded). The backward
    of topk is a scatter of grad_values into a zeros tensor shaped like ``x`` --
    the same memory volume as the forward selection.
    """
    leaf = x.detach().clone().requires_grad_(True)
    values, _ = torch.topk(leaf, k=k, dim=1)
    grad_seed = torch.ones_like(values)

    for _ in range(n_warmup):
        leaf.grad = None
        values.backward(grad_seed, retain_graph=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        leaf.grad = None
        values.backward(grad_seed, retain_graph=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def bench_event(
    x: torch.Tensor, k: int, n_warmup: int, n_iter: int
) -> float:
    """Wall-clock of torch.topk over n_iter calls using CUDA events, ms per call."""
    for _ in range(n_warmup):
        torch.topk(x, k=k, dim=1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        torch.topk(x, k=k, dim=1)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def bench_profiler(
    x: torch.Tensor, k: int, n_warmup: int, n_iter: int
) -> tuple[float, str]:
    """Isolate the CUDA kernel time using torch.profiler, ms per call.

    Returns (avg_kernel_ms, kernel_name). kernel_name is the first CUDA kernel
    that matches '*opK*' / '*topk*' / 'sbtopk' / 'mbtopk' in the trace.
    """
    for _ in range(n_warmup):
        torch.topk(x, k=k, dim=1)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        for _ in range(n_iter):
            torch.topk(x, k=k, dim=1)
        torch.cuda.synchronize()

    total_us = 0
    calls = 0
    kernel_name = ""
    for evt in prof.key_averages():
        nm = evt.key
        if (
            "opK" in nm
            or "topk" in nm.lower()
            or "sbtopk" in nm
            or "mbtopk" in nm
        ) and evt.device_type == torch.autograd.DeviceType.CUDA:
            # torch >= 2.3 renamed `self_cuda_time_total` to `self_device_time_total`.
            self_us = getattr(
                evt,
                "self_device_time_total",
                getattr(evt, "self_cuda_time_total", 0),
            )
            total_us += self_us
            calls += evt.count
            if not kernel_name or "gatherTopK" in nm:
                kernel_name = nm

    if calls == 0:
        return float("nan"), "UNKNOWN"
    return (total_us / calls) / 1000.0, kernel_name


def benchmark_point(
    device_label: str,
    gpu_name: str,
    num_tokens: int,
    num_experts: int,
    k: int,
    dtype: str,
    n_warmup: int,
    n_iter: int,
) -> Optional[BenchRow]:
    if k > num_experts:
        return None
    torch_dtype = DTYPE_MAP[dtype]
    x = make_input(num_tokens, num_experts, torch_dtype)

    event_ms = bench_event(x, k, n_warmup, n_iter)
    kernel_ms, kernel_name = bench_profiler(x, k, n_warmup, n_iter)
    bwd_event_ms = bench_bwd_event(x, k, n_warmup, n_iter)

    n_elements = num_tokens * num_experts
    eps_event = n_elements / (event_ms / 1000.0) if event_ms > 0 else 0.0
    eps_kernel = (
        n_elements / (kernel_ms / 1000.0)
        if kernel_ms and kernel_ms > 0
        else 0.0
    )
    bwd_eps_event = (
        n_elements / (bwd_event_ms / 1000.0) if bwd_event_ms > 0 else 0.0
    )

    return BenchRow(
        device=device_label,
        gpu_name=gpu_name,
        num_tokens=num_tokens,
        num_experts=num_experts,
        k=k,
        dtype=dtype,
        n_elements=n_elements,
        event_time_ms=event_ms,
        profiler_kernel_ms=kernel_ms,
        kernel_name=kernel_name,
        elements_per_sec_event=eps_event,
        elements_per_sec_kernel=eps_kernel,
        bwd_event_time_ms=bwd_event_ms,
        bwd_elements_per_sec_event=bwd_eps_event,
    )


def build_grid(
    num_tokens: List[int],
    num_experts: List[int],
    ks: List[int],
    dtypes: List[str],
    only_target: bool,
) -> List[dict]:
    if only_target:
        return [dict(TARGET_POINT)]
    grid = [
        dict(num_tokens=nt, num_experts=ne, k=k, dtype=dt)
        for nt, ne, k, dt in itertools.product(num_tokens, num_experts, ks, dtypes)
        if k <= ne
    ]
    # Make sure the target point is in the grid even if overrides exclude it.
    if TARGET_POINT not in grid:
        grid.append(dict(TARGET_POINT))
    return grid


def summarize(df: pd.DataFrame) -> None:
    print("\n=== Summary ===")
    print(f"Device: {df['device'].iloc[0]} ({df['gpu_name'].iloc[0]})")
    print(f"Rows: {len(df)}")

    target_row = df[
        (df["num_tokens"] == TARGET_POINT["num_tokens"])
        & (df["num_experts"] == TARGET_POINT["num_experts"])
        & (df["k"] == TARGET_POINT["k"])
        & (df["dtype"] == TARGET_POINT["dtype"])
    ]
    if len(target_row):
        r = target_row.iloc[0]
        print(
            "\nTarget point (num_tokens=8192, num_experts=128, k=3, fp32):"
        )
        print(f"  Kernel: {r['kernel_name']}")
        print(f"  Fwd event wall-clock: {r['event_time_ms']:.4f} ms")
        print(f"  Fwd profiler kernel:  {r['profiler_kernel_ms']:.4f} ms")
        print(f"  Fwd elements/sec (event):    {r['elements_per_sec_event']:.3e}")
        print(f"  Fwd elements/sec (kernel):   {r['elements_per_sec_kernel']:.3e}")
        print(f"  Bwd event wall-clock: {r['bwd_event_time_ms']:.4f} ms")
        print(f"  Bwd elements/sec (event):    {r['bwd_elements_per_sec_event']:.3e}")

    print("\nRecommended TOPK_THROUGHPUT constants (median elem/sec across grid):")
    med_event = df["elements_per_sec_event"].median()
    med_kernel = df.loc[df["elements_per_sec_kernel"] > 0, "elements_per_sec_kernel"].median()
    med_bwd = df.loc[df["bwd_elements_per_sec_event"] > 0, "bwd_elements_per_sec_event"].median()
    print(f"  FWD median event:  {med_event:.3e}  (moe_router_util.calculate_topk_time)")
    print(f"  FWD median kernel: {med_kernel:.3e}")
    print(f"  BWD median event:  {med_bwd:.3e}  (backward.compute_topk_bwd_time_s)")
    if med_event > 0:
        print(f"  BWD/FWD event ratio: {med_bwd / med_event:.2f}x "
              f"(backward vs forward throughput)")

    print("\nKernel names observed:")
    for nm, cnt in df["kernel_name"].value_counts().items():
        print(f"  {cnt:4d}  {nm}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark torch.topk (ATen sbtopk::gatherTopK) for MoE routing"
    )
    parser.add_argument(
        "--device",
        choices=["a100", "h100", "h200", "b200"],
        default=None,
        help="Device label. If omitted, autodetect from torch.cuda.get_device_name.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="topk_timings.parquet",
        help="Output parquet path (also writes .csv next to it)",
    )
    parser.add_argument("--num-tokens", type=int, nargs="+", default=DEFAULT_NUM_TOKENS)
    parser.add_argument(
        "--num-experts", type=int, nargs="+", default=DEFAULT_NUM_EXPERTS
    )
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_K)
    parser.add_argument("--dtypes", type=str, nargs="+", default=DEFAULT_DTYPES)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument(
        "--only-target",
        action="store_true",
        help="Run only the target shape (8192 x 128, k=3, fp32).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available on this host.", file=sys.stderr)
        return 2

    device_label = args.device or autodetect_device()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Device label: {device_label}")
    print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Host: {platform.node()}")

    grid = build_grid(
        args.num_tokens, args.num_experts, args.ks, args.dtypes, args.only_target
    )
    print(f"Grid size: {len(grid)}")

    rows: List[BenchRow] = []
    for i, cfg in enumerate(grid, 1):
        print(
            f"[{i}/{len(grid)}] "
            f"num_tokens={cfg['num_tokens']:>6} "
            f"num_experts={cfg['num_experts']:>4} "
            f"k={cfg['k']} dtype={cfg['dtype']}",
            end=" ... ",
            flush=True,
        )
        try:
            row = benchmark_point(
                device_label=device_label,
                gpu_name=gpu_name,
                n_warmup=args.n_warmup,
                n_iter=args.n_iter,
                **cfg,
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        if row is None:
            print("skipped (k > num_experts)")
            continue
        rows.append(row)
        print(
            f"event={row.event_time_ms:.4f}ms kernel={row.profiler_kernel_ms:.4f}ms "
            f"({row.kernel_name})"
        )

    if not rows:
        print("No rows produced.", file=sys.stderr)
        return 1

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_parquet(args.output, index=False)
    csv_path = args.output.rsplit(".", 1)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {args.output}")
    print(f"Saved: {csv_path}")

    summarize(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
