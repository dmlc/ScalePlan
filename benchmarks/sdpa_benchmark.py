"""Benchmark SDPA forward and backward wall-clock for the performance model.

Writes a parquet indexed by (seq_len, micro_bs, n_q_heads, n_kv_heads, head_dim)
with columns:
    device          - GPU label ("a100", "h200", "b200")
    dtype           - precision ("bf16", "fp16")
    fwd_time_ms_med - median forward wall-clock, milliseconds per call
    bwd_time_ms_med - median backward wall-clock, milliseconds per call
    fwd_tflops      - 4 * s * m * n_q_heads * head_dim * 2 FLOPs / time
    bwd_tflops      - 10 * s * m * n_q_heads * head_dim FLOPs / time

Two implementations are measured so we can sanity-check against each other:
- TE:  transformer_engine.pytorch.DotProductAttention (the kernel used in
       production Megatron training).
- PT:  torch.nn.functional.scaled_dot_product_attention (SDPA backend;
       requires head_dim <= 128 and power-of-2 head_dim on most hardware).

Usage on a sleeper (torch available as /usr/bin/python):
    python sdpa_benchmark.py --device b200 --output sdpa_b200.parquet
    python sdpa_benchmark.py --device a100 --output sdpa_a100.parquet
    python sdpa_benchmark.py --device h200 --output sdpa_h200.parquet
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


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# Shapes chosen to cover the configs in examples/*.yaml. Keep the grid small so
# each sleeper can finish in a few minutes.
DEFAULT_SEQ_LENS = [2048, 4096, 8192, 16384]
DEFAULT_MICRO_BS = [1, 2, 4]
# (n_q_heads, n_kv_heads) pairs - GQA-aware. Covers llama3 (64,8), 5p3b (32,8),
# gpt-oss (16,8), and some small cases for unit-test-style sanity checks.
DEFAULT_HEAD_PAIRS = [
    (8, 1),   # tiny/debug
    (16, 8),  # gpt-oss 120b
    (32, 8),  # 5p3b
    (64, 8),  # llama3-70b
]
DEFAULT_HEAD_DIMS = [64, 128]
DEFAULT_DTYPES = ["bf16"]

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
    dtype: str
    seq_len: int
    micro_bs: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    backend: str
    fwd_time_ms_med: float
    bwd_time_ms_med: float
    fwd_tflops: float
    bwd_tflops: float
    status: str


def autodetect_device() -> str:
    name = torch.cuda.get_device_name(0)
    for needle, label in DEVICE_AUTODETECT:
        if needle in name:
            return label
    raise RuntimeError(f"Could not autodetect device from '{name}'.")


def make_qkv(
    seq_len: int,
    micro_bs: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q,K,V in sbhd layout matching TE default, with requires_grad for bwd."""
    q = torch.randn(
        seq_len, micro_bs, n_q_heads, head_dim,
        dtype=dtype, device="cuda", requires_grad=True,
    )
    k = torch.randn(
        seq_len, micro_bs, n_kv_heads, head_dim,
        dtype=dtype, device="cuda", requires_grad=True,
    )
    v = torch.randn(
        seq_len, micro_bs, n_kv_heads, head_dim,
        dtype=dtype, device="cuda", requires_grad=True,
    )
    return q, k, v


def time_fn_median(fn, n_warmup: int, n_iter: int) -> float:
    """Return the median per-call time in milliseconds."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def bench_te(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_q_heads: int, n_kv_heads: int, head_dim: int,
    n_warmup: int, n_iter: int,
) -> tuple[float, float]:
    """Bench TE.DotProductAttention. Returns (fwd_ms, bwd_ms)."""
    import transformer_engine.pytorch as te

    dpa = te.DotProductAttention(
        num_attention_heads=n_q_heads,
        kv_channels=head_dim,
        num_gqa_groups=n_kv_heads,
        attention_dropout=0.0,
        qkv_format="sbhd",
        attn_mask_type="causal",
    ).cuda()

    def fwd_only():
        with torch.no_grad():
            return dpa(q.detach(), k.detach(), v.detach())

    # Grad tensor of the output shape. TE output is sbhd with hidden = n_q_heads*head_dim.
    out = dpa(q, k, v)
    grad_out = torch.randn_like(out)

    def fwd_plus_bwd():
        q.grad = None
        k.grad = None
        v.grad = None
        out = dpa(q, k, v)
        out.backward(grad_out, retain_graph=False)

    fwd_ms = time_fn_median(fwd_only, n_warmup, n_iter)
    total_ms = time_fn_median(fwd_plus_bwd, n_warmup, n_iter)
    bwd_ms = total_ms - fwd_ms
    return fwd_ms, bwd_ms


def bench_pt(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    n_q_heads: int, n_kv_heads: int, head_dim: int,
    n_warmup: int, n_iter: int,
) -> tuple[float, float]:
    """Bench torch.nn.functional.scaled_dot_product_attention.

    SDPA expects bhsd layout: (batch, heads, seq, head_dim). For GQA we
    repeat K/V heads to match Q heads. Returns (fwd_ms, bwd_ms).
    """
    import torch.nn.functional as F

    s, b, _, d = q.shape
    q_b = q.permute(1, 2, 0, 3).contiguous()  # (b, n_q_heads, s, d)
    k_b = k.permute(1, 2, 0, 3).contiguous()  # (b, n_kv_heads, s, d)
    v_b = v.permute(1, 2, 0, 3).contiguous()

    if n_kv_heads != n_q_heads:
        repeat = n_q_heads // n_kv_heads
        k_b = k_b.repeat_interleave(repeat, dim=1)
        v_b = v_b.repeat_interleave(repeat, dim=1)

    # These leaf tensors drive requires_grad through the permute.
    q_b.requires_grad_(True)
    k_b.requires_grad_(True)
    v_b.requires_grad_(True)

    def fwd_only():
        with torch.no_grad():
            return F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)

    out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
    grad_out = torch.randn_like(out)

    def fwd_plus_bwd():
        q_b.grad = None
        k_b.grad = None
        v_b.grad = None
        out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
        out.backward(grad_out, retain_graph=False)

    fwd_ms = time_fn_median(fwd_only, n_warmup, n_iter)
    total_ms = time_fn_median(fwd_plus_bwd, n_warmup, n_iter)
    bwd_ms = total_ms - fwd_ms
    return fwd_ms, bwd_ms


def benchmark_point(
    device: str, gpu_name: str, dtype: str,
    seq_len: int, micro_bs: int,
    n_q_heads: int, n_kv_heads: int, head_dim: int,
    backend: str, n_warmup: int, n_iter: int,
) -> Optional[BenchRow]:
    torch_dtype = DTYPE_MAP[dtype]
    if n_q_heads % n_kv_heads != 0:
        return None
    try:
        q, k, v = make_qkv(seq_len, micro_bs, n_q_heads, n_kv_heads, head_dim, torch_dtype)
        if backend == "te":
            fwd_ms, bwd_ms = bench_te(q, k, v, n_q_heads, n_kv_heads, head_dim, n_warmup, n_iter)
        elif backend == "pt":
            fwd_ms, bwd_ms = bench_pt(q, k, v, n_q_heads, n_kv_heads, head_dim, n_warmup, n_iter)
        else:
            raise ValueError(backend)
    except Exception as e:
        return BenchRow(
            device=device, gpu_name=gpu_name, dtype=dtype,
            seq_len=seq_len, micro_bs=micro_bs,
            n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            backend=backend,
            fwd_time_ms_med=float("nan"), bwd_time_ms_med=float("nan"),
            fwd_tflops=0.0, bwd_tflops=0.0,
            status=f"ERROR: {type(e).__name__}: {e}"[:120],
        )

    # Attention FLOPs (causal, halved): fwd = 2 * (1/2) * s * s * n_q_heads * head_dim * 2
    # Use non-causal count (s*s) for comparability with sdpa.parquet's tflops_te.
    fwd_flops = 4.0 * seq_len * seq_len * n_q_heads * head_dim * micro_bs
    bwd_flops = 10.0 * seq_len * seq_len * n_q_heads * head_dim * micro_bs
    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) / 1e12 if fwd_ms > 0 else 0.0
    bwd_tflops = bwd_flops / (bwd_ms * 1e-3) / 1e12 if bwd_ms > 0 else 0.0

    return BenchRow(
        device=device, gpu_name=gpu_name, dtype=dtype,
        seq_len=seq_len, micro_bs=micro_bs,
        n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        backend=backend,
        fwd_time_ms_med=fwd_ms, bwd_time_ms_med=bwd_ms,
        fwd_tflops=fwd_tflops, bwd_tflops=bwd_tflops,
        status="OK",
    )


def build_grid(
    seq_lens: List[int], micro_bss: List[int],
    head_pairs: List[tuple[int, int]], head_dims: List[int],
    dtypes: List[str], backends: List[str],
) -> List[dict]:
    grid = []
    for s, m, (qh, kvh), d, dt, be in itertools.product(
        seq_lens, micro_bss, head_pairs, head_dims, dtypes, backends
    ):
        grid.append(dict(
            seq_len=s, micro_bs=m, n_q_heads=qh, n_kv_heads=kvh,
            head_dim=d, dtype=dt, backend=be,
        ))
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description="SDPA fwd+bwd benchmark")
    parser.add_argument("--device", choices=["a100", "h100", "h200", "b200"], default=None)
    parser.add_argument("--output", default="sdpa_timings.parquet")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--micro-bs", type=int, nargs="+", default=DEFAULT_MICRO_BS)
    parser.add_argument("--head-dims", type=int, nargs="+", default=DEFAULT_HEAD_DIMS)
    parser.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES)
    parser.add_argument("--backends", nargs="+", default=["te", "pt"])
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    device_label = args.device or autodetect_device()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} -> label '{device_label}'")
    print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Host: {platform.node()}")

    grid = build_grid(
        args.seq_lens, args.micro_bs, DEFAULT_HEAD_PAIRS,
        args.head_dims, args.dtypes, args.backends,
    )
    print(f"Grid: {len(grid)} points")

    rows: List[BenchRow] = []
    for i, cfg in enumerate(grid, 1):
        print(
            f"[{i}/{len(grid)}] "
            f"backend={cfg['backend']} s={cfg['seq_len']:>5} b={cfg['micro_bs']} "
            f"qh={cfg['n_q_heads']} kvh={cfg['n_kv_heads']} d={cfg['head_dim']} "
            f"dt={cfg['dtype']}",
            end=" ... ", flush=True,
        )
        row = benchmark_point(
            device=device_label, gpu_name=gpu_name,
            n_warmup=args.n_warmup, n_iter=args.n_iter, **cfg,
        )
        if row is None:
            print("skipped")
            continue
        rows.append(row)
        if row.status == "OK":
            print(
                f"fwd={row.fwd_time_ms_med:.4f}ms "
                f"bwd={row.bwd_time_ms_med:.4f}ms "
                f"({row.fwd_tflops:.0f}/{row.bwd_tflops:.0f} TFLOPS)"
            )
        else:
            print(row.status)

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_parquet(args.output, index=False)
    csv_path = args.output.rsplit(".", 1)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {args.output}")
    print(f"Saved: {csv_path}")
    print("\nSummary (OK rows only):")
    ok = df[df["status"] == "OK"]
    print(f"  rows: {len(ok)}")
    if len(ok):
        print(f"  median fwd: {ok['fwd_time_ms_med'].median():.4f} ms")
        print(f"  median bwd: {ok['bwd_time_ms_med'].median():.4f} ms")
        print(f"  bwd/fwd:    {(ok['bwd_time_ms_med'] / ok['fwd_time_ms_med']).median():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
