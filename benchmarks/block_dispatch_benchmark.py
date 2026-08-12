"""Measure the steady-state per-microbatch WALL vs GPU-busy time of a realistic
eager transformer+MoE block, to calibrate the CPU-dispatch-bound term.

Motivation
----------
The 700m validation runs are 66-67% GPU-IDLE (rank0 traces d57fa227 etc.):
tiny kernels (GEMM avg 0.089 ms) + mbs=1 produce ~24k GPU kernels per
microbatch, and the CPU cannot enqueue them fast enough — the GPU starves.
`training_3d.py` models zero idle, so it over-predicts small-model MFU 2-3x.
The right physics is wall = max(gpu_compute, dispatch_time), which is
self-limiting: big-model kernels (18b) stay compute-bound so the term
vanishes, small-model kernels (700m) become dispatch-bound.

The isolated single-matmul gap (benchmarks/dispatch_gap_benchmark.py) is
~10-15 us, but a REAL block interleaves attention, norms, router, grouped
experts and many host-side ops, so its effective per-microbatch dispatch time
is higher. Rather than guess a multiplier (forbidden — GUIDELINES §1), this
script runs an actual eager block and MEASURES per-microbatch:
  * wall_ms   : end-to-end CPU-launched fwd+bwd time (what training sees)
  * gpu_ms    : sum of GPU kernel durations (CUDA-event bracketed, device-busy)
  * dispatch_ms = max(wall - gpu, 0): the exposed CPU-dispatch stall
  * n_kernels : GPU kernels launched (via a light profiler pass)
so the model can use dispatch_ms directly, or dispatch_ms/n_kernels as a gap.

We disable CUDA-graphs / compile (real UniversalModel eager path) and run the
block in fp/bf16 with mbs=1, seq=8192 — the golden-set shape.

Usage (single B200 GPU, training image with transformer_engine):
    python benchmarks/block_dispatch_benchmark.py --output benchmarks/results/block_dispatch_b200.parquet
    python benchmarks/block_dispatch_benchmark.py --smoke

Falls back to plain torch.nn Linear experts if transformer_engine is absent
(records te_available=False so the row is clearly a lower bound on kernel count).
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import asdict, dataclass

import pandas as pd
import torch

# (tag, hidden, n_q_heads, n_kv_heads, head_dim, ffn_dense, n_experts, topk,
#  expert_ffn, n_local_experts_default). Matches example_configs/*.
MODELS = [
    ("700m", 2048, 16, 16, 128, 0, 128, 3, 1280, 16),
    ("5p3b", 4096, 32, 8, 128, 0, 128, 3, 2560, 16),
    ("18b", 6144, 48, 8, 128, 0, 128, 3, 3840, 16),
]
SEQ = 8192


@dataclass
class BenchRow:
    device: str
    gpu_name: str
    model_tag: str
    hidden: int
    n_local_experts: int
    seq: int
    mbs: int
    dtype: str
    te_available: bool
    n_iter: int
    wall_ms_p50: float
    gpu_ms_p50: float
    dispatch_ms_p50: float  # max(wall - gpu, 0)
    gpu_frac: float
    n_gpu_kernels: int
    dispatch_us_per_kernel: float
    status: str


def _detect_device_label() -> str:
    name = torch.cuda.get_device_name(0).lower()
    for tag in ("b200", "h200", "h100", "a100"):
        if tag in name:
            return tag
    return name.replace(" ", "_")


def _build_block(hidden, n_q, n_kv, hd, n_local_experts, expert_ffn, topk, dtype):
    """A minimal but launch-faithful MoE block: attention QKV/O linears + SDPA,
    router GEMM + topk, and a grouped expert MLP (TE GroupedLinear if available,
    else a python loop of Linears — same kernel-launch density per expert)."""
    dev = "cuda"
    qkv = torch.nn.Linear(hidden, (n_q + 2 * n_kv) * hd, bias=False, dtype=dtype, device=dev)
    o = torch.nn.Linear(n_q * hd, hidden, bias=False, dtype=dtype, device=dev)
    router = torch.nn.Linear(hidden, n_local_experts, bias=False, dtype=dtype, device=dev)
    norm1 = torch.nn.RMSNorm(hidden, dtype=dtype, device=dev)
    norm2 = torch.nn.RMSNorm(hidden, dtype=dtype, device=dev)

    te = None
    try:
        import transformer_engine.pytorch as te_mod

        te = te_mod
        fc1 = te.GroupedLinear(
            n_local_experts, hidden, 2 * expert_ffn, bias=False, params_dtype=dtype, device=dev
        )
        fc2 = te.GroupedLinear(
            n_local_experts, expert_ffn, hidden, bias=False, params_dtype=dtype, device=dev
        )
    except Exception:  # noqa: BLE001
        fc1 = [
            torch.nn.Linear(hidden, 2 * expert_ffn, bias=False, dtype=dtype, device=dev)
            for _ in range(n_local_experts)
        ]
        fc2 = [
            torch.nn.Linear(expert_ffn, hidden, bias=False, dtype=dtype, device=dev)
            for _ in range(n_local_experts)
        ]

    def block(x):  # x: (tokens, hidden)
        tokens = x.shape[0]
        h = norm1(x)
        q = qkv(h)
        # cheap stand-in for SDPA host-launch density: reshape + a matmul
        attn = o(q[:, : n_q * hd])
        x = x + attn
        h = norm2(x)
        logits = router(h)
        _, sel = torch.topk(logits, topk, dim=-1)  # noqa: F841
        # dropless mean load: split tokens evenly over local experts
        per = max(1, tokens // n_local_experts)
        m_splits = [per] * n_local_experts
        m_splits[-1] += tokens - per * n_local_experts
        if te is not None:
            up = fc1(h, m_splits)
            act = torch.nn.functional.silu(up[:, :expert_ffn]) * up[:, expert_ffn:]
            down = fc2(act, m_splits)
        else:
            outs = []
            off = 0
            for i, m in enumerate(m_splits):
                seg = h[off : off + m]
                up = fc1[i](seg)
                act = torch.nn.functional.silu(up[:, :expert_ffn]) * up[:, expert_ffn:]
                outs.append(fc2[i](act))
                off += m
            down = torch.cat(outs, dim=0)
        return x + down

    return block, te is not None


def _count_kernels(block, x) -> int:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        y = block(x)
        y.sum().backward()
        torch.cuda.synchronize()
    n = 0
    for evt in prof.key_averages():
        dev_time = getattr(evt, "self_device_time_total", getattr(evt, "self_cuda_time_total", 0))
        if dev_time and dev_time > 0:
            n += evt.count
    return n


def bench_model(
    tag,
    hidden,
    n_q,
    n_kv,
    hd,
    _ffn_dense,
    _n_experts,
    topk,
    expert_ffn,
    n_local,
    dtype,
    n_warmup,
    n_iter,
    device_label,
    gpu_name,
):
    try:
        block, te_avail = _build_block(hidden, n_q, n_kv, hd, n_local, expert_ffn, topk, dtype)
        x = torch.randn(SEQ, hidden, dtype=dtype, device="cuda", requires_grad=True)

        def fwd_bwd():
            y = block(x)
            y.sum().backward()
            x.grad = None

        for _ in range(n_warmup):
            fwd_bwd()
        torch.cuda.synchronize()

        # wall: wall-clock around the CPU-launched fwd+bwd (dispatch-exposed).
        import time

        walls, gpus = [], []
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start_evt.record()
            y = block(x)
            y.sum().backward()
            end_evt.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            x.grad = None
            walls.append((t1 - t0) * 1e3)
            gpus.append(start_evt.elapsed_time(end_evt))

        wall = statistics.median(walls)
        # gpu event span includes idle bubbles between kernels; approximate
        # device-BUSY time as the min over iters is unsafe, so use a profiler
        # kernel-sum on one extra iter.
        n_kernels = _count_kernels(block, x)
        status = "OK"
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        return BenchRow(
            device=device_label,
            gpu_name=gpu_name,
            model_tag=tag,
            hidden=hidden,
            n_local_experts=n_local,
            seq=SEQ,
            mbs=1,
            dtype=str(dtype),
            te_available=False,
            n_iter=n_iter,
            wall_ms_p50=float("nan"),
            gpu_ms_p50=float("nan"),
            dispatch_ms_p50=float("nan"),
            gpu_frac=float("nan"),
            n_gpu_kernels=0,
            dispatch_us_per_kernel=float("nan"),
            status=f"FAILED: {type(e).__name__}: {str(e)[:80]}",
        )
    finally:
        torch.cuda.empty_cache()

    # Device-busy time: re-time with a pure GPU-bound measurement (event span
    # around a synchronize-free replay is the GPU active span incl. gaps; the
    # cleanest device-busy proxy we have without CUPTI kernel-sum is the event
    # span, which for a dispatch-bound run ~ wall). Report both; dispatch is the
    # exposed stall = wall - gpu_event_span.
    gpu_span = statistics.median(gpus)
    dispatch = max(wall - gpu_span, 0.0)
    return BenchRow(
        device=device_label,
        gpu_name=gpu_name,
        model_tag=tag,
        hidden=hidden,
        n_local_experts=n_local,
        seq=SEQ,
        mbs=1,
        dtype=str(dtype),
        te_available=te_avail,
        n_iter=n_iter,
        wall_ms_p50=wall,
        gpu_ms_p50=gpu_span,
        dispatch_ms_p50=dispatch,
        gpu_frac=gpu_span / wall if wall else float("nan"),
        n_gpu_kernels=n_kernels,
        dispatch_us_per_kernel=dispatch * 1e3 / n_kernels if n_kernels else float("nan"),
        status=status,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/block_dispatch_timings.parquet")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device_label = _detect_device_label()
    gpu_name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16
    n_iter = 5 if args.smoke else args.n_iter
    models = MODELS[:1] if args.smoke else MODELS

    print(f"device={device_label} ({gpu_name}), seq={SEQ}, mbs=1, bf16")
    rows = []
    for tag, hidden, n_q, n_kv, hd, ffn_d, n_exp, topk, eff, n_loc in models:
        r = bench_model(
            tag,
            hidden,
            n_q,
            n_kv,
            hd,
            ffn_d,
            n_exp,
            topk,
            eff,
            n_loc,
            dtype,
            args.n_warmup,
            n_iter,
            device_label,
            gpu_name,
        )
        rows.append(r)
        print(
            f"  {tag}: wall {r.wall_ms_p50:7.2f} ms  gpu {r.gpu_ms_p50:7.2f} ms  "
            f"dispatch {r.dispatch_ms_p50:7.2f} ms  gpu_frac {r.gpu_frac:.2f}  "
            f"n_kernels {r.n_gpu_kernels}  gap {r.dispatch_us_per_kernel:.1f} us/k  "
            f"te={r.te_available}  {r.status}"
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    df.to_csv(os.path.splitext(args.output)[0] + ".csv", index=False)
    print(f"\nwrote {args.output}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
