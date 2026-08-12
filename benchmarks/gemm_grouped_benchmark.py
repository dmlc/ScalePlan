"""Benchmark TransformerEngine `te.pytorch.GroupedLinear` for MoE expert MLPs.

Motivation
----------
`dlcalc` times the MoE expert MLP by looking up `gemm_util.parquet` — a SINGLE
dense-GEMM utilization table keyed only by (M, N, K). But the expert MLP is a
GROUPED GEMM: `num_gemms = n_local_experts` independent matmuls (one per local
expert), each with M = that expert's token count, run by ONE fused kernel. The
single-GEMM table cannot express this, so `training_3d.py` currently looks the
util up at the aggregate `M_eff = n_local_experts * expert_capacity`, which
over-credits utilization by 5-40x (see perf_scratch/MODEL_VS_TRACE.md table D):
16 tiny M=192 GEMMs do NOT run at the efficiency of one M=3072 GEMM.

This script measures the REAL operator the training stack runs so the expert-MLP
cost model can be rebuilt from measured grouped-kernel efficiency instead of a
single-GEMM proxy.

Operator of interest
--------------------
`transformer_engine.pytorch.GroupedLinear` — this is what AGI3P-Megatron-LM's
`TEGroupedMLP` (linear_fc1=TEColumnParallelGroupedLinear,
linear_fc2=TERowParallelGroupedLinear) wraps when `moe_grouped_gemm=True` and
`use_transformer_engine=True` (the UniversalModel default; see
moe_module_specs.py:47-57). It is NOT the fanshiqing `grouped_gemm.ops.gmm`
(legacy `GroupedMLP`) or `SequentialMLP`.

We benchmark BOTH expert linears the way TEGroupedMLP builds them (experts.py):
  fc1: in_features = hidden,           out_features = 2*ffn (SwiGLU gate+up)
  fc2: in_features = ffn,              out_features = hidden
called as `layer(x, m_splits)` where m_splits is the per-expert token-count list
(sum == total permuted tokens). The MoE dispatcher handles TP/EP comm, so the
expert linears run with parallel_mode=None — i.e. this measures the LOCAL
per-rank grouped GEMM (num_gemms = n_local_experts), exactly the trace kernel.

Token distribution
------------------
The sonic/utm_moe config is DROPLESS (`moe_expert_capacity_factor: null`), so
tokens-per-expert is uneven at runtime. We sweep both:
  * "balanced": every expert gets the mean load (what dlcalc approximates with
    a fixed expert_capacity) — the clean point to calibrate util against.
  * "skewed":   a realistic imbalance (a few hot experts) — shows how much the
    grouped kernel degrades under load imbalance, which dropless training incurs.

Usage (inside the training image on a B200 sleeper pod, one SKU at a time)
--------------------------------------------------------------------------
    # Full sweep, autodetect device label from torch.cuda.get_device_name:
    python gemm_grouped_benchmark.py --output gemm_grouped_b200.parquet

    # fp8 too (the training image is bf16 by default; fp8=null in sonic):
    python gemm_grouped_benchmark.py --dtypes bf16 fp8 --output gemm_grouped_b200.parquet

    # Only the sonic config points (mean-load, k=3):
    python gemm_grouped_benchmark.py --only-target

Emits a parquet + csv keyed by
    (device, gpu_name, num_gemms, tokens_per_expert, ffn_hidden, hidden, dtype,
     proj, distribution)
with measured fwd / bwd / fwd+bwd kernel time, achieved TFLOP/s, and utilization.
`tokens_per_expert` is the MEAN per-expert token count (mean of m_splits); the
exact m_splits used are recorded in `m_splits_repr` for reproducibility.
"""

from __future__ import annotations

import argparse
import itertools
import platform
import sys
from dataclasses import asdict, dataclass

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Sweep defaults. Sized to the sonic/utm_moe ladder + neighbours so the model
# can interpolate. Override any axis on the CLI.
# ---------------------------------------------------------------------------
# num_gemms = n_local_experts = n_experts / EP. n_experts=128, EP in {1,2,4,8,32}
# -> n_local in {128,64,32,16,4}; include 8 for EP=16.
DEFAULT_NUM_GEMMS = [4, 8, 16, 32, 64, 128]
# mean tokens/expert. Dropless mean ≈ seq*mbs*top_k / n_experts. For the golden
# set this is ~192 (8192*1*3/128); sweep around it and up to the well-filled regime.
DEFAULT_TOKENS_PER_EXPERT = [64, 128, 192, 256, 512, 1024, 2048, 4096]
# (hidden, ffn) pairs across the sonic rungs and neighbours. ffn = moe_ffn_hidden_size.
DEFAULT_SHAPES = [
    (1024, 1280),
    (2048, 1280),
    (2560, 1600),
    (4096, 1280),
    (6144, 1536),
]
DEFAULT_DTYPES = ["bf16"]

# Sonic config point: hidden=2048, ffn=1280, k=3, dropless mean load 192, EP8 -> 16 experts.
TARGET_POINTS = [
    dict(num_gemms=16, tokens_per_expert=192, hidden=2048, ffn=1280, dtype="bf16"),
]

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    # fp8 handled specially via te.fp8_autocast; the tensor dtype stays bf16.
    "fp8": torch.bfloat16,
}

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
    proj: str  # "fc1" (hidden->2*ffn, SwiGLU) or "fc2" (ffn->hidden)
    distribution: str  # "balanced" or "skewed"
    num_gemms: int
    tokens_per_expert: int  # MEAN per-expert token count
    total_tokens: int
    hidden: int
    ffn: int
    in_features: int
    out_features: int
    dtype: str
    n_flops: float  # 2*M*N*K over all groups, fwd
    fwd_ms: float
    bwd_ms: float
    fwd_bwd_ms: float
    fwd_tflops: float
    bwd_tflops: float
    fwd_util_pct: float
    bwd_util_pct: float
    kernel_name: str
    n_fwd_kernels: int  # GPU kernels launched by one forward grouped GEMM
    m_splits_repr: str
    te_version: str
    status: str


# Peak dense bf16/fp16 TFLOP/s per SKU (datasheet, no sparsity). fp8 doubles.
PEAK_TFLOPS = {
    "a100": {"bf16": 312, "fp16": 312, "fp8": 624, "fp32": 19.5},
    "h100": {"bf16": 989.4, "fp16": 989.4, "fp8": 1978.9, "fp32": 67},
    "h200": {"bf16": 989.4, "fp16": 989.4, "fp8": 1978.9, "fp32": 67},
    "b200": {"bf16": 2250, "fp16": 2250, "fp8": 4500, "fp32": 80},
}


def autodetect_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; cannot benchmark GroupedLinear on GPU")
    name = torch.cuda.get_device_name(0)
    for needle, label in DEVICE_AUTODETECT:
        if needle in name:
            return label
    raise RuntimeError(
        f"Could not autodetect device label from '{name}'. Pass --device explicitly."
    )


def make_m_splits(num_gemms: int, tokens_per_expert: int, distribution: str) -> list[int]:
    """Per-expert token counts summing to num_gemms * tokens_per_expert.

    balanced: every expert == tokens_per_expert.
    skewed:   a realistic dropless imbalance — top ~25% of experts get ~2x the
              mean, the rest share the remainder (min 1 token), total preserved.
    """
    total = num_gemms * tokens_per_expert
    if distribution == "balanced" or num_gemms == 1:
        return [tokens_per_expert] * num_gemms
    n_hot = max(1, num_gemms // 4)
    hot = min(total // n_hot, tokens_per_expert * 2)
    splits = [hot] * n_hot
    remaining = total - hot * n_hot
    n_cold = num_gemms - n_hot
    base = max(1, remaining // n_cold)
    splits += [base] * n_cold
    # fix rounding so the sum is exactly `total` (grouped GEMM needs sum==rows)
    diff = total - sum(splits)
    splits[-1] += diff
    if splits[-1] < 1:  # pathological; fall back to balanced
        return [tokens_per_expert] * num_gemms
    return splits


def _import_te():
    try:
        import transformer_engine  # noqa: F401
        import transformer_engine.pytorch as te

        ver = getattr(transformer_engine, "__version__", "unknown")
        return te, ver
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "transformer_engine is not importable. Run this INSIDE the training "
            f"image (agi-training-...-nv4-p6), which ships TE. Import error: {e}"
        )


def _time_fwd_bwd(
    layer, x: torch.Tensor, m_splits: list[int], fp8: bool, n_warmup: int, n_iter: int
) -> tuple[float, float, str]:
    """CUDA-event timing of fwd and fwd+bwd, ms per call. Returns (fwd_ms, fwd_bwd_ms, kernel_name)."""
    import contextlib

    te, _ = _import_te()
    fp8_ctx = te.fp8_autocast(enabled=True) if fp8 else contextlib.nullcontext()

    def fwd():
        with fp8_ctx:
            out = layer(x, m_splits)
        # GroupedLinear returns (out, bias) when return_bias, else out. Normalize.
        return out[0] if isinstance(out, (tuple, list)) else out

    # ---- warmup (both fwd and bwd) ----
    for _ in range(n_warmup):
        x.grad = None
        y = fwd()
        y.sum().backward()
    torch.cuda.synchronize()

    # ---- forward-only ----
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fwd()
        end.record()
        torch.cuda.synchronize()
        fwd_ms = start.elapsed_time(end) / n_iter

    # ---- fwd + bwd ----
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        x.grad = None
        y = fwd()
        y.sum().backward()
    end.record()
    torch.cuda.synchronize()
    fwd_bwd_ms = start.elapsed_time(end) / n_iter

    # ---- kernel name + GPU-kernel COUNT via one profiled fwd ----
    # The count is the real driver of the small-model dispatch bound: a grouped
    # GEMM over many tiny experts launches thousands of tile-kernels (the trace
    # nvjet_tst_* flood), so fwd n_kernels(num_gemms, tokens/expert) is the
    # measured input the dispatch model needs.
    kernel_name = ""
    n_fwd_kernels = 0
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False
        ) as prof:
            with torch.no_grad():
                fwd()
            torch.cuda.synchronize()
        best_us = 0.0
        for evt in prof.key_averages():
            if evt.device_type != torch.autograd.DeviceType.CUDA:
                continue
            us = getattr(evt, "self_device_time_total", getattr(evt, "self_cuda_time_total", 0))
            if us and us > 0:
                n_fwd_kernels += evt.count
            if us > best_us:
                best_us = us
                kernel_name = evt.key
    except Exception:  # noqa: BLE001
        kernel_name = "profiler_failed"

    return fwd_ms, fwd_bwd_ms, kernel_name, n_fwd_kernels


def benchmark_point(
    device_label: str,
    gpu_name: str,
    proj: str,
    distribution: str,
    num_gemms: int,
    tokens_per_expert: int,
    hidden: int,
    ffn: int,
    dtype: str,
    n_warmup: int,
    n_iter: int,
    te_version: str,
) -> BenchRow | None:
    te, _ = _import_te()

    # fc1: hidden -> 2*ffn (SwiGLU gate+up).  fc2: ffn -> hidden.
    if proj == "fc1":
        in_features, out_features = hidden, 2 * ffn
    elif proj == "fc2":
        in_features, out_features = ffn, hidden
    else:
        raise ValueError(proj)

    fp8 = dtype == "fp8"
    torch_dtype = DTYPE_MAP[dtype]
    m_splits = make_m_splits(num_gemms, tokens_per_expert, distribution)
    total_tokens = sum(m_splits)

    # Build the real TE op. parallel_mode=None: the MoE dispatcher owns TP/EP comm,
    # so the expert GroupedLinear is a plain local grouped GEMM (what the trace shows).
    try:
        layer = te.GroupedLinear(
            num_gemms,
            in_features,
            out_features,
            bias=False,
            params_dtype=torch_dtype,
            device="cuda",
        )
    except Exception as e:  # noqa: BLE001
        return BenchRow(
            device=device_label,
            gpu_name=gpu_name,
            proj=proj,
            distribution=distribution,
            num_gemms=num_gemms,
            tokens_per_expert=tokens_per_expert,
            total_tokens=total_tokens,
            hidden=hidden,
            ffn=ffn,
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,
            n_flops=0.0,
            fwd_ms=float("nan"),
            bwd_ms=float("nan"),
            fwd_bwd_ms=float("nan"),
            fwd_tflops=0.0,
            bwd_tflops=0.0,
            fwd_util_pct=0.0,
            bwd_util_pct=0.0,
            kernel_name="",
            m_splits_repr=repr(m_splits),
            te_version=te_version,
            status=f"CONSTRUCT_FAILED: {type(e).__name__}: {e}",
        )

    x = torch.randn(total_tokens, in_features, dtype=torch_dtype, device="cuda", requires_grad=True)

    try:
        fwd_ms, fwd_bwd_ms, kernel_name, n_fwd_kernels = _time_fwd_bwd(
            layer, x, m_splits, fp8, n_warmup, n_iter
        )
        status = "OK"
    except Exception as e:  # noqa: BLE001
        return BenchRow(
            device=device_label,
            gpu_name=gpu_name,
            proj=proj,
            distribution=distribution,
            num_gemms=num_gemms,
            tokens_per_expert=tokens_per_expert,
            total_tokens=total_tokens,
            hidden=hidden,
            ffn=ffn,
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,
            n_flops=0.0,
            fwd_ms=float("nan"),
            bwd_ms=float("nan"),
            fwd_bwd_ms=float("nan"),
            fwd_tflops=0.0,
            bwd_tflops=0.0,
            fwd_util_pct=0.0,
            bwd_util_pct=0.0,
            kernel_name="",
            n_fwd_kernels=0,
            m_splits_repr=repr(m_splits),
            te_version=te_version,
            status=f"RUN_FAILED: {type(e).__name__}: {e}",
        )

    bwd_ms = max(fwd_bwd_ms - fwd_ms, 0.0)

    # FLOPs. Forward grouped GEMM over all groups: sum_g 2 * m_g * out * in.
    # With equal in/out per group, total_tokens * 2 * out_features * in_features.
    fwd_flops = 2.0 * total_tokens * out_features * in_features
    # Backward is ~2x forward (dgrad + wgrad).
    bwd_flops = 2.0 * fwd_flops

    def tflops(flops, ms):
        return (flops / (ms / 1000.0)) / 1e12 if ms and ms > 0 else 0.0

    peak = PEAK_TFLOPS.get(device_label, {}).get(dtype, 0.0)
    fwd_tf = tflops(fwd_flops, fwd_ms)
    bwd_tf = tflops(bwd_flops, bwd_ms)

    return BenchRow(
        device=device_label,
        gpu_name=gpu_name,
        proj=proj,
        distribution=distribution,
        num_gemms=num_gemms,
        tokens_per_expert=tokens_per_expert,
        total_tokens=total_tokens,
        hidden=hidden,
        ffn=ffn,
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
        n_flops=fwd_flops,
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        fwd_bwd_ms=fwd_bwd_ms,
        fwd_tflops=fwd_tf,
        bwd_tflops=bwd_tf,
        fwd_util_pct=100.0 * fwd_tf / peak if peak else 0.0,
        bwd_util_pct=100.0 * bwd_tf / peak if peak else 0.0,
        kernel_name=kernel_name,
        n_fwd_kernels=n_fwd_kernels,
        m_splits_repr=repr(m_splits),
        te_version=te_version,
        status=status,
    )


def build_grid(args) -> list[dict]:
    if args.only_target:
        grid = []
        for tp in TARGET_POINTS:
            for proj in ("fc1", "fc2"):
                for dist in ("balanced",):
                    grid.append({**tp, "proj": proj, "distribution": dist})
        return grid

    shapes = (
        [tuple(int(x) for x in s.split(":")) for s in args.shapes]
        if args.shapes
        else DEFAULT_SHAPES
    )
    grid = []
    for num_gemms, tpe, (hidden, ffn), dtype, proj, dist in itertools.product(
        args.num_gemms,
        args.tokens_per_expert,
        shapes,
        args.dtypes,
        ("fc1", "fc2"),
        args.distributions,
    ):
        grid.append(
            dict(
                num_gemms=num_gemms,
                tokens_per_expert=tpe,
                hidden=hidden,
                ffn=ffn,
                dtype=dtype,
                proj=proj,
                distribution=dist,
            )
        )
    return grid


def summarize(df: pd.DataFrame) -> None:
    print("\n=== Summary ===")
    ok = df[df["status"] == "OK"]
    print(
        f"Device: {df['device'].iloc[0]} ({df['gpu_name'].iloc[0]})  TE={df['te_version'].iloc[0]}"
    )
    print(f"Rows: {len(df)} ({len(ok)} OK, {len(df) - len(ok)} failed)")
    if ok.empty:
        print("No OK rows; check the status column for errors.")
        return

    tgt = ok[
        (ok.num_gemms == 16)
        & (ok.tokens_per_expert == 192)
        & (ok.hidden == 2048)
        & (ok.ffn == 1280)
        & (ok.distribution == "balanced")
    ]
    if len(tgt):
        print("\nSonic point (num_gemms=16, tokens/expert=192, hidden=2048, ffn=1280, balanced):")
        for _, r in tgt.iterrows():
            print(
                f"  {r['proj']}: fwd={r['fwd_ms']:.4f}ms util={r['fwd_util_pct']:.1f}%  "
                f"bwd={r['bwd_ms']:.4f}ms util={r['bwd_util_pct']:.1f}%  ({r['kernel_name'][:50]})"
            )
        print("  -> compare fwd_util_pct to the single-GEMM gemm_util.parquet value at")
        print(
            "     M_eff=3072 (~29%) vs per-group M=192 (~3%): the truth is the measured number here."
        )

    print(
        "\nfwd utilization %% by mean tokens/expert (balanced, fc1, median over shapes/num_gemms):"
    )
    b = ok[(ok.distribution == "balanced") & (ok.proj == "fc1")]
    for tpe in sorted(b.tokens_per_expert.unique()):
        sub = b[b.tokens_per_expert == tpe]
        print(
            f"  tokens/expert={tpe:>5}: util median={sub.fwd_util_pct.median():5.1f}%  "
            f"(n={len(sub)}, tflops median={sub.fwd_tflops.median():.0f})"
        )

    print("\nkernels observed:")
    for nm, cnt in ok["kernel_name"].value_counts().head(8).items():
        print(f"  {cnt:4d}  {nm}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark te.pytorch.GroupedLinear (MoE expert grouped GEMM)"
    )
    parser.add_argument(
        "--device",
        choices=["a100", "h100", "h200", "b200"],
        default=None,
        help="Device label; autodetect if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gemm_grouped_timings.parquet",
        help="Output parquet path (also writes .csv next to it).",
    )
    parser.add_argument("--num-gemms", type=int, nargs="+", default=DEFAULT_NUM_GEMMS)
    parser.add_argument(
        "--tokens-per-expert", type=int, nargs="+", default=DEFAULT_TOKENS_PER_EXPERT
    )
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="+",
        default=None,
        help="hidden:ffn pairs (e.g. 4096:2560 6144:3840); default DEFAULT_SHAPES.",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=DEFAULT_DTYPES,
        choices=["bf16", "fp16", "fp32", "fp8"],
    )
    parser.add_argument(
        "--distributions",
        type=str,
        nargs="+",
        default=["balanced", "skewed"],
        choices=["balanced", "skewed"],
    )
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument(
        "--only-target",
        action="store_true",
        help="Run only the sonic config point (num_gemms=16, 192 tok/exp, 2048/1280).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available on this host.", file=sys.stderr)
        return 2

    try:
        _, te_version = _import_te()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    device_label = args.device or autodetect_device()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Device label: {device_label}")
    print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}, TE: {te_version}")
    print(f"Host: {platform.node()}")

    grid = build_grid(args)
    print(f"Grid size: {len(grid)}")

    rows: list[BenchRow] = []
    for i, cfg in enumerate(grid, 1):
        print(
            f"[{i}/{len(grid)}] gemms={cfg['num_gemms']:>3} tok/exp={cfg['tokens_per_expert']:>5} "
            f"h={cfg['hidden']:>5} ffn={cfg['ffn']:>5} {cfg['dtype']} {cfg['proj']} "
            f"{cfg['distribution']}",
            end=" ... ",
            flush=True,
        )
        try:
            row = benchmark_point(
                device_label=device_label,
                gpu_name=gpu_name,
                n_warmup=args.n_warmup,
                n_iter=args.n_iter,
                te_version=te_version,
                **cfg,
            )
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {e}")
            continue
        if row is None:
            print("skipped")
            continue
        rows.append(row)
        if row.status == "OK":
            print(f"fwd={row.fwd_ms:.4f}ms util={row.fwd_util_pct:.1f}% bwd={row.bwd_ms:.4f}ms")
        else:
            print(row.status)

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
