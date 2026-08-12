"""Benchmark the LM-head vocab GEMM (fwd + autograd bwd) and cross-entropy loss.

Motivation:
    The "Embedding + LM Head/CE" stage term (dlcalc/utils/pipeline.py::
    compute_stage_imbalance_extra_time_s, wired in training_3d.py) prices the
    last pipeline stage's per-microbatch surplus at ~59 ms/mb on the 18b
    config, while the rank0 traces imply ~87 ms/mb of last-stage pacing
    (bc59887c: ~24 s of steady-state PP-recv wait over 256 microbatches).
    Two of its inputs are currently estimates:
      1. GEMM util at the vocab shape: gemm_util.parquet has no K=6144 row
         (power-of-2 grid) -- the lookup interpolates to 0.61 at
         (M=8192, N=257536, K=6144). The N dimension is also far off-grid.
      2. CE/softmax cost: modeled as 4 fp32 HBM passes over the logits.
         Real Megatron fuses the vocab-parallel CE; the pass count is a guess.
    This script MEASURES both so the term's inputs come from data
    (GUIDELINES §1). Single GPU, no distributed setup needed.

What it measures (bf16 GEMM, fp32 logits for CE, like Megatron):
    A) lm_head = Linear(hidden -> vocab, bias=False):
         fwd:      y = x @ W.T
         fwd+bwd:  autograd (dX = dY @ W, dW = dY.T @ X)
       at M in {4096, 8192, 16384, 32768} tokens (mbs x seq combinations),
       hidden 6144 (18b), 4096 (5p3b), 2048 (700m); vocab 257540 (padded 257664).
    B) F.cross_entropy on fp32 logits (M, vocab), fwd and fwd+bwd, including
       the .float() upcast of bf16 logits (what Megatron's CE does).

Usage on a sleeper pod (single GPU):
    python benchmarks/vocab_gemm_ce_benchmark.py \
        --output benchmarks/results/vocab_gemm_ce_b200.parquet

    # quick smoke:
    python benchmarks/vocab_gemm_ce_benchmark.py --smoke
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import asdict, dataclass

import pandas as pd
import torch

PEAK_TFLOPS_BF16 = {"a100": 312.0, "h100": 989.4, "h200": 989.4, "b200": 2250.0}
HBM_GBPS = {"a100": 2039, "h100": 3350, "h200": 4800, "b200": 8000}

# (model tag, hidden). vocab is shared across the ladder.
HIDDEN_SIZES = [("18b", 6144), ("5p3b", 4096), ("700m", 2048)]
VOCAB = 257540
VOCAB_PADDED = 257664  # padded to 128 (Megatron make_vocab_size_divisible_by)

DEFAULT_M = [4096, 8192, 16384, 32768]
SMOKE_M = [8192]


@dataclass
class BenchRow:
    device: str
    gpu_name: str
    op: str  # "lm_head_gemm" | "cross_entropy"
    model_tag: str
    m_tokens: int
    hidden: int
    vocab_padded: int
    dtype: str
    n_iter: int
    fwd_ms_p50: float
    fwd_bwd_ms_p50: float
    bwd_ms_p50: float  # fwd_bwd - fwd
    fwd_tflops: float
    bwd_tflops: float
    fwd_util_pct: float
    bwd_util_pct: float
    # CE only: effective HBM passes over the fp32 logits implied by the time
    # (bytes_moved / logits_bytes), for calibrating the model's pass count.
    ce_implied_hbm_passes_fwd: float
    ce_implied_hbm_passes_fwdbwd: float
    status: str


def _detect_device_label() -> str:
    name = torch.cuda.get_device_name(0).lower()
    for tag in ("b200", "h200", "h100", "a100"):
        if tag in name:
            return tag
    return name.replace(" ", "_")


def _time_ms(fn, n_warmup: int, n_iter: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def bench_lm_head(
    device_label: str,
    gpu_name: str,
    model_tag: str,
    m: int,
    hidden: int,
    n_warmup: int,
    n_iter: int,
) -> BenchRow:
    peak = PEAK_TFLOPS_BF16.get(device_label, 0.0)
    try:
        w = torch.randn(VOCAB_PADDED, hidden, dtype=torch.bfloat16, device="cuda")
        w.requires_grad_(True)
        x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        def fwd():
            return x @ w.t()

        fwd_ms = _time_ms(lambda: fwd(), n_warmup, n_iter)

        def fwd_bwd():
            y = x @ w.t()
            y.backward(torch.ones_like(y))
            x.grad = None
            w.grad = None

        fwd_bwd_ms = _time_ms(fwd_bwd, n_warmup, n_iter)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        return BenchRow(
            device=device_label,
            gpu_name=gpu_name,
            op="lm_head_gemm",
            model_tag=model_tag,
            m_tokens=m,
            hidden=hidden,
            vocab_padded=VOCAB_PADDED,
            dtype="bf16",
            n_iter=n_iter,
            fwd_ms_p50=float("nan"),
            fwd_bwd_ms_p50=float("nan"),
            bwd_ms_p50=float("nan"),
            fwd_tflops=0.0,
            bwd_tflops=0.0,
            fwd_util_pct=0.0,
            bwd_util_pct=0.0,
            ce_implied_hbm_passes_fwd=float("nan"),
            ce_implied_hbm_passes_fwdbwd=float("nan"),
            status=f"FAILED: {type(e).__name__}: {str(e)[:80]}",
        )
    finally:
        torch.cuda.empty_cache()

    fwd_flops = 2.0 * m * hidden * VOCAB_PADDED
    bwd_flops = 2.0 * fwd_flops  # dX + dW
    bwd_ms = max(fwd_bwd_ms - fwd_ms, 1e-9)
    fwd_tf = fwd_flops / (fwd_ms / 1e3) / 1e12
    bwd_tf = bwd_flops / (bwd_ms / 1e3) / 1e12
    return BenchRow(
        device=device_label,
        gpu_name=gpu_name,
        op="lm_head_gemm",
        model_tag=model_tag,
        m_tokens=m,
        hidden=hidden,
        vocab_padded=VOCAB_PADDED,
        dtype="bf16",
        n_iter=n_iter,
        fwd_ms_p50=fwd_ms,
        fwd_bwd_ms_p50=fwd_bwd_ms,
        bwd_ms_p50=bwd_ms,
        fwd_tflops=fwd_tf,
        bwd_tflops=bwd_tf,
        fwd_util_pct=100.0 * fwd_tf / peak if peak else 0.0,
        bwd_util_pct=100.0 * bwd_tf / peak if peak else 0.0,
        ce_implied_hbm_passes_fwd=float("nan"),
        ce_implied_hbm_passes_fwdbwd=float("nan"),
        status="OK",
    )


def bench_cross_entropy(
    device_label: str,
    gpu_name: str,
    model_tag: str,
    m: int,
    hidden: int,
    n_warmup: int,
    n_iter: int,
) -> BenchRow:
    hbm = HBM_GBPS.get(device_label, 0)
    try:
        # Megatron path: bf16 logits from the GEMM, upcast to fp32 inside CE.
        logits_bf16 = torch.randn(
            m, VOCAB_PADDED, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        targets = torch.randint(0, VOCAB, (m,), device="cuda")

        def fwd():
            return torch.nn.functional.cross_entropy(logits_bf16.float(), targets)

        fwd_ms = _time_ms(lambda: fwd(), n_warmup, n_iter)

        def fwd_bwd():
            loss = torch.nn.functional.cross_entropy(logits_bf16.float(), targets)
            loss.backward()
            logits_bf16.grad = None

        fwd_bwd_ms = _time_ms(fwd_bwd, n_warmup, n_iter)
        status = "OK"
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        return BenchRow(
            device=device_label,
            gpu_name=gpu_name,
            op="cross_entropy",
            model_tag=model_tag,
            m_tokens=m,
            hidden=hidden,
            vocab_padded=VOCAB_PADDED,
            dtype="fp32",
            n_iter=n_iter,
            fwd_ms_p50=float("nan"),
            fwd_bwd_ms_p50=float("nan"),
            bwd_ms_p50=float("nan"),
            fwd_tflops=0.0,
            bwd_tflops=0.0,
            fwd_util_pct=0.0,
            bwd_util_pct=0.0,
            ce_implied_hbm_passes_fwd=float("nan"),
            ce_implied_hbm_passes_fwdbwd=float("nan"),
            status=f"FAILED: {type(e).__name__}: {str(e)[:80]}",
        )
    finally:
        torch.cuda.empty_cache()

    logits_fp32_bytes = m * VOCAB_PADDED * 4
    bytes_per_pass_s = hbm * 1e9
    implied_fwd = (fwd_ms / 1e3) * bytes_per_pass_s / logits_fp32_bytes if hbm else float("nan")
    implied_fwdbwd = (
        (fwd_bwd_ms / 1e3) * bytes_per_pass_s / logits_fp32_bytes if hbm else float("nan")
    )
    return BenchRow(
        device=device_label,
        gpu_name=gpu_name,
        op="cross_entropy",
        model_tag=model_tag,
        m_tokens=m,
        hidden=hidden,
        vocab_padded=VOCAB_PADDED,
        dtype="fp32",
        n_iter=n_iter,
        fwd_ms_p50=fwd_ms,
        fwd_bwd_ms_p50=fwd_bwd_ms,
        bwd_ms_p50=max(fwd_bwd_ms - fwd_ms, 0.0),
        fwd_tflops=0.0,
        bwd_tflops=0.0,
        fwd_util_pct=0.0,
        bwd_util_pct=0.0,
        ce_implied_hbm_passes_fwd=implied_fwd,
        ce_implied_hbm_passes_fwdbwd=implied_fwdbwd,
        status=status,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/vocab_gemm_ce_timings.parquet")
    parser.add_argument("--m-tokens", type=int, nargs="*", default=None)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device_label = _detect_device_label()
    gpu_name = torch.cuda.get_device_name(0)
    ms_list = args.m_tokens or (SMOKE_M if args.smoke else DEFAULT_M)
    n_iter = 3 if args.smoke else args.n_iter

    print(f"device={device_label} ({gpu_name}), M sweep {ms_list}")
    rows: list[BenchRow] = []
    for model_tag, hidden in HIDDEN_SIZES:
        for m in ms_list:
            r = bench_lm_head(device_label, gpu_name, model_tag, m, hidden, args.n_warmup, n_iter)
            rows.append(r)
            print(
                f"  gemm {model_tag} h={hidden} M={m}: fwd {r.fwd_ms_p50:7.2f} ms "
                f"({r.fwd_util_pct:4.1f}%)  bwd {r.bwd_ms_p50:7.2f} ms "
                f"({r.bwd_util_pct:4.1f}%)  {r.status}"
            )
            r = bench_cross_entropy(
                device_label, gpu_name, model_tag, m, hidden, args.n_warmup, n_iter
            )
            rows.append(r)
            print(
                f"  ce   {model_tag} h={hidden} M={m}: fwd {r.fwd_ms_p50:7.2f} ms "
                f"(≈{r.ce_implied_hbm_passes_fwd:4.1f} passes)  "
                f"fwd+bwd {r.fwd_bwd_ms_p50:7.2f} ms "
                f"(≈{r.ce_implied_hbm_passes_fwdbwd:4.1f} passes)  {r.status}"
            )

    df = pd.DataFrame([asdict(r) for r in rows])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    df.to_csv(os.path.splitext(args.output)[0] + ".csv", index=False)
    print(f"\nwrote {args.output}  ({len(df)} rows)")

    ok = df[(df.op == "lm_head_gemm") & (df.status == "OK")]
    if len(ok):
        print("\nGEMM util summary (for gemm_util comparison at the vocab shape):")
        for _, r in ok.iterrows():
            print(
                f"  h={r.hidden} M={r.m_tokens}: fwd util {r.fwd_util_pct:.1f}% "
                f"(model currently interpolates 61%)"
            )
    okce = df[(df.op == "cross_entropy") & (df.status == "OK")]
    if len(okce):
        med = okce.ce_implied_hbm_passes_fwdbwd.median()
        print(f"\nCE fwd+bwd implied HBM passes (median): {med:.1f} (model currently charges 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
