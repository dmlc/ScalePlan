"""Regression test for the expert-GEMM grouped-utilization model.

The expert MLP is a grouped GEMM (TE GroupedLinear; `nvjet_tst_*` on B200). Its
per-rank FLOPs are the aggregate over the rank's ``n_local = n_experts/EP`` groups,
but its EFFICIENCY is set by the per-group tile M (= dropless capacity =
seq*top_k/n_experts, which is EP-INVARIANT). The model therefore looks utilization
up at the per-group M and keeps FLOPs at the aggregate.

History:
  * The ORIGINAL code multiplied a per-group time by n_local -> compute ∝ n_local
    ∝ 1/EP (the "Effect A" over-credit).
  * The first fix ("model A1") looked util up at the AGGREGATE M_eff to make it
    flat in EP — but that OVER-credited the grouped kernel's utilization 5-11x in
    the dropless regime (a single M_eff GEMM hits ~29% util; the real 16x-grouped
    kernel at cap=192 hits ~4.5%, measured on B200 — see
    benchmarks/results/gemm_grouped_b200.parquet).
  * Current model: util at the per-group M (capacity). This makes the PER-RANK
    expert time scale ~linearly with n_local (util ~const at fixed cap), matching
    the measured grouped kernel — so per-rank block time GROWS as EP falls. That
    is the property pinned below (replacing the earlier wrong "flat in EP").
"""

from __future__ import annotations

import copy
import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout

import pytest
import yaml

# A clean single-node-ish MoE config (700M-class) with world size = tp*pp*cp*dp
# held FIXED while EP varies. pp=1 so there is no pipeline bubble to confound the
# block-time reading; the expert region is what changes with EP.
_BASE_CONFIG: dict = {
    "model": {
        "n_layers": 16,
        "hidden_sz": 2048,
        "inter_sz": 5120,
        "n_q_heads": 16,
        "n_kv_heads": 8,
        "head_dim": 128,
        "vocab_sz": 257152,
        "precision": "bf16",
        "sdpa_precision": "bf16",
        "glu": True,
        "rotary_embeds": True,
        "dropout": False,
        "tie_embeddings": False,
        "moe": {
            "n_experts": 128,
            "experts_per_token": 3,
            "capacity_factor": 1,
            "expert_inter_sz": 1280,
            "moe_frequency": 1.0,
            "expert_tp_degree": 1,
        },
    },
    "parallelism": {
        "tp": 1,
        "ep": 8,  # overridden per case
        "pp": 1,
        "cp": 1,
        "dp": 32,  # fixed world = tp*pp*cp*dp = 32 across all EP
        "vpp": 1,
        "sp": True,
        "zero_level": 1,
        "n_param_buckets": 8,
    },
    "performance": {"activation_checkpointing_type": "none"},
    "optimizer": {"optimizer_type": "muon"},
    "data": {"gbs": 256, "seqlen": 8192, "microbatch_sz": 1},
    "hardware": {"node_type": "p6-b200.48xlarge"},
}

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _transformer_block_ms(cfg: dict) -> float:
    """Run 3dtrn in-process and return the per-iteration Transformer Block time (ms)."""
    import dlcalc.training_3d as training_3d

    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    cfg_path = fh.name
    buf = io.StringIO()
    saved_argv = sys.argv
    sys.argv = ["3dtrn", cfg_path]
    try:
        yaml.safe_dump(cfg, fh, sort_keys=False)
        fh.close()
        with redirect_stdout(buf):
            training_3d.main()
    finally:
        sys.argv = saved_argv
        fh.close()
        os.remove(cfg_path)

    clean = _ANSI.sub("", buf.getvalue())
    match = re.search(r"Transformer Block\s+([0-9]+\.[0-9]+)\s*ms", clean)
    if match is None:
        raise AssertionError(
            "Could not find 'Transformer Block' in 3dtrn output. Tail:\n" + clean[-800:]
        )
    return float(match.group(1))


@pytest.mark.slow
def test_expert_gemm_per_group_util_scales_with_n_local() -> None:
    """Per-rank transformer-block compute GROWS as EP decreases (n_local rises).

    Physics (measurement-validated on B200, benchmarks/gemm_grouped_benchmark.py):
    the expert MLP is a grouped GEMM whose EFFICIENCY is set by the per-group tile
    M (= dropless capacity = seq*top_k/n_experts, EP-INVARIANT), while its per-rank
    FLOPs are the aggregate over the rank's n_local = n_experts/EP groups. So at
    fixed capacity the per-rank expert time scales ~linearly with n_local: lowering
    EP (more experts per rank) raises the per-rank block time. The measured grouped
    kernel confirms this (fwd_ms ∝ num_gemms at fixed cap; util ~const ~3-6%).

    This REPLACES the earlier (wrong) "flat in EP" invariant, which came from
    looking util up at the aggregate M_eff — that over-credited util 5-11x in the
    dropless regime and wrongly predicted a per-rank time independent of n_local.
    (Global expert work is ~EP-invariant — 1/EP per rank x EP ranks — but the model
    times the per-rank grouped kernel, which is what the trace measures. The old
    "flat" reading conflated the attention-dominated total GEMM bucket with the
    expert part.)
    """
    times = {}
    for ep in (1, 2, 4, 8):
        cfg = copy.deepcopy(_BASE_CONFIG)
        cfg["parallelism"]["ep"] = ep
        times[ep] = _transformer_block_ms(cfg)

    # Monotonic: fewer experts/rank (higher EP) => less per-rank expert work
    # (more local experts = more grouped-GEMM sub-kernels = more compute AND more
    # dispatch). The Transformer Block term is now max(compute, dispatch) per
    # microbatch, so at high n_local the CPU-dispatch floor also grows with
    # n_local — both mechanisms push EP1 above EP8.
    assert times[1] > times[2] > times[4] > times[8], (
        f"per-rank block time must decrease with EP (n_local shrinks): {times}"
    )
    # Unmistakably NOT flat (the old M_eff-lookup bug gave <1.10x spread). The
    # spread is smaller than the expert-compute term alone would give because
    # (a) attention/norm/comm are EP-invariant and (b) small configs are
    # dispatch-bound so the max() floor compresses the compute difference — but
    # the measured grouped-GEMM + dispatch model still separates the regimes.
    spread = times[1] / times[8]
    assert spread > 1.2, (
        f"per-rank block time barely moved with EP (spread {spread:.3f}x): {times}. "
        f"Expert-GEMM time must come from the measured grouped-kernel table at the "
        f"per-group M, not a single-GEMM proxy at aggregate M_eff."
    )
