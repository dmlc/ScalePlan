"""Measured cross-entropy / LM-head vocab-GEMM timing (B200, 2026-07-17).

The LM head + cross-entropy on the last pipeline stage is priced by the
stage-imbalance term (utils/pipeline). Two of its inputs were estimates the
measured data corrects:

  * cross-entropy: modeled as 4 fp32 HBM passes over the (tokens x vocab)
    logits, but Megatron's fused vocab-parallel CE (softmax + logsumexp +
    grad-of-logits, all fp32) is ~19.6 effective passes — the naive count was
    ~5x too low (benchmarks/vocab_gemm_ce_benchmark.py ->
    results/vocab_gemm_ce_b200.parquet).
  * LM-head GEMM util: gemm_util.parquet has no K=6144 / N~257k rows (rounds to
    powers of two), interpolating ~0.61; measured fwd util is 0.65-0.72, bwd
    0.72-0.83.

This module returns the measured cross-entropy fwd+bwd seconds for a given
(m_tokens, vocab_padded) on B200; the LM-head GEMM itself stays on the existing
compute_gemm_time_s / _linear_bwd path (its util is close enough post-rounding,
and vocab N is off-grid the same way for all models). CE is the ~5x error.

Interpolation: log-linear in m_tokens within the measured grid; CE cost is
~linear in the logits volume (m_tokens x vocab), so we scale by vocab too.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import pandas as pd  # type: ignore[import-untyped]

_PARQUET = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "benchmarks",
    "results",
    "vocab_gemm_ce_b200.parquet",
)
_MEASURED_DEVICE_PREFIX = "p6-b200"


def vocab_ce_measured(machine_name: str) -> bool:
    """True if the measured CE table applies to this machine."""
    return machine_name.startswith(_MEASURED_DEVICE_PREFIX) and os.path.exists(_PARQUET)


@lru_cache(maxsize=1)
def _load_ce() -> pd.DataFrame:  # type: ignore[no-any-unimported]
    df = pd.read_parquet(_PARQUET)
    return df[(df["op"] == "cross_entropy") & (df["status"] == "OK")]


def cross_entropy_fwd_bwd_time_s(m_tokens: int, vocab_padded: int) -> float:
    """Measured cross-entropy fwd+bwd time (s) over (m_tokens x vocab) fp32 logits.

    Interpolates the measured m_tokens grid (log-linear) and scales linearly by
    the vocab dimension (CE is HBM-bandwidth-bound in the logits volume, which is
    proportional to m_tokens*vocab). The measured rows share vocab_padded, so the
    vocab scale is 1.0 for the golden set; kept explicit for other vocabs.
    """
    df = _load_ce()
    meas_vocab = int(df["vocab_padded"].iloc[0])
    pts = sorted(zip(df["m_tokens"].astype(float), df["fwd_bwd_ms_p50"].astype(float)))
    x = float(m_tokens)
    if x <= pts[0][0]:
        ms = pts[0][1] * x / pts[0][0]  # CE ~linear in tokens below the grid
    elif x >= pts[-1][0]:
        ms = pts[-1][1] * x / pts[-1][0]
    else:
        ms = None
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
                ms = y0 + f * (y1 - y0)
                break
        assert ms is not None
    return float(ms / 1e3) * (vocab_padded / meas_vocab)
