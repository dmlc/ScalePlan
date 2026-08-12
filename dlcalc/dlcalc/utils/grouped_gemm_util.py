"""Measured grouped-GEMM (MoE expert MLP) timing lookup.

The expert MLP is a GROUPED GEMM: `num_gemms = n_local_experts` independent
matmuls run by one fused TE `GroupedLinear` kernel (`nvjet_tst_*` on B200).
Its efficiency is set by the PER-GROUP tile M (= tokens/expert), not the
aggregate M_eff, and it does not match any single-GEMM cell of gemm_util.parquet
(rounding M/N/K to powers of two lands on the wrong efficiency). This module
reads the DIRECTLY measured grouped-kernel timings
(benchmarks/results/gemm_grouped_b200.parquet, produced by
benchmarks/gemm_grouped_benchmark.py on B200) and returns fwd / bwd seconds for
a given (n_local_experts, tokens_per_expert, hidden, ffn) point.

Keyed by proj ("fc1" = hidden->2*ffn SwiGLU gate+up; "fc2" = ffn->hidden). The
caller sums fc1+fc2 for a full expert-MLP fwd (or bwd). Interpolation:
  * tokens_per_expert: log-linear within the measured grid (64..4096), clamped
    outside.
  * num_gemms / (hidden, ffn): exact match required — the golden set's
    (hidden, ffn) are all measured, and num_gemms is a small measured set
    {4,8,16,32,64,128} = n_experts / EP for n_experts=128 and every golden EP.
    A miss raises (GUIDELINES §5) rather than silently extrapolating.
Only bf16 is measured; other dtypes raise.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import pandas as pd  # type: ignore[import-untyped]

from dlcalc.utils.hardware import DType

_PARQUET = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "benchmarks",
    "results",
    "gemm_grouped_b200.parquet",
)

_MEASURED_DEVICE_PREFIX = "p6-b200"


def grouped_gemm_measured(machine_name: str, dtype: DType) -> bool:
    """True if the measured grouped-GEMM table applies to this machine+dtype."""
    return (
        machine_name.startswith(_MEASURED_DEVICE_PREFIX)
        and dtype == DType.BF16
        and os.path.exists(_PARQUET)
    )


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:  # type: ignore[no-any-unimported]
    df = pd.read_parquet(_PARQUET)
    return df[(df["status"] == "OK") & (df["distribution"] == "balanced")]


def _interp_log(curve: list[tuple[float, float]], x: float) -> float:
    pts = sorted(curve)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)
    raise AssertionError("unreachable")


def _lookup_ms(
    proj: str, num_gemms: int, tokens_per_expert: int, hidden: int, ffn: int, col: str
) -> float:
    df = _load()
    sub = df[
        (df["proj"] == proj)
        & (df["num_gemms"] == num_gemms)
        & (df["hidden"] == hidden)
        & (df["ffn"] == ffn)
    ]
    if sub.empty:
        raise KeyError(
            f"grouped GEMM: no measured row for proj={proj} num_gemms={num_gemms} "
            f"hidden={hidden} ffn={ffn}. Measured shapes: "
            f"{sorted(set(zip(df['hidden'], df['ffn'])))}, num_gemms "
            f"{sorted(df['num_gemms'].unique())}. Re-run benchmarks/gemm_grouped_benchmark.py."
        )
    curve = list(zip(sub["tokens_per_expert"].astype(float), sub[col].astype(float)))
    return _interp_log(curve, float(tokens_per_expert))


def grouped_mlp_fwd_time_s(
    n_local_experts: int, tokens_per_expert: int, hidden: int, ffn: int
) -> float:
    """Measured forward time (s) of the full expert MLP (fc1 + fc2 grouped GEMMs)."""
    fc1 = _lookup_ms("fc1", n_local_experts, tokens_per_expert, hidden, ffn, "fwd_ms")
    fc2 = _lookup_ms("fc2", n_local_experts, tokens_per_expert, hidden, ffn, "fwd_ms")
    return (fc1 + fc2) / 1e3


def grouped_mlp_bwd_time_s(
    n_local_experts: int, tokens_per_expert: int, hidden: int, ffn: int
) -> float:
    """Measured backward time (s) of the full expert MLP (fc1 + fc2 grouped GEMMs)."""
    fc1 = _lookup_ms("fc1", n_local_experts, tokens_per_expert, hidden, ffn, "bwd_ms")
    fc2 = _lookup_ms("fc2", n_local_experts, tokens_per_expert, hidden, ffn, "bwd_ms")
    return (fc1 + fc2) / 1e3


def grouped_mlp_up_fwd_time_s(
    n_local_experts: int, tokens_per_expert: int, hidden: int, ffn: int
) -> float:
    """Measured forward time (s) of ONLY fc1 (up-proj, hidden->2*ffn SwiGLU)."""
    return _lookup_ms("fc1", n_local_experts, tokens_per_expert, hidden, ffn, "fwd_ms") / 1e3


def grouped_mlp_down_fwd_time_s(
    n_local_experts: int, tokens_per_expert: int, hidden: int, ffn: int
) -> float:
    """Measured forward time (s) of ONLY fc2 (down-proj, ffn->hidden)."""
    return _lookup_ms("fc2", n_local_experts, tokens_per_expert, hidden, ffn, "fwd_ms") / 1e3


def grouped_mlp_up_bwd_time_s(
    n_local_experts: int, tokens_per_expert: int, hidden: int, ffn: int
) -> float:
    """Measured backward time (s) of ONLY fc1 (up-proj)."""
    return _lookup_ms("fc1", n_local_experts, tokens_per_expert, hidden, ffn, "bwd_ms") / 1e3


def grouped_mlp_down_bwd_time_s(
    n_local_experts: int, tokens_per_expert: int, hidden: int, ffn: int
) -> float:
    """Measured backward time (s) of ONLY fc2 (down-proj)."""
    return _lookup_ms("fc2", n_local_experts, tokens_per_expert, hidden, ffn, "bwd_ms") / 1e3
