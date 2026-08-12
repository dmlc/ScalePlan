"""Empirical SDPA (fwd and bwd) timing lookup.

Backs `compute_sdpa_bwd_time_s` with measured data collected on A100/H200/B200
sleeper pods via `benchmarks/sdpa_benchmark.py`. Avoids using theoretical FLOP
counts for the backward pass, which mis-predict by ~30-40% on modern flash
attention kernels (the "2.5x fwd" rule-of-thumb the literature quotes comes
from an old softmax-materializing implementation).

Data file: `benchmarks/results/sdpa_fwd_bwd_timings.parquet`. Columns:
  device, dtype, seq_len, micro_bs, n_q_heads, n_kv_heads, head_dim, backend,
  fwd_time_ms_med, bwd_time_ms_med, status

If the exact shape is missing we fall back to the closest matching row on the
same (device, dtype, backend) by Euclidean distance in normalized shape space -
same pattern as `norm_rope_util.py`.
"""

import os
from functools import lru_cache
from typing import Optional

import pandas as pd

from dlcalc.utils.hardware import DType, MachineSpec


MACHINE_SPEC_TO_GPU_MODEL = {
    "p4d.24xlarge": "a100",
    "p4de.24xlarge": "a100",
    # H100 and H200 are both Hopper with nearly identical SDPA kernel behavior;
    # we only have H200 measurements so H100 maps to h200 too.
    "p5.48xlarge": "h200",
    "p5e.48xlarge": "h200",
    "p5en.48xlarge": "h200",
    "p6-b200.48xlarge": "b200",
    # No AWS-native Trainium SDPA timings yet; A100 is the closest proxy.
    "trn1n.32xlarge": "a100",
}

DTYPE_TO_STR = {
    DType.FP16: "fp16",
    DType.BF16: "bf16",
    DType.FP8: "fp8",       # no FP8 SDPA measurements yet; fall back to bf16
    DType.FP8_E4M3: "fp8",  # SDPA typically stays in bf16 even in FP8 configs
    DType.FP32: "bf16",     # SDPA on modern stacks rarely runs pure FP32
    DType.TF32: "bf16"
}


@lru_cache(maxsize=1)
def _load_sdpa_timings() -> Optional[pd.DataFrame]:
    """Load measured SDPA timings, or None if the file is missing."""
    here = os.path.dirname(os.path.abspath(__file__))
    # Repo layout: .../dlcalc/dlcalc/utils/sdpa_util.py -> ../../.. is the worktree root.
    for candidate in [
        os.path.abspath(
            os.path.join(
                here, "..", "..", "..",
                "benchmarks", "results", "sdpa_fwd_bwd_timings.parquet",
            )
        ),
        os.path.join("benchmarks", "results", "sdpa_fwd_bwd_timings.parquet"),
    ]:
        if os.path.exists(candidate):
            try:
                return pd.read_parquet(candidate)
            except Exception:
                return None
    return None


def _lookup_closest(
    df: pd.DataFrame,
    device: str,
    dtype: str,
    seq_len: int,
    micro_bs: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    backend: str = "te",
) -> Optional[pd.Series]:
    """Find the closest (device, dtype, backend) row by shape distance."""
    filtered = df[
        (df["device"] == device)
        & (df["dtype"] == dtype)
        & (df["backend"] == backend)
        & (df["status"] == "OK")
    ]
    if len(filtered) == 0:
        # Relax: drop backend constraint.
        filtered = df[
            (df["device"] == device)
            & (df["dtype"] == dtype)
            & (df["status"] == "OK")
        ]
    if len(filtered) == 0:
        # Relax: drop dtype constraint.
        filtered = df[(df["device"] == device) & (df["status"] == "OK")]
    if len(filtered) == 0:
        return None

    filtered = filtered.copy()
    filtered["distance"] = (
        ((filtered["seq_len"] - seq_len) / max(seq_len, 2048)) ** 2
        + ((filtered["micro_bs"] - micro_bs) / max(micro_bs, 1)) ** 2
        + ((filtered["n_q_heads"] - n_q_heads) / max(n_q_heads, 8)) ** 2
        + ((filtered["n_kv_heads"] - n_kv_heads) / max(n_kv_heads, 1)) ** 2
        + ((filtered["head_dim"] - head_dim) / max(head_dim, 64)) ** 2
    ) ** 0.5
    return filtered.loc[filtered["distance"].idxmin()]


def get_sdpa_bwd_time_s(
    seq_len: int,
    micro_bs: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> Optional[float]:
    """Look up measured SDPA backward time in seconds.

    Returns None if no parquet data is available at all. If the exact shape is
    missing, falls back to the closest match for the same device/dtype/backend.
    """
    df = _load_sdpa_timings()
    if df is None:
        return None

    device = MACHINE_SPEC_TO_GPU_MODEL.get(machine_spec.name)
    if device is None:
        return None
    dtype_str = DTYPE_TO_STR.get(dtype)
    if dtype_str is None:
        return None

    match = _lookup_closest(
        df,
        device=device,
        dtype=dtype_str,
        seq_len=seq_len,
        micro_bs=micro_bs,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        backend="te",
    )
    if match is None or pd.isna(match["bwd_time_ms_med"]):
        return None
    return match["bwd_time_ms_med"] / 1000.0


def get_sdpa_fwd_time_s(
    seq_len: int,
    micro_bs: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> Optional[float]:
    """Look up measured SDPA forward time in seconds (same table as bwd)."""
    df = _load_sdpa_timings()
    if df is None:
        return None

    device = MACHINE_SPEC_TO_GPU_MODEL.get(machine_spec.name)
    if device is None:
        return None
    dtype_str = DTYPE_TO_STR.get(dtype)
    if dtype_str is None:
        return None

    match = _lookup_closest(
        df,
        device=device,
        dtype=dtype_str,
        seq_len=seq_len,
        micro_bs=micro_bs,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        backend="te",
    )
    if match is None or pd.isna(match["fwd_time_ms_med"]):
        return None
    return match["fwd_time_ms_med"] / 1000.0
