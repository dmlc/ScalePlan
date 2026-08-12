"""Utilities for looking up LayerNorm and RoPE timing from measured data.

This module replaces rough HBM load/store approximations with empirical kernel
timing lookups, similar to gemm_util.py and sdpa.parquet lookups.
"""

import os
from functools import lru_cache
from typing import Optional

import pandas as pd

from dlcalc.utils.hardware import DType, MachineSpec


# Mapping from machine spec names to GPU model names in the parquet file
MACHINE_SPEC_TO_GPU_MODEL = {
    "p5.48xlarge": "h100",
    "p5e.48xlarge": "h100",
    "p5en.48xlarge": "h100",
    "p6-b200.48xlarge": "b200",
}

# Mapping from DType enum to dtype strings in parquet
DTYPE_TO_STR = {
    DType.FP16: "fp16",
    DType.BF16: "bf16",
    DType.FP8: "fp8",
    DType.FP8_E4M3: "fp8",
}

# Default fallback values (in seconds)
# Based on typical HBM bandwidth characteristics
DEFAULT_NORM_TIME_S = 0.0001  # 0.1ms
DEFAULT_ROPE_TIME_S = 0.0001  # 0.1ms


@lru_cache(maxsize=1)
def _load_norm_rope_timings() -> Optional[pd.DataFrame]:
    """Load timing data from parquet file with caching.

    Returns:
        DataFrame with timing data, or None if file not found
    """
    # Try to find the parquet file in the package directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    parquet_path = os.path.join(pkg_root, "norm_rope_timings.parquet")

    if not os.path.exists(parquet_path):
        # Try relative to current working directory
        parquet_path = "norm_rope_timings.parquet"
        if not os.path.exists(parquet_path):
            return None

    try:
        df = pd.read_parquet(parquet_path)
        return df
    except Exception:
        return None


def _find_closest_match(
    df: pd.DataFrame,
    batch: int,
    seqlen: int,
    hidden_dim: int,
    dtype: str,
    device: str,
) -> Optional[pd.Series]:
    """Find closest matching timing entry in the dataframe.

    Uses exact match on device and dtype, then finds closest match on
    batch, seqlen, and hidden_dim using Euclidean distance.

    Args:
        df: DataFrame with timing data
        batch: Batch size
        seqlen: Sequence length
        hidden_dim: Hidden dimension
        dtype: Data type string
        device: Device name

    Returns:
        Series with timing data, or None if no match found
    """
    # Filter by device and dtype
    filtered = df[(df["device"] == device) & (df["dtype"] == dtype)]

    if len(filtered) == 0:
        # Try without device filter
        filtered = df[df["dtype"] == dtype]

    if len(filtered) == 0:
        return None

    # Find closest match by computing distance in parameter space
    # Normalize parameters to similar scales for distance calculation
    filtered = filtered.copy()
    filtered["distance"] = (
        ((filtered["batch"] - batch) / max(batch, 1)) ** 2
        + ((filtered["seqlen"] - seqlen) / max(seqlen, 1024)) ** 2
        + ((filtered["hidden_dim"] - hidden_dim) / max(hidden_dim, 4096)) ** 2
    ) ** 0.5

    # Return the closest match
    closest_idx = filtered["distance"].idxmin()
    return filtered.loc[closest_idx]


def get_norm_time(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    dtype: DType,
    machine_spec: MachineSpec,
) -> float:
    """Get LayerNorm timing from lookup table.

    Falls back to HBM bandwidth estimation if lookup fails.

    Args:
        batch: Batch size
        seqlen: Sequence length
        hidden_dim: Hidden dimension
        dtype: Data type
        machine_spec: Machine specification

    Returns:
        LayerNorm time in seconds
    """
    df = _load_norm_rope_timings()

    if df is None:
        # Fallback: Use HBM bandwidth estimation
        data_size_bytes = batch * seqlen * hidden_dim * dtype.size_bytes()
        # LayerNorm: ~3x data transfers (2 read + 1 write)
        transfers = 3
        return (data_size_bytes * transfers) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec

    # Convert machine spec to device name
    device = MACHINE_SPEC_TO_GPU_MODEL.get(machine_spec.name, "h100")
    dtype_str = DTYPE_TO_STR.get(dtype, "bf16")

    # Find closest match
    match = _find_closest_match(df, batch, seqlen, hidden_dim, dtype_str, device)

    if match is None or pd.isna(match["norm_time_ms"]):
        # Fallback to bandwidth estimation
        data_size_bytes = batch * seqlen * hidden_dim * dtype.size_bytes()
        transfers = 3
        return (data_size_bytes * transfers) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec

    # Convert milliseconds to seconds
    return match["norm_time_ms"] / 1000.0


def get_rope_time(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    dtype: DType,
    machine_spec: MachineSpec,
) -> float:
    """Get RoPE timing from lookup table.

    Falls back to HBM bandwidth estimation if lookup fails.

    Args:
        batch: Batch size
        seqlen: Sequence length
        hidden_dim: Hidden dimension
        dtype: Data type
        machine_spec: Machine specification

    Returns:
        RoPE time in seconds
    """
    df = _load_norm_rope_timings()

    if df is None:
        # Fallback: Use HBM bandwidth estimation
        data_size_bytes = batch * seqlen * hidden_dim * dtype.size_bytes()
        # RoPE: ~3x data transfers (2 read + 1 write)
        transfers = 3
        return (data_size_bytes * transfers) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec

    # Convert machine spec to device name
    device = MACHINE_SPEC_TO_GPU_MODEL.get(machine_spec.name, "h100")
    dtype_str = DTYPE_TO_STR.get(dtype, "bf16")

    # Find closest match
    match = _find_closest_match(df, batch, seqlen, hidden_dim, dtype_str, device)

    if match is None or pd.isna(match["rope_time_ms"]):
        # Fallback to bandwidth estimation
        data_size_bytes = batch * seqlen * hidden_dim * dtype.size_bytes()
        transfers = 3
        return (data_size_bytes * transfers) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec

    # Convert milliseconds to seconds
    return match["rope_time_ms"] / 1000.0


def get_norm_time_or_default(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    dtype: DType,
    machine_spec: MachineSpec,
) -> float:
    """Get LayerNorm timing with default fallback.

    Alias for get_norm_time for consistency with gemm_util API.

    Args:
        batch: Batch size
        seqlen: Sequence length
        hidden_dim: Hidden dimension
        dtype: Data type
        machine_spec: Machine specification

    Returns:
        LayerNorm time in seconds
    """
    return get_norm_time(batch, seqlen, hidden_dim, dtype, machine_spec)


def get_rope_time_or_default(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    dtype: DType,
    machine_spec: MachineSpec,
) -> float:
    """Get RoPE timing with default fallback.

    Alias for get_rope_time for consistency with gemm_util API.

    Args:
        batch: Batch size
        seqlen: Sequence length
        hidden_dim: Hidden dimension
        dtype: Data type
        machine_spec: Machine specification

    Returns:
        RoPE time in seconds
    """
    return get_rope_time(batch, seqlen, hidden_dim, dtype, machine_spec)
