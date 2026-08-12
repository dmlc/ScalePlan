"""Utilities for looking up GEMM utilization from measured data."""
import os
from functools import lru_cache
from typing import Optional

import pandas as pd

from dlcalc.utils.hardware import DType, MachineSpec


# Mapping from machine spec names to GPU model names in the parquet file
MACHINE_SPEC_TO_GPU_MODEL = {
    "p5.48xlarge": "H100",
    "p5e.48xlarge": "H100",
    "p5en.48xlarge": "H100",
    "p6-b200.48xlarge": "B200"
    # Add B200 mappings when available
}

# Mapping from DType enum to dtype strings in parquet
DTYPE_TO_STR = {
    DType.FP16: "fp16",
    DType.BF16: "bf16",
    DType.FP8: "fp8",
    DType.FP8_E4M3: "fp8"  # FP8_E4M3 also maps to "fp8" in parquet
}

# Default fallback utilization if lookup fails
DEFAULT_GEMM_UTIL = 0.7


def _round_to_nearest_power_of_2(n: int) -> int:
    """Round n to the nearest power of 2.

    Args:
        n: Input integer

    Returns:
        Nearest power of 2 to n
    """
    if n <= 0:
        return 1

    # Find the two closest powers of 2
    lower = 1 << (n.bit_length() - 1)  # 2^floor(log2(n))
    upper = 1 << n.bit_length()  # 2^ceil(log2(n))

    # Return the closer one
    if abs(n - lower) <= abs(n - upper):
        return lower
    else:
        return upper


@lru_cache(maxsize=1)
def _load_gemm_util_data() -> pd.DataFrame:
    """Load GEMM utilization data from parquet file.

    Returns:
        DataFrame with multi-index (M, N, K, GPU_MODEL, dtype) and util_% column.
    """
    # Measured data lives with the other benchmark artifacts.
    parquet_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "benchmarks", "results", "gemm_util.parquet",
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"GEMM utilization parquet file not found at {parquet_path}"
        )
    return pd.read_parquet(parquet_path)


def get_gemm_utilization(
    m: int,
    n: int,
    k: int,
    machine_spec: MachineSpec,
    dtype: DType,
    use_default_on_missing: bool = True,
) -> float:
    """Look up GEMM utilization from measured data.

    Args:
        m: First dimension of GEMM (number of tokens/batch size)
        n: Second dimension of GEMM (output features)
        k: Third dimension of GEMM (input features)
        machine_spec: Machine specification
        dtype: Data type for the GEMM operation
        use_default_on_missing: If True, return DEFAULT_GEMM_UTIL when lookup fails.
                                If False, raise KeyError.

    Returns:
        GEMM utilization as a fraction (0-1), e.g., 0.7 for 70% utilization.

    Raises:
        KeyError: If the lookup fails and use_default_on_missing is False.
        ValueError: If the GEMM measurement has a non-OK status.
    """
    # Map machine spec to GPU model using the predefined mapping
    if machine_spec.name not in MACHINE_SPEC_TO_GPU_MODEL:
        if use_default_on_missing:
            return DEFAULT_GEMM_UTIL
        raise KeyError(
            f"Machine spec {machine_spec.name} not found in MACHINE_SPEC_TO_GPU_MODEL"
        )

    gpu_model = MACHINE_SPEC_TO_GPU_MODEL[machine_spec.name]

    # Map dtype
    if dtype not in DTYPE_TO_STR:
        if use_default_on_missing:
            return DEFAULT_GEMM_UTIL
        raise KeyError(f"Unsupported dtype: {dtype}")

    dtype_str = DTYPE_TO_STR[dtype]

    # Load data and lookup
    try:
        df = _load_gemm_util_data()
        # Round m, n, k to nearest power of 2 for lookup
        m_rounded = _round_to_nearest_power_of_2(m)
        n_rounded = _round_to_nearest_power_of_2(n)
        k_rounded = _round_to_nearest_power_of_2(k)
        if df.loc[(m_rounded, n_rounded, k_rounded, gpu_model, dtype_str), 'status'] != 'OK':
            status = df.loc[(m_rounded, n_rounded, k_rounded, gpu_model, dtype_str), 'status']
            raise ValueError(
                f"GEMM measurement has non-OK status '{status}' for M={m_rounded}, N={n_rounded}, K={k_rounded}, "
                f"GPU={gpu_model}, dtype={dtype_str}"
            )
        util_pct = df.loc[(m_rounded, n_rounded, k_rounded, gpu_model, dtype_str), 'util_%']
        return util_pct / 100.0
    except (KeyError, FileNotFoundError) as e:
        if use_default_on_missing:
            return DEFAULT_GEMM_UTIL
        raise KeyError(
            f"No GEMM utilization data found for M={m}, N={n}, K={k}, "
            f"GPU={gpu_model}, dtype={dtype_str}: {e}"
        )


def get_gemm_utilization_or_default(
    m: int,
    n: int,
    k: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """Look up GEMM utilization with fallback to default.

    This is a convenience wrapper that always returns a value.

    Args:
        m: First dimension of GEMM (number of tokens/batch size)
        n: Second dimension of GEMM (output features)
        k: Third dimension of GEMM (input features)
        machine_spec: Machine specification
        dtype: Data type for the GEMM operation

    Returns:
        GEMM utilization as a fraction (0-1).
    """
    return get_gemm_utilization(
        m=m,
        n=n,
        k=k,
        machine_spec=machine_spec,
        dtype=dtype,
        use_default_on_missing=True
    )
