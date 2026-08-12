"""Utilities for modeling MoE router operations.

This module provides functions to calculate the time for MoE (Mixture of Experts)
router operations, including GEMM, TopK selection, and token permutation/unpermutation.

Physics basis:
- Router GEMM: Standard matrix multiplication, uses existing GEMM utilization infrastructure
- TopK selection: Memory-bound operation with GPU-specific throughput
- Token permutation: Memory bandwidth-bound operation (scatter/gather pattern)
"""

from .gemm_util import get_gemm_utilization_or_default
from .hardware import DType, DeviceSpec, MachineSpec

# TopK throughput by GPU architecture (elements/second), event wall-clock.
# Measured with benchmarks/topk_benchmark.py (torch.topk, ATen sbtopk::gatherTopK);
# see benchmarks/results/topk_{a100,h200,b200}.parquet. Event time (not isolated
# kernel time) is the right metric because dlcalc sums component times serially and
# implicitly pays per-launch overhead per call. The B200 value is the cross-shape
# median from the 2026-07-14 sweep on a p6-b200 node (240 pts, torch
# 2.7 / TE 2.4.90); the target model shape alone runs faster (~1.8e10).
_TOPK_THROUGHPUT_ELEM_PER_S = {
    "a100": 1.1e10,  # A100-SXM4-40GB (PyTorch 2.4, CUDA 12.5)
    "h100": 1.7e10,  # H100 assumed same as H200 (same sbtopk kernel family)
    "h200": 1.7e10,  # H200-SXM5 (PyTorch 2.4, CUDA 12.5)
    "b200": 1.5e10,  # B200 cross-shape median (2026-07-14 sweep)
}
_TOPK_THROUGHPUT_DEFAULT = 9e9  # unknown device: conservative ~A100-class


def topk_throughput_elem_per_s(machine_spec: MachineSpec) -> float:
    """Measured torch.topk throughput (elements/sec) for this GPU.

    Shared by the forward (calculate_topk_time) and the backward
    (backward.compute_topk_bwd_time_s): the topk backward is a scatter of
    grad_values back into the full [tokens, experts] logits tensor -- the same
    memory volume as the forward selection, and MEASURED to run at ~0.7-1.2x the
    forward throughput on B200 (bench_bwd_event in topk_benchmark.py). So both
    directions use this one calibrated table, rather than a duplicated (and
    previously 100x-wrong) backward constant.
    """
    device_key = machine_spec.name.lower().split("_")[0].split(".")[0]
    for key, tput in _TOPK_THROUGHPUT_ELEM_PER_S.items():
        if key in device_key:
            return tput
    return _TOPK_THROUGHPUT_DEFAULT


def calculate_router_gemm_time(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    n_experts: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """
    Calculate MoE router GEMM time.

    The router GEMM computes logits for expert selection:
    Router: [batch * seqlen, hidden_dim] @ [hidden_dim, n_experts] -> [batch * seqlen, n_experts]

    This uses the existing GEMM utilization infrastructure to account for hardware efficiency.

    Args:
        batch: Batch size (microbatch size)
        seqlen: Sequence length
        hidden_dim: Hidden dimension size
        n_experts: Number of experts in the MoE layer
        machine_spec: Machine specification (includes device spec and GPU type)
        dtype: Data type (DType enum: BF16, FP16, FP32, FP8)

    Returns:
        Router GEMM time in seconds

    Example:
        For batch=1, seqlen=4096, hidden_dim=4096, n_experts=64 on H100:
        - M = 1 * 4096 = 4096
        - K = 4096
        - N = 64
        - FLOPs = 2 * 4096 * 64 * 4096 ≈ 2.1B FLOPs
        - With ~80% utilization on H100: ~0.01ms
    """
    # Router GEMM dimensions
    m = batch * seqlen  # Number of tokens
    k = hidden_dim  # Hidden dimension
    n = n_experts  # Number of experts (output dimension)

    # Get GEMM utilization from empirical lookup table
    gemm_util = get_gemm_utilization_or_default(
        m=m,
        n=n,
        k=k,
        machine_spec=machine_spec,
        dtype=dtype,
    )

    # Calculate FLOPs: 2MNK for matrix multiplication
    flops = 2 * m * n * k

    # Calculate time: FLOPs / (peak_flops * utilization)
    time_s = flops / (machine_spec.device_spec.peak_flops(dtype) * gemm_util)

    return time_s


def calculate_topk_time(
    batch: int,
    seqlen: int,
    n_experts: int,
    k: int,
    machine_spec: MachineSpec,
) -> float:
    """
    Calculate TopK selection time for MoE routing.

    The TopK operation selects the top-K experts for each token based on router logits.
    This is a memory-bound operation that scans through expert scores.

    Physics basis:
    - TopK is primarily memory-bound (reading logits, writing indices)
    - Throughput depends on GPU memory system and kernel implementation
    - Measured from profiling traces across different GPUs

    Throughput (elements/second) comes from ``topk_throughput_elem_per_s``, a
    measured per-GPU table shared with the backward (see that helper and
    `benchmarks/topk_benchmark.py` / `benchmarks/results/topk_*.parquet`).

    Args:
        batch: Batch size (microbatch size)
        seqlen: Sequence length
        n_experts: Number of experts to select from
        k: Number of experts to select per token (typically 1 or 2)
        machine_spec: Machine specification

    Returns:
        TopK selection time in seconds

    Example:
        For batch=1, seqlen=4096, n_experts=64, k=2 on H200:
        - Elements to process = 1 * 4096 * 64 = 262,144
        - Throughput = 1.7e10 elements/sec
        - Time = 262,144 / 1.7e10 ≈ 15 μs
    """
    # Total elements to process: each token needs TopK across all experts
    n_elements = batch * seqlen * n_experts

    # Time = elements / throughput
    return n_elements / topk_throughput_elem_per_s(machine_spec)


def calculate_permutation_time(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    machine_spec: MachineSpec,
    dtype_bytes: int = 2,
) -> float:
    """
    Calculate token permutation/unpermutation time for MoE routing.

    MoE routing requires reordering tokens based on expert assignment:
    1. Permutation: Gather tokens assigned to each expert (before expert processing)
    2. Unpermutation: Scatter tokens back to original positions (after expert processing)

    Physics basis:
    - Memory bandwidth-bound operation (scatter/gather pattern)
    - Each token is read once and written once for both permutation and unpermutation
    - Total transfers: 2 reads + 2 writes = 4x data size

    Memory bandwidth model:
    - Permutation: read original positions, write to expert-sorted positions
    - Unpermutation: read expert-sorted positions, write to original positions
    - Each operation: 1 read + 1 write = 2x data transfers
    - Total: 4x data transfers

    Args:
        batch: Batch size (microbatch size)
        seqlen: Sequence length
        hidden_dim: Hidden dimension size
        machine_spec: Machine specification
        dtype_bytes: Bytes per element (default 2 for BF16/FP16)

    Returns:
        Total permutation time in seconds (permutation + unpermutation)

    Example:
        For batch=1, seqlen=4096, hidden_dim=4096, BF16 on H100 (2TB/s):
        - Data size = 1 * 4096 * 4096 * 2 bytes = 32MB
        - Total transfers = 4 * 32MB = 128MB
        - Time = 128MB / 2TB/s ≈ 64μs
    """
    # Calculate data size for all tokens
    # Each token: hidden_dim * dtype_bytes
    data_bytes = batch * seqlen * hidden_dim * dtype_bytes

    # Total memory transfers: 2x for permutation + 2x for unpermutation
    # Permutation: 1 read + 1 write
    # Unpermutation: 1 read + 1 write
    total_transfers = 4

    # Memory bandwidth in bytes/sec
    memory_bw_bytes_per_sec = machine_spec.device_spec.mem_bandwidth_bytes_per_sec

    # Time = (data_size * num_transfers) / bandwidth
    time_s = (data_bytes * total_transfers) / memory_bw_bytes_per_sec

    return time_s


def calculate_total_moe_router_overhead(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    n_experts: int,
    k: int,
    machine_spec: MachineSpec,
    dtype: DType,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """
    Calculate total MoE router overhead including all operations.

    Combines all router operations for easy integration:
    1. Router GEMM: Compute expert scores
    2. TopK selection: Select top-K experts per token
    3. Token permutation/unpermutation: Reorder tokens for expert processing

    Args:
        batch: Batch size
        seqlen: Sequence length
        hidden_dim: Hidden dimension
        n_experts: Number of experts
        k: Top-K experts per token
        machine_spec: Machine specification
        dtype: Data type (DType enum)
        dtype_bytes: Bytes per element

    Returns:
        Dictionary with breakdown of router operation times:
        - "Router GEMM": Router GEMM time
        - "Router TopK": TopK selection time
        - "Router Permutation": Permutation + unpermutation time
        - "Total": Sum of all operations
    """
    router_gemm_time = calculate_router_gemm_time(
        batch, seqlen, hidden_dim, n_experts, machine_spec, dtype
    )

    topk_time = calculate_topk_time(batch, seqlen, n_experts, k, machine_spec)

    permutation_time = calculate_permutation_time(
        batch, seqlen, hidden_dim, machine_spec, dtype_bytes
    )

    return {
        "Router GEMM": router_gemm_time,
        "Router TopK": topk_time,
        "Router Permutation": permutation_time,
        "Total": router_gemm_time + topk_time + permutation_time,
    }
