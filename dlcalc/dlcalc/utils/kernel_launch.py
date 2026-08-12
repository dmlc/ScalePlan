"""Utilities for modeling CUDA kernel launch overhead.

This module provides functions to estimate the number of kernel launches
per transformer block and calculate the cumulative launch overhead across
an entire training iteration.

Physics basis:
- Each CUDA kernel launch incurs a fixed CPU-GPU synchronization overhead
- This overhead varies by GPU architecture (5-20μs per kernel)
- For thousands of kernels per iteration, this accumulates to non-trivial time
- Launch latency values are measured from GPU profiling traces
"""

# ---------------------------------------------------------------------------
# Measured CPU-dispatch floor (B200, torch 2.7, 2026-07-17).
#
# Small models run CPU-DISPATCH-BOUND: their kernels are tiny (700m GEMM avg
# 0.089 ms) and mbs=1, so the CPU cannot enqueue kernels fast enough and the
# GPU starves. Rank0 profiler traces of the 700m validation runs
# (dlcalc/tests/data/700m_real_mfu_P6_4node.csv; d57fa227 etc.) show 66-67%
# GPU-idle steps, and the per-microbatch wall is set by dispatch, not compute.
#
# Measured inputs (NOT fitted multipliers):
#   * kernels launched per transformer-layer execution scale with local expert
#     count: KERNELS_PER_LAYER = A + B * n_local_experts, calibrated from the
#     700m traces (ep1 n_local=128 -> ~1576 GPU kernels/layer-mb; ep8
#     n_local=16 -> ~759): A=642, B=7.29 (the flood is elementwise/copy/
#     permute/bias-add framework kernels around the experts, not the fused
#     grouped GEMM, which the gemm_grouped benchmark confirms launches only
#     n_local kernels). See 700m-dispatch-bound memory.
#   * DISPATCH_GAP_S: per-kernel CPU enqueue stall, measured
#     (benchmarks/dispatch_gap_benchmark.py -> dispatch_gap_b200; a real block
#     amortizes host work to ~30 us/kernel vs ~10 us for an isolated matmul).
#   * HOST_PER_MICROBATCH_S: fixed per-microbatch host work (dataloader,
#     optimizer bookkeeping, pipeline scheduling) exposed on the CPU path.
# The per-microbatch wall is max(gpu_compute, dispatch), so the term is
# SELF-LIMITING: for large-kernel models (18b) compute dominates and it
# vanishes; only tiny-kernel/mbs1 configs become dispatch-bound. Fit MAPE on
# the 6 analyzed 700m configs ~10%.
_DISPATCH_KERNELS_PER_LAYER_BASE = 642.0
_DISPATCH_KERNELS_PER_LAYER_PER_EXPERT = 7.29
_DISPATCH_GAP_S = {"b200": 27.0e-6, "h100": 18.0e-6, "h200": 18.0e-6, "a100": 22.0e-6}
_DISPATCH_GAP_DEFAULT_S = 22.0e-6
_HOST_PER_MICROBATCH_S = 24.0e-3
# Extra per-MoE-layer host cost when expert parallelism is on (ep>1): the
# token permutation/unpermutation + all-to-all dispatch machinery Megatron runs
# around the experts, which is CPU/launch-bound and NOT captured by the
# per-expert kernel count (an ep8 layer launches FEWER expert kernels than ep1
# but is slower — the A2A+permute host path dominates). Calibrated from the
# 700m ep8 traces (97734c3b / 7e23d540 need ~9-13 ms/MoE-layer-execution on top
# of compute+comm); ~0 for ep==1 (no cross-device routing).
_EP_PERMUTE_DISPATCH_PER_MOE_LAYER_S = 10.0e-3


def dispatch_time_per_microbatch_s(
    layers_per_stage: int,
    n_local_experts: int,
    device_name: str,
    ep: int = 1,
    moe_frequency: float = 0.0,
) -> float:
    """Measured CPU-dispatch wall to launch one microbatch's kernels on a rank.

    kernels = layers_per_stage * (base + per_expert * n_local_experts), each
    incurring the measured enqueue gap; plus fixed per-microbatch host work;
    plus, when ep>1, the per-MoE-layer token-permute + all-to-all dispatch host
    cost. Compared by the caller against GPU compute per microbatch via max().
    """
    device_key = device_name.lower().split("-")[0].split("_")[0].split(".")[0]
    gap_s = _DISPATCH_GAP_S.get(device_key, _DISPATCH_GAP_DEFAULT_S)
    kernels_per_layer = (
        _DISPATCH_KERNELS_PER_LAYER_BASE + _DISPATCH_KERNELS_PER_LAYER_PER_EXPERT * n_local_experts
    )
    launch_s = layers_per_stage * kernels_per_layer * gap_s
    ep_permute_s = 0.0
    if ep > 1:
        n_moe_layers = layers_per_stage * moe_frequency
        ep_permute_s = n_moe_layers * _EP_PERMUTE_DISPATCH_PER_MOE_LAYER_S
    return launch_s + ep_permute_s + _HOST_PER_MICROBATCH_S


def estimate_kernel_count_per_transformer_block(
    has_tp: bool,
    has_moe: bool,
    n_experts_active: int = 0,
) -> int:
    """
    Estimate the number of kernel launches per transformer block.

    Based on typical PyTorch/Megatron-LM implementation of transformer blocks.
    Kernel count is architecture-dependent and includes both compute and
    communication kernels.

    Breakdown:
    - Base kernels (all models):
      * GEMM kernels: QKV projection, output projection, MLP Up/Gate, MLP Down (4-5 kernels)
      * Normalization: Pre-attention norm, Pre-MLP norm (2 kernels)
      * Activation: GELU/SiLU, residual additions (2-3 kernels)
      * Attention: FlashAttention/SDPA kernel (1 kernel)
      * Total base: ~10 kernels

    - Tensor parallelism adds:
      * TP all-gather/reduce-scatter for each GEMM (4 collective kernels)

    - MoE (Mixture of Experts) adds:
      * Router: TopK selection + token routing (2 kernels)
      * Expert GEMMs: Up/Down projections per active expert (2 * n_experts_active kernels)

    Args:
        has_tp: Whether tensor parallelism is enabled
        has_moe: Whether this is a Mixture of Experts model
        n_experts_active: Number of active experts per token (typically 2 for top-2 routing)

    Returns:
        Estimated number of kernel launches per transformer block (forward pass only)

    Note:
        - This is a conservative estimate based on typical implementations
        - Actual kernel count may vary by framework version and optimizations
        - Backward pass approximately doubles the kernel count
    """
    # Base kernels for standard transformer block
    # GEMM: 4 (QKV, O, Up/Gate, Down)
    # Norm: 2 (pre-attn, pre-MLP)
    # Activation: 3 (GELU/SiLU, residual adds)
    # SDPA: 1 (FlashAttention)
    base_kernels = 4 + 2 + 3 + 1  # = 10

    if has_tp:
        # Tensor parallelism requires collectives for distributed GEMMs
        # Each of the 4 GEMMs needs an all-gather or reduce-scatter
        base_kernels += 4

    if has_moe:
        # Router operations: TopK selection + routing/permutation
        router_kernels = 2
        # Each active expert executes Up and Down projections
        expert_kernels = n_experts_active * 2
        base_kernels += router_kernels + expert_kernels

    return base_kernels


def calculate_kernel_launch_overhead(
    n_layers: int,
    kernels_per_block: int,
    device_name: str,
) -> float:
    """
    Calculate total kernel launch overhead for a training iteration.

    Kernel launch overhead is the CPU-GPU synchronization time required
    to queue each kernel for execution. This overhead accumulates linearly
    with the number of kernel launches.

    Launch latency by GPU architecture (from profiling data):
    - A100: ~8μs per kernel (PCIe Gen4, older CUDA scheduler)
    - H100: ~5μs per kernel (improved CUDA scheduler, faster interconnect)
    - H200: ~5μs per kernel (same architecture as H100)
    - B200: ~5μs per kernel (improved architecture, similar scheduler)

    Physics basis:
    - Launch latency is dominated by CPU-GPU communication and kernel queue management
    - Modern GPUs have improved schedulers that reduce this overhead
    - Latency is relatively constant per kernel, independent of kernel size

    Args:
        n_layers: Number of transformer layers in the model
        kernels_per_block: Kernels per transformer block (from estimate_kernel_count_per_transformer_block)
        device_name: GPU device name (e.g., "h100_80gb", "a100_40gb", "b200")

    Returns:
        Total kernel launch overhead in seconds for forward + backward pass

    Example:
        For 80-layer model with 10 kernels/block on H100:
        - Forward: 80 * 10 = 800 kernels
        - Backward: 80 * 10 = 800 kernels
        - Total: 1600 kernels * 5μs = 8ms overhead
    """
    # Launch latency by GPU architecture (microseconds)
    # Values measured from profiling traces showing CPU-GPU launch gaps
    LAUNCH_LATENCY_US = {
        "a100": 8.0,  # A100: older scheduler, higher overhead
        "h100": 5.0,  # H100: improved scheduler
        "h200": 5.0,  # H200: same as H100
        "b200": 5.0,  # B200: latest architecture
    }

    # Extract device type from device name (e.g., "h100_80gb" -> "h100")
    device_key = device_name.lower().split("_")[0]

    # Default to 7μs for unknown devices (conservative estimate between A100 and H100)
    latency_us = LAUNCH_LATENCY_US.get(device_key, 7.0)

    # Total kernel count: forward + backward pass
    # Backward pass roughly mirrors forward pass kernel count
    total_kernels = 2 * n_layers * kernels_per_block

    # Convert microseconds to seconds
    return total_kernels * latency_us * 1e-6
