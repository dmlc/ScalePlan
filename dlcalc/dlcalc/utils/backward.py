"""Per-op backward pass timing models.

Rather than approximating the backward pass as a constant multiple of the
forward pass, this module provides an explicit, autograd-style model: each
forward op has a corresponding backward function that computes the time for
the actual operations autograd would execute.

References:
* Flash-Attention-2 backward: Dao 2023 (https://arxiv.org/pdf/2307.08691), §3.1.2
* LayerNorm backward: Megatron-LM fused LayerNorm kernel
* Standard linear backward: see any autograd textbook (dX = dY @ W.T, dW = X.T @ dY)
"""

from .data import Size
from .gemm_util import get_gemm_utilization_or_default
from .hardware import DType, MachineSpec
from .moe_router_util import topk_throughput_elem_per_s
from .sdpa_util import get_sdpa_bwd_time_s as _lookup_sdpa_bwd_time_s


def compute_linear_bwd_time_s(
    n_tokens: int,
    weight_shape: tuple[int, int],
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """Backward pass time for a linear layer Y = X @ W.

    Autograd executes two GEMMs:
    1. dX = dY @ W.T    shape: (n_tokens, n_out) @ (n_out, n_in) -> (n_tokens, n_in)
                        FLOPs = 2 * n_tokens * n_out * n_in
    2. dW = X.T @ dY    shape: (n_in, n_tokens) @ (n_tokens, n_out) -> (n_in, n_out)
                        FLOPs = 2 * n_in * n_tokens * n_out

    Each backward GEMM has the same FLOP count as the forward pass but different
    shape characteristics, so their utilization is looked up independently.

    Args:
        n_tokens: Number of tokens in the forward pass (M dimension of the forward GEMM)
        weight_shape: (n_in, n_out) shape of the weight matrix
        machine_spec: Target hardware spec
        dtype: Compute precision

    Returns:
        Total backward time in seconds (dX GEMM + dW GEMM)
    """
    n_in, n_out = weight_shape
    peak_flops = machine_spec.device_spec.peak_flops(dtype)

    # dX = dY @ W.T: (M=n_tokens, N=n_in, K=n_out)
    dx_util = get_gemm_utilization_or_default(
        m=n_tokens, n=n_in, k=n_out, machine_spec=machine_spec, dtype=dtype
    )
    dx_flops = 2 * n_tokens * n_out * n_in
    dx_time_s = dx_flops / (peak_flops * dx_util)

    # dW = X.T @ dY: (M=n_in, N=n_out, K=n_tokens)
    dw_util = get_gemm_utilization_or_default(
        m=n_in, n=n_out, k=n_tokens, machine_spec=machine_spec, dtype=dtype
    )
    dw_flops = 2 * n_in * n_tokens * n_out
    dw_time_s = dw_flops / (peak_flops * dw_util)

    return dx_time_s + dw_time_s


def compute_grouped_linear_bwd_time_s(
    n_tokens_per_group: int,
    n_groups: int,
    weight_shape: tuple[int, int],
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """Backward for a grouped GEMM where each group has its own weight matrix.

    Used for MoE expert MLPs. The forward is ONE grouped/batched launch
    (``_GroupedLinear``, cutlass) processing all local tokens, so the backward
    is likewise two grouped GEMMs (dX, dW) over the aggregate token-rows -- not
    ``n_groups`` independent small GEMMs.

    FLOPs are the aggregate over all local groups
    (``M_eff = n_tokens_per_group * n_groups``; ``2 * M_eff * N * K`` for each of
    dX and dW), but the grouped kernel's EFFICIENCY is set by the PER-GROUP tile,
    so utilization is looked up at the per-group M (= ``n_tokens_per_group``), NOT
    the aggregate M_eff. Looking it up at M_eff over-credited util by 5-11x in the
    dropless regime (measured on B200; see
    training_3d.compute_expert_gemm_time_s and
    benchmarks/results/gemm_grouped_b200.parquet). This is EP-invariant
    (``n_tokens_per_group`` and hence util is flat in EP), so the grouped backward
    time is flat in EP as measured (~the forward's ~140ms scaled by the bwd 2x).

    See training_3d.compute_expert_gemm_time_s for the symmetric forward fix.
    """
    n_in, n_out = weight_shape
    peak_flops = machine_spec.device_spec.peak_flops(dtype)

    # Aggregate token-rows across all local groups (the grouped GEMM's FLOP count).
    m_eff = n_tokens_per_group * n_groups

    # dX = dY @ W.T: FLOPs over M_eff rows, but util at the per-group tile M.
    dx_util = get_gemm_utilization_or_default(
        m=n_tokens_per_group, n=n_in, k=n_out, machine_spec=machine_spec, dtype=dtype
    )
    dx_flops = 2 * m_eff * n_out * n_in
    dx_time_s = dx_flops / (peak_flops * dx_util)

    # dW = X.T @ dY: the wgrad contracts over the per-group token rows (K=tokens/
    # group), accumulating into the [n_in, n_out] weight -- so its efficiency is
    # also set by the small per-group contraction, not the aggregate.
    dw_util = get_gemm_utilization_or_default(
        m=n_in, n=n_out, k=n_tokens_per_group, machine_spec=machine_spec, dtype=dtype
    )
    dw_flops = 2 * n_in * m_eff * n_out
    dw_time_s = dw_flops / (peak_flops * dw_util)

    return dx_time_s + dw_time_s


def compute_sdpa_bwd_time_s(
    seqlen_per_cp: int,
    seqlen_full: int,
    head_dim: int,
    n_q_heads_local: int,
    n_kv_heads_local: int,
    micro_bs: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """Measured SDPA backward wall-clock time in seconds.

    Looks up the closest-match entry in `sdpa_fwd_bwd_timings.parquet`
    (produced by `benchmarks/sdpa_benchmark.py` on A100/H200/B200 sleepers
    using TE.DotProductAttention with causal masking, bf16). No theoretical
    multiplier is applied - this is raw measured kernel time.

    Raises RuntimeError if no measurement data is available. Callers should
    ensure the parquet is shipped with the install (the repo includes it).

    Args:
        seqlen_per_cp: per-CP-rank sequence length (unused in lookup; the
            forward parquet is indexed by full seqlen because CP is a runtime
            sharding, not a kernel parameter). Kept for API symmetry.
        seqlen_full: sequence length used to index the parquet
        head_dim: per-head feature dim
        n_q_heads_local: Q heads on this TP shard (total_q_heads / tp)
        n_kv_heads_local: KV heads on this TP shard (total_kv_heads / tp)
        micro_bs: microbatch size
        machine_spec: target hardware (for device label lookup)
        dtype: compute precision

    Returns:
        Backward time in seconds (from the measured parquet).
    """
    del seqlen_per_cp  # unused; CP is handled at the iteration level

    t = _lookup_sdpa_bwd_time_s(
        seq_len=seqlen_full,
        micro_bs=micro_bs,
        n_q_heads=n_q_heads_local,
        n_kv_heads=n_kv_heads_local,
        head_dim=head_dim,
        machine_spec=machine_spec,
        dtype=dtype,
    )
    if t is None:
        raise RuntimeError(
            f"No measured SDPA backward time for "
            f"device={machine_spec.name} dtype={dtype} "
            f"seqlen={seqlen_full} mbs={micro_bs} "
            f"qh={n_q_heads_local} kvh={n_kv_heads_local} hd={head_dim}. "
            f"Run benchmarks/sdpa_benchmark.py to extend the parquet."
        )
    return t


def compute_layernorm_bwd_time_s(
    numel: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """LayerNorm backward time (memory-bandwidth bound).

    Autograd computes:
      dX   = (1 / sigma) * (dY * gamma - mean(dY * gamma) - xhat * mean(dY * gamma * xhat))
      dGamma = sum_batch(dY * xhat)
      dBeta  = sum_batch(dY)

    Memory transfers on the (B, S, H) tensor:
    - Reads : dY, X, gamma, mean, sigma (roughly 3x + 2x broadcast ≈ 3x B*S*H)
    - Writes: dX                       (1x B*S*H)
    - dGamma/dBeta writes are O(H) so negligible here.

    We model this as ~5x the activation size (conservative; Megatron's fused
    kernel saves some read traffic by caching the forward mean/sigma).
    """
    data_size_bytes = numel * dtype.size_bytes()
    transfers = 5
    return (data_size_bytes * transfers) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec


def compute_rope_bwd_time_s(
    numel: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """RoPE backward time (memory-bandwidth bound).

    RoPE is an element-wise rotation; the backward applies the transposed rotation
    (same formula with swapped sign). Memory transfers are symmetric with the
    forward: read dY, read cos/sin tables, write dX ≈ 3x transfers.
    """
    data_size_bytes = numel * dtype.size_bytes()
    transfers = 3
    return (data_size_bytes * transfers) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec


def compute_residual_bwd_time_s(
    activation_size: Size,
    machine_spec: MachineSpec,
) -> float:
    """Residual connection backward time.

    For y = x1 + x2 (forward), autograd splits the grad: dx1 = dy, dx2 = dy.
    Memory pattern: 1 read of dy, 2 writes (to dx1 and dx2).
    """
    return 3 * activation_size.bytes() / machine_spec.device_spec.mem_bandwidth_bytes_per_sec


def compute_glu_bwd_time_s(
    numel: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """GLU (SwiGLU) backward time.

    Forward: y = silu(a) * b (where a, b are the two halves of the up-projection).
    Backward: da = dy * b * silu'(a), db = dy * silu(a).
    Memory transfers on the intermediate tensor:
    - Reads : dy, a, b (3 reads)
    - Writes: da, db   (2 writes)
    Total: 5 transfers.
    """
    data_size_bytes = numel * dtype.size_bytes()
    transfers = 5
    return (data_size_bytes * transfers) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec


def compute_permutation_bwd_time_s(
    n_tokens: int,
    hidden_dim: int,
    machine_spec: MachineSpec,
    dtype: DType,
) -> float:
    """MoE token permutation backward time (memory-bandwidth bound).

    Forward does permute + unpermute (4x data transfers). Backward does the same
    pattern in reverse (unpermute dy, scatter to original positions) for the same
    total data movement.
    """
    data_bytes = n_tokens * hidden_dim * dtype.size_bytes()
    total_transfers = 4
    return (data_bytes * total_transfers) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec


def compute_topk_bwd_time_s(
    batch: int,
    seqlen: int,
    n_experts: int,
    machine_spec: MachineSpec,
) -> float:
    """TopK (router) backward time.

    The TopK selection itself is not differentiable; only the softmax/scores are.
    Autograd scatters the grad from the selected experts back into the full
    [tokens, experts] logits tensor -- the same memory volume as the forward TopK
    selection, and MEASURED (benchmarks/topk_benchmark.py bench_bwd_event, B200
    2026-07-14) to run at ~0.7-1.2x the forward throughput. So it uses the SAME
    calibrated per-GPU throughput as the forward.

    Previously this hardcoded a separate table (b200=1.5e8 elem/s) that was ~100x
    too slow -- a plausible-looking but uncalibrated guess -- producing a ~7ms
    phantom TopK-backward per MoE block and a ~30x over-prediction of the whole
    routing bucket (perf_scratch/COMPONENT_COMPARE.md). Sharing the forward's
    measured table removes both the error and the drift-prone duplication.
    """
    n_elements = batch * seqlen * n_experts
    return n_elements / topk_throughput_elem_per_s(machine_spec)
