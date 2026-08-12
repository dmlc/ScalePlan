"""
Reduce-Scatter:
* Ring: https://github.com/NVIDIA/nccl/blob/ab2b89c4c339bd7f816fbc114a4b05d386b66290/src/device/reduce_scatter.h#L12-L54

All-Gather:
* Ring: https://github.com/NVIDIA/nccl/blob/ab2b89c4c339bd7f816fbc114a4b05d386b66290/src/device/all_gather.h#L12-L64

All-Reduce (not relevant for comm patterns we have represented for now):
some info on tree AR: https://github.com/NVIDIA/nccl/issues/545

NCCL Protocol Overhead:
* Protocol selection and overhead now modeled (see get_nccl_protocol_overhead)
* Reference: https://github.com/NVIDIA/nccl/issues/281
* LL (<32KB), LL128 (32KB-1MB), Simple (>1MB) protocols with bandwidth/latency multipliers

NOTE on protocol selection (NCCL src/graph/tuning.cc): NCCL does NOT actually
switch protocols on fixed byte thresholds. `ncclTopoGetAlgoTime` estimates
`lat*latCount + nBytes/(1000*bw)` for every (algorithm, protocol) pair and picks
the minimum. The message-size thresholds below are a simplification of that
cost-model crossover that is adequate for the DP/TP ring collectives we model
(their buckets are multi-MB, firmly in the Simple regime), but the per-protocol
*bandwidth efficiencies* are pinned to NCCL's own line-efficiency constants so
the Simple/LL128 numbers are physical rather than hand-tuned.
"""

from enum import Enum

from .comm_bench_util import (
    measured_a2a_egress_bw_bytes_per_s,
    measured_ring_busbw_bytes_per_s,
    measured_tables_cover,
)
from .configurations import CrossDCConfig
from .data import Size
from .hardware import MachineSpec
from .model_3d import ParallelConfig

# NCCL Protocol Constants
# Based on NCCL source analysis and empirical benchmarks
# Reference: https://github.com/NVIDIA/nccl/issues/281

# NCCL Protocol Thresholds (from NCCL source)
NCCL_LL_THRESHOLD = 32 * 1024  # 32KB - Low Latency protocol threshold
NCCL_LL128_THRESHOLD = 1 * 1024 * 1024  # 1MB - LL128 protocol threshold
NCCL_CHUNK_SIZE = 512 * 1024  # 512KB - Default chunk size for message pipelining

# Protocol bandwidth efficiency factors. NCCL's per-protocol LINE efficiency
# (src/graph/tuning.cc) times a protocol-independent residual ring/staging factor
# (~0.90) that NCCL's per-GPU busBw caps also impose — NOT hand-tuned benchmark fits:
#   line efficiency (bytes carried per wire byte):
#     LL:     4 useful bytes per 8-byte flit  -> 4/8   = 0.500
#             (NCCL: `busBw = std::min(llMaxBw, busBw * .5)`)
#     LL128:  120 useful bytes per 128-byte line -> 120/128 = 0.9375
#             (NCCL comment: `0.92 /*120.0/128.0*/`)
#     Simple: no fractional line overhead -> 1.0
#   effective = line * 0.90 residual, preserving the physical ordering
#     LL(0.45) < LL128(0.84) < Simple(0.90):
#       LL:     already caps below its line rate in NCCL (llMaxBw), so we keep the
#               raw 0.50 line value rather than re-discount it (0.50, conservative).
#       LL128:  0.9375 * 0.90 = 0.84375
#       Simple: 1.0    * 0.90 = 0.90
# Prior value LL128=0.70 was a guess and under-credited every 32KB..1MB message;
# 120/128 * residual is the physical value (not a fit to MFU). NOTE: all DP/TP
# buckets we currently model are multi-MB (Simple regime), so LL128 is latent —
# it only affects 32KB..1MB messages (small-model / high-TP configs).
NCCL_LL_BW_EFFICIENCY = 0.50
NCCL_LL128_BW_EFFICIENCY = 0.9375 * 0.90  # 120/128 line eff. * ring residual = 0.84375
NCCL_SIMPLE_BW_EFFICIENCY = 0.90

# Protocol latency multipliers (relative to base latency)
# Smaller messages incur higher latency overhead due to protocol headers
NCCL_LL_LATENCY_MULTIPLIER = 2.0
NCCL_LL128_LATENCY_MULTIPLIER = 1.5
NCCL_SIMPLE_LATENCY_MULTIPLIER = 1.0

# Chunk pipeline latency (empirical from profiling data)
# Each additional chunk adds a small pipeline overhead
NCCL_CHUNK_PIPELINE_LATENCY_US = 2.0


class ParallelismType(Enum):
    """Types of parallelism in hierarchical order."""

    PP = 0  # Pipeline Parallel
    DP = 1  # Data Parallel
    CP = 2  # Context Parallel
    TP = 3  # Tensor Parallel

    # Expert parallelism types
    EDP = 4  # Expert Data Parallel
    EP = 5  # Expert Parallel
    ETP = 6  # Expert Tensor Parallel


def get_nccl_protocol_overhead(message_size_bytes: int) -> tuple[float, float]:
    """
    Calculate NCCL protocol efficiency based on message size.

    NCCL uses different protocols depending on message size:
    - LL (Low Latency): <32KB, optimized for latency (50% line efficiency)
    - LL128: 32KB-1MB, balanced approach (120/128 = 93.75% line efficiency)
    - Simple: >1MB, optimized for bandwidth (~90% effective)

    Physics basis: Protocol selection is algorithmic, based on NCCL source code.
    Efficiency factors are NCCL's own per-protocol line efficiencies (see the
    module-level constants and src/graph/tuning.cc), not fits to MFU. This is a
    threshold simplification of NCCL's actual per-(algo,proto) cost-model
    minimization, adequate for the multi-MB ring collectives modeled here.

    Args:
        message_size_bytes: Size of the message in bytes

    Returns:
        Tuple of (bandwidth_efficiency, latency_multiplier)
        - bandwidth_efficiency: Factor to apply to raw bandwidth (0.0-1.0)
        - latency_multiplier: Factor to multiply base latency by
    """
    if message_size_bytes < NCCL_LL_THRESHOLD:
        return NCCL_LL_BW_EFFICIENCY, NCCL_LL_LATENCY_MULTIPLIER
    elif message_size_bytes < NCCL_LL128_THRESHOLD:
        return NCCL_LL128_BW_EFFICIENCY, NCCL_LL128_LATENCY_MULTIPLIER
    else:
        return NCCL_SIMPLE_BW_EFFICIENCY, NCCL_SIMPLE_LATENCY_MULTIPLIER


def calculate_chunking_overhead(message_size_bytes: int, base_latency: float) -> float:
    """
    Calculate additional latency from NCCL message chunking.

    NCCL splits large messages into chunks (typically 512KB) for pipelining.
    Each additional chunk adds a small pipeline latency overhead.

    Physics basis: Pipeline latency is inherent to chunked transmission.
    The chunk size is from NCCL configuration, and the per-chunk latency
    is measured from profiling traces.

    Args:
        message_size_bytes: Size of the message in bytes
        base_latency: Base latency in seconds (not used in current implementation
                      but kept for interface compatibility)

    Returns:
        Additional latency in seconds due to chunking overhead
    """
    n_chunks = max(1, message_size_bytes // NCCL_CHUNK_SIZE)
    # Each additional chunk adds pipeline latency
    # First chunk doesn't add overhead (already in base latency)
    return (n_chunks - 1) * NCCL_CHUNK_PIPELINE_LATENCY_US * 1e-6


def _get_effective_bw(
    *,
    parallelism_type: ParallelismType,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool,
) -> float:
    """Calculate effective bandwidth for a given parallelism type.

    Few things we need to figure out:

    Which kind of link the communication will utilize.
    -------------------------------------------------------------------------------------------
    Given some non-overlapping parallelism order like [PP, DP, CP, TP] or [PP, eDP, EP, eTP]
    if the product of parallelisms including and after the parallelism in question is less than
    or equal to the number of workers per node, then we will use intra-node links. Otherwise
    we will use inter-node links.

    How many participants will share the link.
    -------------------------------------------------------------------------------------------
    If the products of the parallelisms coming after the parallelism in question is greater than
    or equal to the number of workers per node, then the cross-node bandwidth is divided by the
    number of workers per node. Otherwise, it is divided by the number of workers per node divided
    product of following parallelisms.

    NOTE: we only account for link being shared across one communication type at a time. We do
    not account for cross-parallelism competition (like for example overlapped DP comms at the
    same time as cross-node EP comms).

    Args:
        parallelism_type: The type of parallelism for which to calculate bandwidth
        parallel_config: The parallelism configuration
        machine_spec: The machine specification
        is_expert_comm: Whether this is for expert parallelism communication

    Returns:
        Effective bandwidth in bytes per second
    """
    n_devices_per_node = machine_spec.n_devices

    if is_expert_comm:
        assert parallel_config.expert_mesh is not None
        parallelism_values = {
            ParallelismType.PP: parallel_config.pp,
            ParallelismType.EDP: parallel_config.expert_mesh.dp,
            ParallelismType.EP: parallel_config.expert_mesh.ep,
            ParallelismType.ETP: parallel_config.expert_mesh.tp,
        }
        hierarchy = [
            ParallelismType.PP,
            ParallelismType.EDP,
            ParallelismType.EP,
            ParallelismType.ETP,
        ]
    else:
        parallelism_values = {
            ParallelismType.PP: parallel_config.pp,
            ParallelismType.DP: parallel_config.dp,
            ParallelismType.CP: parallel_config.cp,
            ParallelismType.TP: parallel_config.tp,
        }
        hierarchy = [ParallelismType.PP, ParallelismType.DP, ParallelismType.CP, ParallelismType.TP]

    if parallelism_type not in hierarchy:
        raise AssertionError(
            f"Parallelism type {parallelism_type} not found in hierarchy {hierarchy}."
        )

    product_including_current = parallelism_values[parallelism_type]
    product_after_current = 1
    for parallelism_type in hierarchy[hierarchy.index(parallelism_type) + 1 :]:
        product_including_current *= parallelism_values[parallelism_type]
        product_after_current *= parallelism_values[parallelism_type]

    if product_including_current <= n_devices_per_node:  # use intra-node link (assume no sharing)
        return machine_spec.intra_node_connect.unidirectional_bw_bytes_per_sec

    # otherwise, use inter-node link
    base_bw = machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec

    return base_bw / n_devices_per_node


def get_tp_reduce_scatter_comm_time_s(
    size: Size, parallel_config: ParallelConfig, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size,
        n_participants=parallel_config.tp,
        machine_spec=machine_spec,
        parallel_config=parallel_config,
        is_expert_comm=False,
    )


def get_tp_all_gather_comm_time_s(
    size: Size, parallel_config: ParallelConfig, machine_spec: MachineSpec
) -> float:
    """assumes ring algorithm."""
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size,
        n_participants=parallel_config.tp,
        machine_spec=machine_spec,
        parallel_config=parallel_config,
        is_expert_comm=False,
    )


def _dp_group(
    parallel_config: ParallelConfig, is_expert_comm: bool
) -> tuple[int, "ParallelismType"]:
    """Resolve the (n_participants, ParallelismType) for a DP-family collective.

    A MoE step has TWO distinct DP reductions with different replica-group sizes:
      * expert params  -> reduced over expert_mesh.dp   (is_expert_comm=True)
      * every other param -> reduced over the full dp    (is_expert_comm=False)
    Previously the DP funcs used expert_mesh.dp for ALL params ("assume most params
    are MoE"), which under-counts (and, when expert_dp==1, zeroes) the dense-param
    reduction. Callers now select the group explicitly per parameter partition.
    """
    if is_expert_comm:
        assert parallel_config.expert_mesh is not None
        return parallel_config.expert_mesh.dp, ParallelismType.EDP
    return parallel_config.dp, ParallelismType.DP


def get_dp_reduce_scatter_latency_term_s(
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """assumes ring algorithm."""
    n_participants, _ = _dp_group(parallel_config, is_expert_comm)
    return _ring_ag_or_rs_latency_term_s(
        n_participants=n_participants,
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def _dp_group_is_strided(parallel_config: ParallelConfig, machine_spec: MachineSpec) -> bool:
    """True when the EXPERT-DP replica group has one rank per node (or sparser).

    Mesh order is [PP, eDP, EP, eTP], so expert-DP group members are spaced
    ep*etp ranks apart. When that stride >= GPUs/node, every ring hop crosses
    EFA on the GPU's own NIC rail (measured plateau ~47 GB/s busbw) instead of
    threading NVLink inside nodes like a contiguous dense-DP ring (~350 GB/s).
    """
    assert parallel_config.expert_mesh is not None
    stride = parallel_config.expert_mesh.ep * parallel_config.expert_mesh.tp
    return stride >= machine_spec.n_devices


def _measured_dp_busbw(
    collective: str,
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool,
    n_participants: int,
) -> float | None:
    """Measured ring busbw for this DP collective, or None off the measured SKU."""
    if not measured_tables_cover(machine_spec.name):
        return None
    strided = is_expert_comm and _dp_group_is_strided(parallel_config, machine_spec)
    return measured_ring_busbw_bytes_per_s(
        collective=collective,
        group_size=n_participants,
        payload_bytes=size.bytes(),
        strided=strided,
    )


def get_dp_reduce_scatter_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """assumes ring algorithm."""
    n_participants, ptype = _dp_group(parallel_config, is_expert_comm)
    if n_participants <= 1:
        return 0.0
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=n_participants,
        unidirectional_link_bw_bytes_per_sec=int(
            _get_effective_bw(
                parallelism_type=ptype,
                parallel_config=parallel_config,
                machine_spec=machine_spec,
                is_expert_comm=is_expert_comm,
            )
        ),
        measured_busbw_bytes_per_s=_measured_dp_busbw(
            "reduce_scatter", size, parallel_config, machine_spec,
            is_expert_comm, n_participants,
        ),
    )


def get_dp_reduce_scatter_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """assumes ring algorithm."""
    n_participants, _ = _dp_group(parallel_config, is_expert_comm)
    if n_participants <= 1:
        # single replica: no gradient reduction over this group
        return 0.0

    latency_term = get_dp_reduce_scatter_latency_term_s(
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    )

    bw_term = get_dp_reduce_scatter_bw_term_s(
        size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    )

    return latency_term + bw_term


def get_dp_all_gather_latency_term_s(
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """assumes ring algorithm."""
    n_participants, _ = _dp_group(parallel_config, is_expert_comm)
    return _ring_ag_or_rs_latency_term_s(
        n_participants=n_participants,
        link_latency_s=machine_spec.inter_node_connect.latency_sec,
    )


def get_dp_all_gather_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """assumes ring algorithm."""
    n_participants, ptype = _dp_group(parallel_config, is_expert_comm)
    if n_participants <= 1:
        return 0.0
    return _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=n_participants,
        unidirectional_link_bw_bytes_per_sec=int(
            _get_effective_bw(
                parallelism_type=ptype,
                parallel_config=parallel_config,
                machine_spec=machine_spec,
                is_expert_comm=is_expert_comm,
            )
        ),
        measured_busbw_bytes_per_s=_measured_dp_busbw(
            "all_gather", size, parallel_config, machine_spec,
            is_expert_comm, n_participants,
        ),
    )


def get_dp_all_gather_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """assumes ring algorithm."""
    n_participants, _ = _dp_group(parallel_config, is_expert_comm)
    if n_participants <= 1:
        return 0.0

    latency_term = get_dp_all_gather_latency_term_s(
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    )

    bw_term = get_dp_all_gather_bw_term_s(
        size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    )

    return latency_term + bw_term


def get_dp_all_reduce_latency_term_s(
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """Ring all-reduce latency term.

    Ring all-reduce is implemented as reduce-scatter followed by all-gather,
    so latency term is 2 * (n_participants - 1) hops.
    """
    return 2.0 * get_dp_reduce_scatter_latency_term_s(
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    )


def get_dp_all_reduce_bw_term_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """Ring all-reduce bandwidth term.

    Ring all-reduce transfers (2 * (N-1)/N) * size_bytes per participant in total
    (one pass of reduce-scatter + one pass of all-gather on the same-sized tensor).
    On the measured SKU we use the DIRECTLY measured all_reduce busbw curve
    (NCCL may pick tree/NVLS rather than ring — measured AR busbw is up to
    1.8x the RS busbw at n=16, so 2x-the-RS-term would over-charge).
    """
    n_participants, _ = _dp_group(parallel_config, is_expert_comm)
    if n_participants <= 1:
        return 0.0
    measured = _measured_dp_busbw(
        "all_reduce", size, parallel_config, machine_spec, is_expert_comm, n_participants
    )
    if measured is not None:
        # t = 2*(n-1)/n * bytes / busbw
        return 2.0 * (n_participants - 1) / n_participants * size.bytes() / measured
    return 2.0 * get_dp_reduce_scatter_bw_term_s(
        size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    )


def get_dp_all_reduce_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    is_expert_comm: bool = False,
) -> float:
    """Ring all-reduce time = latency term + bandwidth term.

    Used when zero_level=NONE (use_distributed_optimizer=false) where gradients
    are all-reduced across the DP dimension rather than reduce-scatter + all-gather.
    """
    n_participants, _ = _dp_group(parallel_config, is_expert_comm)
    if n_participants <= 1:
        return 0.0
    return get_dp_all_reduce_latency_term_s(
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    ) + get_dp_all_reduce_bw_term_s(
        size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        is_expert_comm=is_expert_comm,
    )


def get_cross_dc_dp_all_reduce_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    cross_dc_config: CrossDCConfig,
) -> float:
    """Cross-DC all-reduce = cross-DC RS + cross-DC AG on same tensor."""
    return get_cross_dc_dp_reduce_scatter_comm_time_s(
        size=size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        cross_dc_config=cross_dc_config,
    ) + get_cross_dc_dp_all_gather_comm_time_s(
        size=size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        cross_dc_config=cross_dc_config,
    )


def _ring_ag_or_rs_latency_term_s(
    n_participants: int,
    link_latency_s: float,
    message_size_bytes: int | None = None,
) -> float:
    """
    Calculate ring all-gather/reduce-scatter latency term with NCCL protocol overhead.

    Args:
        n_participants: Number of participants in the ring
        link_latency_s: Base link latency in seconds
        message_size_bytes: Message size in bytes for protocol selection (optional)

    Returns:
        Total latency term in seconds
    """
    base_latency = (n_participants - 1) * link_latency_s

    if message_size_bytes is not None:
        # Apply NCCL protocol latency multiplier
        _, latency_mult = get_nccl_protocol_overhead(message_size_bytes)
        # Add chunking overhead
        chunking_overhead = calculate_chunking_overhead(message_size_bytes, link_latency_s)
        return base_latency * latency_mult + chunking_overhead
    else:
        # Fallback to original behavior when message size not provided
        return base_latency


def _ring_ag_or_rs_bw_term_s(
    size: Size,
    n_participants: int,
    # NOTE: assumes duplex bw = 2x unidirectional
    unidirectional_link_bw_bytes_per_sec: int,
    measured_busbw_bytes_per_s: float | None = None,
) -> float:
    """
    Calculate ring all-gather/reduce-scatter bandwidth term with NCCL protocol efficiency.

    Args:
        size: Data size to communicate
        n_participants: Number of participants in the ring
        unidirectional_link_bw_bytes_per_sec: Unidirectional link bandwidth
        measured_busbw_bytes_per_s: If given, a MEASURED ring bus bandwidth
            (bytes*(n-1)/n / t) from comm_bench_util — already includes
            protocol/channel efficiency, so no analytic factor is applied.

    Returns:
        Total bandwidth term in seconds
    """
    phase_sent_data_size_bytes = int(size.bytes() / n_participants)

    if measured_busbw_bytes_per_s is not None:
        # busbw is defined so that t = (n-1)/n * bytes / busbw, which equals
        # the per-phase form below with effective_bw = busbw.
        effective_bw = measured_busbw_bytes_per_s
    else:
        # Apply NCCL protocol bandwidth efficiency
        message_size_bytes = int(size.bytes())
        bw_efficiency, _ = get_nccl_protocol_overhead(message_size_bytes)
        effective_bw = unidirectional_link_bw_bytes_per_sec * bw_efficiency

    return (n_participants - 1) * (phase_sent_data_size_bytes / effective_bw)


def _get_ring_tp_ag_or_rs_comm_time_s(
    size: Size,
    n_participants: int,
    machine_spec: MachineSpec,
    parallel_config: ParallelConfig,
    is_expert_comm: bool = False,
) -> float:
    # A collective over a single rank is a no-op: nothing to gather/scatter, and
    # no peer to incur latency against. Without this guard the latency term is
    # still charged at tp==1 / expert_tp==1 (every golden-set config is tp1), a
    # phantom ~0.4-1.2ms per SP all-gather/reduce-scatter folded into compute.
    if n_participants <= 1:
        return 0.0
    lat_term_s = _ring_ag_or_rs_latency_term_s(
        n_participants=n_participants,
        # TODO. need to fix
        link_latency_s=machine_spec.intra_node_connect.latency_sec,
        message_size_bytes=int(size.bytes()),
    )
    bw_term_s = _ring_ag_or_rs_bw_term_s(
        size,
        n_participants=n_participants,
        unidirectional_link_bw_bytes_per_sec=int(
            _get_effective_bw(
                parallelism_type=ParallelismType.ETP if is_expert_comm else ParallelismType.TP,
                parallel_config=parallel_config,
                machine_spec=machine_spec,
                is_expert_comm=is_expert_comm,
            )
        ),
    )

    return bw_term_s + lat_term_s


def get_all_to_all_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    assert parallel_config.expert_mesh is not None

    alltoall_n_nodes = parallel_config.expert_mesh.ep // (
        machine_spec.n_devices // parallel_config.expert_mesh.tp
    )

    lat_term_s = (
        machine_spec.inter_node_connect.latency_sec
        if alltoall_n_nodes > 1
        else machine_spec.intra_node_connect.latency_sec
    )

    n_participants = parallel_config.expert_mesh.ep
    if n_participants <= 1:
        # single expert group: dispatch/combine is a local permutation, no A2A.
        return 0.0

    # Measured path (B200): NCCL all_to_all_single runs as (n-1) point-to-point
    # copies per rank, which does NOT reach the NVSwitch/EFA link peak the
    # analytic branch assumes (measured 8-rank NVSwitch egress ~550 GB/s vs the
    # 900 peak at large buffers, with a strong size ramp below ~64MB; inter-node
    # 16/32-rank ~85/57 GB/s). See comm_bench_util + benchmarks/a2a_benchmark.py;
    # trace cross-check: 18b runs' In-msg sizes/p50s sit on this curve within
    # the in-workload desync margin.
    if measured_tables_cover(machine_spec.name):
        egress_bytes = size.bytes() * (n_participants - 1) / n_participants
        bw = measured_a2a_egress_bw_bytes_per_s(
            group_size=n_participants, buffer_bytes=size.bytes()
        )
        return lat_term_s + egress_bytes / bw

    # we need to calculate the percentage of the comms that occur over the slowest link
    # type, and only calculate the comm cost for the percentage of the message that
    # travels over the slowest links.
    #
    # this is kind of a special situation for all-to-all. the other comm types predominantly
    # use rings, where the slowest links act as a dam that rate limit the rest.
    if alltoall_n_nodes > 1:
        alltoall_internode_fraction = (alltoall_n_nodes - 1) / alltoall_n_nodes
        bw_term_s = (
            (size.bytes() / n_participants) * alltoall_internode_fraction * (n_participants - 1)
        ) / (
            machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec / machine_spec.n_devices
        )
    else:
        bw_term_s = (
            (size.bytes() / n_participants) * (n_participants - 1)
        ) / machine_spec.intra_node_connect.unidirectional_bw_bytes_per_sec

    return lat_term_s + bw_term_s


def get_expert_tp_all_gather_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """All-gather for expert tensor parallelism. Uses expert_tp degree."""
    if not parallel_config.expert_mesh:
        return 0.0
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size,
        n_participants=parallel_config.expert_mesh.tp,
        machine_spec=machine_spec,
        parallel_config=parallel_config,
        is_expert_comm=True,
    )


def get_expert_tp_reduce_scatter_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
) -> float:
    """Reduce-scatter for expert tensor parallelism. Uses expert_tp degree."""
    if not parallel_config.expert_mesh:
        return 0.0
    return _get_ring_tp_ag_or_rs_comm_time_s(
        size,
        n_participants=parallel_config.expert_mesh.tp,
        machine_spec=machine_spec,
        parallel_config=parallel_config,
        is_expert_comm=True,
    )


def get_cross_dc_dp_all_gather_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    cross_dc_config: CrossDCConfig,
) -> float:
    """Calculate cross-DC data parallel all-gather time.

    Args:
        size: Size of data to communicate
        parallel_config: Parallelism configuration
        machine_spec: Machine/hardware specification
        cross_dc_config: Cross-DC configuration

    Returns:
        Time in seconds for cross-DC all-gather
    """
    return _get_cross_dc_dp_all_gather_or_reduce_scatter_comm_time_s(
        size=size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        cross_dc_config=cross_dc_config,
    )


def get_cross_dc_dp_reduce_scatter_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    cross_dc_config: CrossDCConfig,
) -> float:
    """Calculate cross-DC data parallel reduce-scatter time.

    Args:
        size: Size of data to communicate
        parallel_config: Parallelism configuration
        machine_spec: Machine/hardware specification
        cross_dc_config: Cross-DC configuration

    Returns:
        Time in seconds for cross-DC reduce-scatter
    """
    return _get_cross_dc_dp_all_gather_or_reduce_scatter_comm_time_s(
        size=size,
        parallel_config=parallel_config,
        machine_spec=machine_spec,
        cross_dc_config=cross_dc_config,
    )


def _get_cross_dc_dp_all_gather_or_reduce_scatter_comm_time_s(
    size: Size,
    parallel_config: ParallelConfig,
    machine_spec: MachineSpec,
    cross_dc_config: CrossDCConfig,
) -> float:
    if parallel_config.expert_mesh is None:
        dp_degree = parallel_config.dp
    else:
        # we'll just use the expert DP case to approximate everything as
        # it'll be the most taxing (most simultaneous rings)
        dp_degree = parallel_config.expert_mesh.dp
    n_dp_rings = parallel_config.world_size() // dp_degree
    cross_dc_bw_per_ring = cross_dc_config.interconnect_bandwidth_bytes_per_sec() / n_dp_rings

    # cross-DC DP means creating heterogeneous rings, where the majority of links
    # are fast(er) inter-node links, and some of them are slower cross-DC links.
    # we use the lowest BW link when calculating how long ring collectives take.
    # this is because we expect the slowest link to behave like a dam, and
    # rate limit any faster links that come after it.
    effective_bw = min(
        cross_dc_bw_per_ring,
        machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec / machine_spec.n_devices,
    )
    bw_term = _ring_ag_or_rs_bw_term_s(
        size=size,
        n_participants=dp_degree,
        unidirectional_link_bw_bytes_per_sec=int(effective_bw),
    )

    # in a ring, there are (dp_degree - 1) total hops
    # n_dcs of these are inter-DC hops, the rest are intra-DC hops
    inter_dc_latency_term = cross_dc_config.n_dcs * cross_dc_config.interconnect_latency_s
    intra_dc_latency_term = (dp_degree - 1 - cross_dc_config.n_dcs) * (
        machine_spec.inter_node_connect.latency_sec
    )
    latency_term = inter_dc_latency_term + intra_dc_latency_term

    return latency_term + bw_term
