"""Unit tests for DP AllReduce comm modeling (zero_level=NONE case).

Covers the path used when `use_distributed_optimizer=false`, where gradients
are all-reduced across the DP dimension at each step. Motivated by a
component-level comparison against measured training runs, which showed this
path was the dominant source of DP-comm error.
"""

from dlcalc.utils.comms import (
    get_cross_dc_dp_all_gather_comm_time_s,
    get_cross_dc_dp_all_reduce_comm_time_s,
    get_cross_dc_dp_reduce_scatter_comm_time_s,
    get_dp_all_gather_comm_time_s,
    get_dp_all_reduce_bw_term_s,
    get_dp_all_reduce_comm_time_s,
    get_dp_all_reduce_latency_term_s,
    get_dp_reduce_scatter_comm_time_s,
)
from dlcalc.utils.configurations import CrossDCConfig
from dlcalc.utils.data import Size
from dlcalc.utils.hardware import MachineSpec
from dlcalc.utils.model_3d import ParallelConfig


def _make_parallel_config(
    dp: int,
    zero_level: ParallelConfig.ZeroLevel = ParallelConfig.ZeroLevel.NONE,
    expert_mesh: ParallelConfig.ExpertParallelCfg | None = None,
) -> ParallelConfig:
    return ParallelConfig(
        tp=1,
        cp=1,
        pp=1,
        dp=dp,
        expert_mesh=expert_mesh,
        vpp=1,
        sp_enabled=True,
        zero_level=zero_level,
    )


class TestDPAllReduceBasic:
    """Algebraic properties of the ring all-reduce model."""

    def test_all_reduce_latency_is_twice_reduce_scatter(self):
        """Ring all-reduce = RS phase + AG phase, so latency term is 2x RS."""
        from dlcalc.utils.comms import get_dp_reduce_scatter_latency_term_s

        parallel_cfg = _make_parallel_config(dp=32)
        machine_spec = MachineSpec.from_str("p5.48xlarge")

        ar_lat = get_dp_all_reduce_latency_term_s(
            parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        rs_lat = get_dp_reduce_scatter_latency_term_s(
            parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        assert ar_lat == 2.0 * rs_lat

    def test_all_reduce_bw_is_twice_reduce_scatter(self):
        """Ring all-reduce BW term is 2x RS BW term on the same tensor."""
        from dlcalc.utils.comms import get_dp_reduce_scatter_bw_term_s

        parallel_cfg = _make_parallel_config(dp=32)
        machine_spec = MachineSpec.from_str("p5.48xlarge")
        size = Size(numel=100_000_000, bits_per_element=32)

        ar_bw = get_dp_all_reduce_bw_term_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        rs_bw = get_dp_reduce_scatter_bw_term_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        assert ar_bw == 2.0 * rs_bw

    def test_all_reduce_matches_rs_plus_ag(self):
        """At the algorithm level, ring AR time ≈ RS time + AG time on same tensor."""
        parallel_cfg = _make_parallel_config(dp=16)
        machine_spec = MachineSpec.from_str("p5.48xlarge")
        size = Size(numel=50_000_000, bits_per_element=32)

        ar_time = get_dp_all_reduce_comm_time_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        rs_time = get_dp_reduce_scatter_comm_time_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        ag_time = get_dp_all_gather_comm_time_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        # Note: RS_time and AG_time each include their own latency term, so RS+AG
        # should equal AR (since AR latency is also 2x RS latency).
        assert abs(ar_time - (rs_time + ag_time)) < 1e-9

    def test_all_reduce_positive(self):
        parallel_cfg = _make_parallel_config(dp=32)
        machine_spec = MachineSpec.from_str("p5.48xlarge")
        size = Size(numel=1_000_000, bits_per_element=32)
        t = get_dp_all_reduce_comm_time_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        assert t > 0

    def test_all_reduce_scales_with_size(self):
        """Doubling payload size should roughly double the bandwidth term."""
        parallel_cfg = _make_parallel_config(dp=32)
        machine_spec = MachineSpec.from_str("p5.48xlarge")

        small_bw = get_dp_all_reduce_bw_term_s(
            Size(numel=1_000_000, bits_per_element=32),
            parallel_config=parallel_cfg,
            machine_spec=machine_spec,
        )
        large_bw = get_dp_all_reduce_bw_term_s(
            Size(numel=2_000_000, bits_per_element=32),
            parallel_config=parallel_cfg,
            machine_spec=machine_spec,
        )
        # Allow small numerical slack from protocol selection transitions.
        assert 1.8 * small_bw <= large_bw <= 2.2 * small_bw


class TestDPAllReduceEdgeCases:
    """Boundary behavior."""

    def test_dp_equals_one_bw_is_zero(self):
        """DP=1 (no DP): ring has (N-1)=0 hops, bandwidth term = 0."""
        parallel_cfg = _make_parallel_config(dp=1)
        machine_spec = MachineSpec.from_str("p5.48xlarge")
        size = Size(numel=1_000_000, bits_per_element=32)
        bw = get_dp_all_reduce_bw_term_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        assert bw == 0.0

    def test_dp_equals_one_latency_is_zero(self):
        parallel_cfg = _make_parallel_config(dp=1)
        machine_spec = MachineSpec.from_str("p5.48xlarge")
        lat = get_dp_all_reduce_latency_term_s(
            parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        assert lat == 0.0

    def test_dp_group_selected_by_is_expert_comm(self):
        """DP collectives select the ring group via is_expert_comm.

        This is the Effect-B fix: expert params reduce over expert_mesh.dp, every
        other param over the full dp. The DEFAULT (is_expert_comm=False) must use
        the dense dp — previously it wrongly used expert_mesh.dp for all params,
        which under-counts (and, when expert_dp==1, zeroes) the dense reduction.
        """
        # world = tp*pp*cp*dp = 32; expert mesh satisfies ep*etp*edp = 32
        expert_mesh = ParallelConfig.ExpertParallelCfg(ep=8, tp=1, dp=4)
        parallel_cfg = _make_parallel_config(dp=32, expert_mesh=expert_mesh)
        machine_spec = MachineSpec.from_str("p5.48xlarge")
        size = Size(numel=1_000_000, bits_per_element=32)

        # Default (dense) group => uses dp=32, matching a plain dp=32 config.
        t_dense = get_dp_all_reduce_comm_time_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        t_plain_32 = get_dp_all_reduce_comm_time_s(
            size, parallel_config=_make_parallel_config(dp=32), machine_spec=machine_spec
        )
        assert t_dense == t_plain_32, "default DP comm must use the dense dp group"

        # Expert group => uses expert_mesh.dp=4, differs from the dense group.
        t_expert = get_dp_all_reduce_comm_time_s(
            size,
            parallel_config=parallel_cfg,
            machine_spec=machine_spec,
            is_expert_comm=True,
        )
        assert t_expert != t_dense, "is_expert_comm=True must use expert_mesh.dp, not dense dp"
        assert t_expert > 0

    def test_expert_dp_equals_one_zeroes_only_expert_group(self):
        """When expert_dp==1, the expert-group reduction is 0 (single replica),
        but the dense-group reduction over dp>1 is still non-zero — the exact
        regression that previously zeroed all DP comm for these configs."""
        # ep8, edp=1 (dp*... = ep) but dense dp=8
        expert_mesh = ParallelConfig.ExpertParallelCfg(ep=8, tp=1, dp=1)
        parallel_cfg = _make_parallel_config(dp=8, expert_mesh=expert_mesh)
        machine_spec = MachineSpec.from_str("p6-b200.48xlarge")
        size = Size(numel=1_000_000, bits_per_element=32)

        t_expert = get_dp_all_reduce_comm_time_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec, is_expert_comm=True
        )
        t_dense = get_dp_all_reduce_comm_time_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec, is_expert_comm=False
        )
        assert t_expert == 0.0, "expert_dp==1 => no expert-group reduction"
        assert t_dense > 0.0, "dense params over dp=8 must still be reduced"


class TestCrossDCAllReduce:
    """Cross-DC variant = cross-DC RS + cross-DC AG."""

    def test_cross_dc_ar_equals_cross_dc_rs_plus_ag(self):
        parallel_cfg = _make_parallel_config(dp=16)
        machine_spec = MachineSpec.from_str("p5.48xlarge")
        cross_dc_cfg = CrossDCConfig(
            n_dcs=3,
            interconnect_bandwidth_gbps=800,
            interconnect_latency_s=0.0035,
        )
        size = Size(numel=10_000_000, bits_per_element=32)

        ar = get_cross_dc_dp_all_reduce_comm_time_s(
            size=size,
            parallel_config=parallel_cfg,
            machine_spec=machine_spec,
            cross_dc_config=cross_dc_cfg,
        )
        rs = get_cross_dc_dp_reduce_scatter_comm_time_s(
            size=size,
            parallel_config=parallel_cfg,
            machine_spec=machine_spec,
            cross_dc_config=cross_dc_cfg,
        )
        ag = get_cross_dc_dp_all_gather_comm_time_s(
            size=size,
            parallel_config=parallel_cfg,
            machine_spec=machine_spec,
            cross_dc_config=cross_dc_cfg,
        )
        assert abs(ar - (rs + ag)) < 1e-9


class TestMeasuredWorkloadValidation:
    """End-to-end sanity: the 5p3b measured numbers should be in-range.

    This is not a strict regression test against 1770.5 ms (too many dimensions
    depend on hardware-specific BW), but it asserts the measured value lives
    inside the model's plausibility envelope.
    """

    def test_grad_bucket_in_envelope(self):
        # 5p3b config: dense DP=256, grad bucket ≈ 1.22B params at fp32,
        # on P6-B200 with expert DP = 32. Measured per-bucket AR ≈ 1770ms/8
        # buckets ≈ 221 ms.
        expert_mesh = ParallelConfig.ExpertParallelCfg(ep=8, tp=1, dp=32)
        parallel_cfg = ParallelConfig(
            tp=1,
            cp=1,
            pp=2,
            dp=256,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.NONE,
        )
        machine_spec = MachineSpec.from_str("p6-b200.48xlarge")

        # 1.22B params * 4 bytes = ~4.88 GB
        # That's 1.22B fp32 grads per bucket across 8 buckets.
        size = Size(numel=1_220_000_000, bits_per_element=32)

        per_bucket_s = get_dp_all_reduce_comm_time_s(
            size, parallel_config=parallel_cfg, machine_spec=machine_spec
        )
        # The 221 ms reference is an IN-WORKLOAD trace bucket (wire + desync
        # wait). The model predicts the straggler-free wire time, which the
        # measured B200 AR busbw curve (dp_collectives_b200_4node.parquet,
        # ~375 GB/s busbw at n=32/2GB; ring busbw n-invariant 16->32 within 2%)
        # puts at ~41 ms for this 4.88 GB bucket. Envelope: wire prediction must
        # be positive, below the contaminated bucket, and not absurdly small.
        assert 0.01 < per_bucket_s < 0.25, (
            f"AllReduce per-bucket prediction {per_bucket_s * 1000:.2f} ms "
            "is outside the plausibility envelope (measured in-workload bucket 221 ms "
            "= wire + desync; wire component ~41 ms from measured busbw)"
        )
