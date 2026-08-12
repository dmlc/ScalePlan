"""Unit tests for the autograd-style backward pass timing model."""

from unittest.mock import MagicMock

import pytest

from dlcalc.utils.backward import (
    compute_glu_bwd_time_s,
    compute_grouped_linear_bwd_time_s,
    compute_layernorm_bwd_time_s,
    compute_linear_bwd_time_s,
    compute_permutation_bwd_time_s,
    compute_residual_bwd_time_s,
    compute_rope_bwd_time_s,
    compute_sdpa_bwd_time_s,
    compute_topk_bwd_time_s,
)
from dlcalc.utils.data import Size
from dlcalc.utils.hardware import DType, MachineSpec


@pytest.fixture
def fake_spec(monkeypatch) -> MachineSpec:
    """A fake MachineSpec whose peak FLOPs, mem bandwidth, and GEMM-util lookups
    are controlled by the test so we can assert on exact time values."""
    spec = MagicMock(spec=MachineSpec)
    spec.name = "h100_test"
    spec.device_spec = MagicMock()
    spec.device_spec.peak_flops = MagicMock(return_value=1e12)  # 1 TFLOP/s
    spec.device_spec.mem_bandwidth_bytes_per_sec = int(1e9)  # 1 GB/s

    # Force GEMM utilization lookups to always return 1.0 so time = FLOPs / peak_flops.
    monkeypatch.setattr(
        "dlcalc.utils.backward.get_gemm_utilization_or_default",
        lambda **kwargs: 1.0,
    )
    return spec


class TestComputeLinearBwdTime:
    """Linear backward executes 2 GEMMs of the same FLOP count as the forward."""

    def test_two_gemms_double_fwd_flops(self, fake_spec: MachineSpec) -> None:
        # Forward: Y = X @ W with X=(M,K), W=(K,N). FLOPs = 2*M*N*K
        # Backward dX = dY @ W.T: FLOPs = 2*M*N*K
        # Backward dW = X.T @ dY: FLOPs = 2*M*N*K
        # Total backward = 4*M*N*K FLOPs
        m, k, n = 1024, 512, 2048
        time_s = compute_linear_bwd_time_s(
            n_tokens=m,
            weight_shape=(k, n),
            machine_spec=fake_spec,
            dtype=DType.BF16,
        )
        expected_flops = 4 * m * n * k
        expected_time = expected_flops / 1e12
        assert abs(time_s - expected_time) < 1e-12


class TestComputeGroupedLinearBwdTime:
    """Grouped (MoE) linear backward: FLOPs are the aggregate over local groups
    (M_eff = n_tokens_per_group * n_groups) but utilization is looked up at the
    PER-GROUP tile M (= n_tokens_per_group), since the grouped kernel's efficiency
    is set by the per-group tile (measured on B200)."""

    def test_flops_scale_with_total_tokens_at_fixed_util(self, fake_spec: MachineSpec) -> None:
        # With util pinned to 1.0 (fake_spec), backward time == FLOPs / peak.
        # Total backward FLOPs = 4 * M_eff * N * K, so at fixed n_tokens_per_group
        # doubling n_groups doubles M_eff and thus the time.
        m_per_group, k, n = 256, 512, 1024
        t1 = compute_grouped_linear_bwd_time_s(
            n_tokens_per_group=m_per_group,
            n_groups=1,
            weight_shape=(k, n),
            machine_spec=fake_spec,
            dtype=DType.BF16,
        )
        t8 = compute_grouped_linear_bwd_time_s(
            n_tokens_per_group=m_per_group,
            n_groups=8,
            weight_shape=(k, n),
            machine_spec=fake_spec,
            dtype=DType.BF16,
        )
        assert abs(t8 / t1 - 8.0) < 1e-9

    def test_total_flops_match_single_grouped_gemm(self, fake_spec: MachineSpec) -> None:
        # At util=1.0 the time is exactly the FLOP-bound time for the aggregate
        # GEMM: 4 * M_eff * N * K / peak (2*M_eff*N*K each for dX and dW).
        m_per_group, n_groups, k, n = 256, 8, 512, 1024
        m_eff = m_per_group * n_groups
        t = compute_grouped_linear_bwd_time_s(
            n_tokens_per_group=m_per_group,
            n_groups=n_groups,
            weight_shape=(k, n),
            machine_spec=fake_spec,
            dtype=DType.BF16,
        )
        expected_time = (4 * m_eff * n * k) / 1e12
        assert abs(t - expected_time) < 1e-12

    def test_scales_with_n_local_at_fixed_capacity(self) -> None:
        """KEY REGRESSION: at a fixed PER-EXPERT capacity (the real EP-invariant in
        dropless MoE), the PER-RANK grouped backward time scales ~linearly with
        n_local (= n_experts / EP).

        Physics (measurement-validated on B200, benchmarks/gemm_grouped_benchmark.py):
        dropless capacity = seq*top_k/n_experts is INDEPENDENT of EP, so the
        per-group tile M — and hence the grouped kernel's utilization — is fixed at
        ~cap across EP. The per-rank FLOPs are the aggregate over the rank's
        n_local groups (M_eff_local = n_local * cap ∝ 1/EP... i.e. ∝ n_local), so
        the per-rank grouped time ∝ n_local at fixed util. The measured grouped
        kernel confirms this: at cap=192, fwd_ms goes 0.13→1.99ms as num_gemms goes
        4→128 (~linear), util only drifts 2.7→5.8% (see
        results/gemm_grouped_b200.parquet).

        The OLD model looked util up at the aggregate M_eff and so predicted a
        per-rank time that was FLAT in n_local — which over-credited util (and thus
        under-counted time) 5-11x in the dropless regime. This test pins that the
        per-rank time now tracks n_local at fixed capacity (util roughly constant).

        (Global expert compute IS ~EP-invariant — each rank does 1/EP of the work,
        EP ranks in parallel — but the MODEL times the PER-RANK grouped kernel,
        which is what the trace measures. The earlier "flat per-rank" invariant
        conflated the attention-dominated total meas_gemm_ms with the expert part.)
        """
        spec = MachineSpec.from_str("p6-b200.48xlarge")
        capacity = 192  # dropless cap = seq(8192)*top_k(3)/n_experts(128); EP-invariant
        n_experts = 128
        weight_shape = (2048, 2560)  # 700M expert up-proj (K=hidden, N=2*expert_inter)

        times = {}
        for ep in (1, 2, 4, 8):
            n_groups = n_experts // ep  # n_local
            times[ep] = compute_grouped_linear_bwd_time_s(
                n_tokens_per_group=capacity,  # per-group M fixed at cap (EP-invariant)
                n_groups=n_groups,
                weight_shape=weight_shape,
                machine_spec=spec,
                dtype=DType.BF16,
            )

        # Per-rank time scales ~linearly with n_local: halving EP (2x n_local)
        # ~doubles the per-rank grouped time (util fixed at the cap tile, FLOPs 2x).
        for ep in (2, 4, 8):
            ratio = times[ep // 2] / times[ep]
            assert abs(ratio - 2.0) < 1e-6, (
                f"per-rank grouped bwd should ~2x when EP halves (2x n_local) at "
                f"fixed capacity; EP {ep}->{ep // 2} ratio={ratio:.3f} times={times}"
            )
        # And it is emphatically NOT flat in EP (the old M_eff-lookup bug).
        assert times[1] / times[8] > 6.0, (
            f"per-rank expert grouped time must grow with n_local (EP1 has 8x the "
            f"n_local of EP8); got EP1/EP8={times[1] / times[8]:.2f} times={times}"
        )


class TestComputeSDPABwdTime:
    """Backward SDPA time is looked up from the measured parquet."""

    def test_lookup_returns_measured_time(self, monkeypatch) -> None:
        # Fake the lookup to return a specific measurement.
        monkeypatch.setattr(
            "dlcalc.utils.backward._lookup_sdpa_bwd_time_s",
            lambda **kwargs: 0.00123,  # 1.23 ms
        )
        spec = MagicMock(spec=MachineSpec)
        spec.name = "p4d.24xlarge"
        time_s = compute_sdpa_bwd_time_s(
            seqlen_per_cp=2048,
            seqlen_full=2048,
            head_dim=128,
            n_q_heads_local=64,
            n_kv_heads_local=8,
            micro_bs=1,
            machine_spec=spec,
            dtype=DType.BF16,
        )
        assert time_s == 0.00123

    def test_missing_data_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "dlcalc.utils.backward._lookup_sdpa_bwd_time_s",
            lambda **kwargs: None,
        )
        spec = MagicMock(spec=MachineSpec)
        spec.name = "unknown"
        with pytest.raises(RuntimeError, match="No measured SDPA backward"):
            compute_sdpa_bwd_time_s(
                seqlen_per_cp=2048,
                seqlen_full=2048,
                head_dim=128,
                n_q_heads_local=64,
                n_kv_heads_local=8,
                micro_bs=1,
                machine_spec=spec,
                dtype=DType.BF16,
            )


class TestLayerNormBwdTime:
    def test_mem_bound_five_transfers(self, fake_spec: MachineSpec) -> None:
        numel = 1024 * 1024
        time_s = compute_layernorm_bwd_time_s(numel=numel, machine_spec=fake_spec, dtype=DType.BF16)
        bytes_total = numel * 2 * 5  # bf16 (2 bytes), 5 transfers
        assert abs(time_s - bytes_total / 1e9) < 1e-9


class TestRoPEBwdTime:
    def test_mem_bound_three_transfers(self, fake_spec: MachineSpec) -> None:
        numel = 2 * 1024 * 1024
        time_s = compute_rope_bwd_time_s(numel=numel, machine_spec=fake_spec, dtype=DType.BF16)
        bytes_total = numel * 2 * 3
        assert abs(time_s - bytes_total / 1e9) < 1e-9


class TestResidualBwdTime:
    def test_three_transfers(self, fake_spec: MachineSpec) -> None:
        size = Size(numel=1024, bits_per_element=16)
        time_s = compute_residual_bwd_time_s(activation_size=size, machine_spec=fake_spec)
        expected_bytes = 3 * size.bytes()
        assert abs(time_s - expected_bytes / 1e9) < 1e-9


class TestGLUBwdTime:
    def test_five_transfers(self, fake_spec: MachineSpec) -> None:
        numel = 4 * 1024 * 1024
        time_s = compute_glu_bwd_time_s(numel=numel, machine_spec=fake_spec, dtype=DType.BF16)
        bytes_total = numel * 2 * 5
        assert abs(time_s - bytes_total / 1e9) < 1e-9


class TestPermutationBwdTime:
    def test_four_transfers(self, fake_spec: MachineSpec) -> None:
        n_tokens = 4096
        hidden = 4096
        time_s = compute_permutation_bwd_time_s(
            n_tokens=n_tokens,
            hidden_dim=hidden,
            machine_spec=fake_spec,
            dtype=DType.BF16,
        )
        bytes_total = n_tokens * hidden * 2 * 4
        assert abs(time_s - bytes_total / 1e9) < 1e-9


class TestTopKBwdTime:
    def test_scales_with_n_elements(self) -> None:
        spec = MagicMock(spec=MachineSpec)
        spec.name = "b200_test"  # picks up the B200 topk throughput
        t_small = compute_topk_bwd_time_s(batch=1, seqlen=1024, n_experts=32, machine_spec=spec)
        t_big = compute_topk_bwd_time_s(batch=1, seqlen=1024, n_experts=128, machine_spec=spec)
        assert abs(t_big / t_small - 4.0) < 1e-9

    def test_uses_measured_forward_throughput_not_stale_constant(self) -> None:
        """The backward must use the shared measured topk throughput, NOT the old
        ~100x-too-slow hardcoded 1.5e8 elem/s. Pins backward == n_elements /
        topk_throughput_elem_per_s (same table as the forward), so it can't drift
        back to a separate uncalibrated constant.

        Regression: the old b200=1.5e8 gave 6.99ms for the 5p3b block
        (1*8192*128 = 1.05M elem); the measured ~1.5e10 gives ~0.07ms.
        """
        from dlcalc.utils.moe_router_util import topk_throughput_elem_per_s

        spec = MachineSpec.from_str("p6-b200.48xlarge")
        n_elem = 1 * 8192 * 128
        t = compute_topk_bwd_time_s(batch=1, seqlen=8192, n_experts=128, machine_spec=spec)
        assert abs(t - n_elem / topk_throughput_elem_per_s(spec)) < 1e-15
        # Sanity: the corrected time is sub-millisecond, not the old ~7ms.
        assert t < 0.5e-3, f"topk bwd {t * 1000:.2f} ms — stale 100x-slow constant regressed?"

    def test_backward_matches_forward_throughput(self) -> None:
        """Backward and forward topk share one calibrated per-GPU throughput
        (measured within ~0.7-1.2x on B200), so at equal element counts the
        backward time equals the forward selection time."""
        from dlcalc.utils.moe_router_util import calculate_topk_time

        spec = MachineSpec.from_str("p6-b200.48xlarge")
        fwd = calculate_topk_time(batch=1, seqlen=8192, n_experts=128, k=3, machine_spec=spec)
        bwd = compute_topk_bwd_time_s(batch=1, seqlen=8192, n_experts=128, machine_spec=spec)
        assert abs(fwd - bwd) < 1e-15
