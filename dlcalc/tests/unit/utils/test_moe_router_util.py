"""Unit tests for moe_router_util.py utility functions.

This module tests MoE router operation modeling functions used to calculate
time for router GEMM, TopK selection, and token permutation/unpermutation.

Test Coverage:
- calculate_router_gemm_time: Router GEMM timing across different configurations
- calculate_topk_time: TopK selection timing for expert routing
- calculate_permutation_time: Token permutation/unpermutation timing
- calculate_total_moe_router_overhead: Combined router overhead
"""

from unittest.mock import MagicMock

import pytest

from dlcalc.utils.hardware import DType, MachineSpec
from dlcalc.utils.moe_router_util import (
    calculate_permutation_time,
    calculate_router_gemm_time,
    calculate_topk_time,
    calculate_total_moe_router_overhead,
)


@pytest.fixture
def h100_machine_spec() -> MachineSpec:
    """Create H100 machine specification for testing."""
    machine_spec = MagicMock(spec=MachineSpec)
    machine_spec.name = "h100_80gb"

    # Create nested device_spec
    device_spec = MagicMock()
    device_spec.mem_bandwidth_bytes_per_sec = 2000 * 1024 * 1024 * 1024  # 2 TB/s
    device_spec.peak_flops = MagicMock(return_value=1000e12)  # 1000 TFLOPs for BF16

    machine_spec.device_spec = device_spec
    return machine_spec


@pytest.fixture
def a100_machine_spec() -> MachineSpec:
    """Create A100 machine specification for testing."""
    machine_spec = MagicMock(spec=MachineSpec)
    machine_spec.name = "a100_80gb"

    # Create nested device_spec
    device_spec = MagicMock()
    device_spec.mem_bandwidth_bytes_per_sec = 1500 * 1024 * 1024 * 1024  # 1.5 TB/s
    device_spec.peak_flops = MagicMock(return_value=600e12)  # 600 TFLOPs for BF16

    machine_spec.device_spec = device_spec
    return machine_spec


class TestCalculateRouterGemmTime:
    """Test cases for the calculate_router_gemm_time function."""

    def test_small_model_h100(self, h100_machine_spec: MachineSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test router GEMM for small model on H100."""
        # Mock get_gemm_utilization_or_default to return fixed utilization
        def mock_gemm_util(*args: object, **kwargs: object) -> float:
            return 0.8  # 80% utilization

        monkeypatch.setattr(
            "dlcalc.utils.moe_router_util.get_gemm_utilization_or_default",
            mock_gemm_util,
        )

        # Small model: batch=1, seqlen=2048, hidden_dim=2048, n_experts=32
        time_s = calculate_router_gemm_time(
            batch=1,
            seqlen=2048,
            hidden_dim=2048,
            n_experts=32,
            machine_spec=h100_machine_spec,
            dtype=DType.BF16,
        )

        # M = 1 * 2048 = 2048
        # K = 2048
        # N = 32
        # FLOPs = 2 * 2048 * 32 * 2048 = 268.4M
        # Time = 268.4M / (1000T * 0.8) = 0.3355μs
        expected_flops = 2 * 2048 * 32 * 2048
        expected_time = expected_flops / (1000e12 * 0.8)
        assert abs(time_s - expected_time) < 1e-9

    def test_large_model_a100(self, a100_machine_spec: MachineSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test router GEMM for large model on A100."""

        def mock_gemm_util(*args: object, **kwargs: object) -> float:
            return 0.75

        monkeypatch.setattr(
            "dlcalc.utils.moe_router_util.get_gemm_utilization_or_default",
            mock_gemm_util,
        )

        # Large model: batch=1, seqlen=4096, hidden_dim=4096, n_experts=128
        time_s = calculate_router_gemm_time(
            batch=1,
            seqlen=4096,
            hidden_dim=4096,
            n_experts=128,
            machine_spec=a100_machine_spec,
            dtype=DType.BF16,
        )

        # M = 1 * 4096 = 4096
        # K = 4096
        # N = 128
        # FLOPs = 2 * 4096 * 128 * 4096 = 4.3B
        expected_flops = 2 * 4096 * 128 * 4096
        expected_time = expected_flops / (600e12 * 0.75)
        assert abs(time_s - expected_time) < 1e-9

    def test_varying_batch_sizes(self, h100_machine_spec: MachineSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that router GEMM time scales with batch size."""

        def mock_gemm_util(*args: object, **kwargs: object) -> float:
            return 0.8

        monkeypatch.setattr(
            "dlcalc.utils.moe_router_util.get_gemm_utilization_or_default",
            mock_gemm_util,
        )

        batch_sizes = [1, 2, 4, 8]
        times = []

        for batch in batch_sizes:
            time_s = calculate_router_gemm_time(
                batch=batch,
                seqlen=2048,
                hidden_dim=2048,
                n_experts=64,
                machine_spec=h100_machine_spec,
                dtype=DType.BF16,
            )
            times.append(time_s)

        # Time should scale linearly with batch size (M = batch * seqlen)
        for i in range(1, len(times)):
            ratio = times[i] / times[i - 1]
            expected_ratio = batch_sizes[i] / batch_sizes[i - 1]
            assert abs(ratio - expected_ratio) < 1e-6


class TestCalculateTopkTime:
    """Test cases for the calculate_topk_time function."""

    def test_h100_top2_routing(self, h100_machine_spec: MachineSpec) -> None:
        """Test TopK time for top-2 routing on H100."""
        # H100 throughput: 1.7e10 elements/sec (calibrated on H200, shared kernel family)
        # batch=1, seqlen=4096, n_experts=64, k=2
        time_s = calculate_topk_time(
            batch=1,
            seqlen=4096,
            n_experts=64,
            k=2,
            machine_spec=h100_machine_spec,
        )

        n_elements = 1 * 4096 * 64
        expected_time = n_elements / 1.7e10
        assert abs(time_s - expected_time) < 1e-9

    def test_a100_top1_routing(self, a100_machine_spec: MachineSpec) -> None:
        """Test TopK time for top-1 routing on A100."""
        # A100 throughput: 1.1e10 elements/sec (calibrated 2026-04-22)
        time_s = calculate_topk_time(
            batch=1,
            seqlen=2048,
            n_experts=32,
            k=1,
            machine_spec=a100_machine_spec,
        )

        n_elements = 1 * 2048 * 32
        expected_time = n_elements / 1.1e10
        assert abs(time_s - expected_time) < 1e-9

    def test_b200_top4_routing(self) -> None:
        """Test TopK time for top-4 routing on B200.

        Asserts against the shared measured throughput table (topk_throughput_elem_per_s),
        not a literal, so a recalibration updates the table in one place. The B200 value
        is the cross-shape median from the 2026-07-14 sweep (~1.5e10 elem/s).
        """
        from dlcalc.utils.moe_router_util import topk_throughput_elem_per_s

        b200_spec = MagicMock(spec=MachineSpec)
        b200_spec.name = "b200"

        time_s = calculate_topk_time(
            batch=2,
            seqlen=4096,
            n_experts=128,
            k=4,
            machine_spec=b200_spec,
        )

        n_elements = 2 * 4096 * 128
        expected_time = n_elements / topk_throughput_elem_per_s(b200_spec)
        assert abs(time_s - expected_time) < 1e-9

    def test_unknown_device_uses_default(self) -> None:
        """Unknown device uses the default throughput (a conservative ~A100-class
        value; the old 90M elem/s default was ~200x too slow vs any real GPU)."""
        from dlcalc.utils.moe_router_util import topk_throughput_elem_per_s

        unknown_spec = MagicMock(spec=MachineSpec)
        unknown_spec.name = "unknown_gpu"

        time_s = calculate_topk_time(
            batch=1,
            seqlen=2048,
            n_experts=64,
            k=2,
            machine_spec=unknown_spec,
        )

        n_elements = 1 * 2048 * 64
        expected_time = n_elements / topk_throughput_elem_per_s(unknown_spec)
        assert abs(time_s - expected_time) < 1e-9

    def test_time_scales_with_elements(self, h100_machine_spec: MachineSpec) -> None:
        """Test that TopK time scales linearly with number of elements."""
        configs = [
            (1, 1024, 32),  # Small
            (1, 2048, 64),  # Medium (4x more elements)
            (1, 4096, 128),  # Large (16x more elements than small)
        ]

        times = []
        elements = []

        for batch, seqlen, n_experts in configs:
            time_s = calculate_topk_time(
                batch=batch,
                seqlen=seqlen,
                n_experts=n_experts,
                k=2,
                machine_spec=h100_machine_spec,
            )
            times.append(time_s)
            elements.append(batch * seqlen * n_experts)

        # Time should scale linearly with elements
        for i in range(1, len(times)):
            ratio = times[i] / times[i - 1]
            expected_ratio = elements[i] / elements[i - 1]
            assert abs(ratio - expected_ratio) < 1e-6


class TestCalculatePermutationTime:
    """Test cases for the calculate_permutation_time function."""

    def test_h100_bf16_permutation(self, h100_machine_spec: MachineSpec) -> None:
        """Test permutation time for BF16 on H100."""
        # H100: 2 TB/s memory bandwidth
        # batch=1, seqlen=4096, hidden_dim=4096, BF16 (2 bytes)
        time_s = calculate_permutation_time(
            batch=1,
            seqlen=4096,
            hidden_dim=4096,
            machine_spec=h100_machine_spec,
            dtype_bytes=2,
        )

        # Data size = 1 * 4096 * 4096 * 2 = 32MB
        # Total transfers = 4 (2 for perm + 2 for unperm)
        # Total data = 32MB * 4 = 128MB
        # Time = 128MB / 2TB/s
        data_bytes = 1 * 4096 * 4096 * 2
        expected_time = (data_bytes * 4) / h100_machine_spec.device_spec.mem_bandwidth_bytes_per_sec
        assert abs(time_s - expected_time) < 1e-9

    def test_a100_fp32_permutation(self, a100_machine_spec: MachineSpec) -> None:
        """Test permutation time for FP32 on A100."""
        # A100: 1.5 TB/s memory bandwidth
        # batch=2, seqlen=2048, hidden_dim=2048, FP32 (4 bytes)
        time_s = calculate_permutation_time(
            batch=2,
            seqlen=2048,
            hidden_dim=2048,
            machine_spec=a100_machine_spec,
            dtype_bytes=4,
        )

        # Data size = 2 * 2048 * 2048 * 4 = 32MB
        # Total transfers = 4
        # Total data = 128MB
        # Time = 128MB / 1.5TB/s
        data_bytes = 2 * 2048 * 2048 * 4
        expected_time = (data_bytes * 4) / a100_machine_spec.device_spec.mem_bandwidth_bytes_per_sec
        assert abs(time_s - expected_time) < 1e-9

    def test_time_scales_with_data_size(self, h100_machine_spec: MachineSpec) -> None:
        """Test that permutation time scales with data size."""
        configs = [
            (1, 1024, 1024),  # Small
            (1, 2048, 2048),  # Medium (4x data)
            (1, 4096, 4096),  # Large (16x data)
        ]

        times = []
        data_sizes = []

        for batch, seqlen, hidden_dim in configs:
            time_s = calculate_permutation_time(
                batch=batch,
                seqlen=seqlen,
                hidden_dim=hidden_dim,
                machine_spec=h100_machine_spec,
                dtype_bytes=2,
            )
            times.append(time_s)
            data_sizes.append(batch * seqlen * hidden_dim)

        # Time should scale linearly with data size
        for i in range(1, len(times)):
            ratio = times[i] / times[i - 1]
            expected_ratio = data_sizes[i] / data_sizes[i - 1]
            assert abs(ratio - expected_ratio) < 1e-6


class TestCalculateTotalMoeRouterOverhead:
    """Test cases for the calculate_total_moe_router_overhead function."""

    def test_typical_moe_configuration(self, h100_machine_spec: MachineSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test total router overhead for typical MoE configuration."""

        def mock_gemm_util(*args: object, **kwargs: object) -> float:
            return 0.8

        monkeypatch.setattr(
            "dlcalc.utils.moe_router_util.get_gemm_utilization_or_default",
            mock_gemm_util,
        )

        # Typical config: batch=1, seqlen=4096, hidden_dim=4096, n_experts=64, k=2
        result = calculate_total_moe_router_overhead(
            batch=1,
            seqlen=4096,
            hidden_dim=4096,
            n_experts=64,
            k=2,
            machine_spec=h100_machine_spec,
            dtype=DType.BF16,
            dtype_bytes=2,
        )

        # Verify all components are present
        assert "Router GEMM" in result
        assert "Router TopK" in result
        assert "Router Permutation" in result
        assert "Total" in result

        # Verify all times are positive
        assert result["Router GEMM"] > 0
        assert result["Router TopK"] > 0
        assert result["Router Permutation"] > 0

        # Verify total is sum of components
        expected_total = (
            result["Router GEMM"] + result["Router TopK"] + result["Router Permutation"]
        )
        assert abs(result["Total"] - expected_total) < 1e-12

    def test_overhead_components_reasonable_magnitude(
        self, h100_machine_spec: MachineSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that router overhead components have reasonable relative magnitudes."""

        def mock_gemm_util(*args: object, **kwargs: object) -> float:
            return 0.8

        monkeypatch.setattr(
            "dlcalc.utils.moe_router_util.get_gemm_utilization_or_default",
            mock_gemm_util,
        )

        result = calculate_total_moe_router_overhead(
            batch=1,
            seqlen=4096,
            hidden_dim=4096,
            n_experts=64,
            k=2,
            machine_spec=h100_machine_spec,
            dtype=DType.BF16,
            dtype_bytes=2,
        )

        # Router GEMM is actually very small (M=4096, N=64, K=4096)
        # and highly optimized, so it can be faster than TopK
        # The key is that all components contribute to total overhead
        assert result["Router GEMM"] > 0
        assert result["Router TopK"] > 0
        assert result["Router Permutation"] > 0

        # Permutation involves more data movement than TopK selection
        # (reading/writing full hidden states vs just indices)
        assert result["Router Permutation"] < result["Router TopK"] * 100  # Reasonable bound

        # Total should be sum of all components
        assert result["Total"] >= result["Router GEMM"]
        assert result["Total"] >= result["Router TopK"]
        assert result["Total"] >= result["Router Permutation"]


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_end_to_end_small_moe_model(self, h100_machine_spec: MachineSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test complete router overhead calculation for small MoE model."""

        def mock_gemm_util(*args: object, **kwargs: object) -> float:
            return 0.8

        monkeypatch.setattr(
            "dlcalc.utils.moe_router_util.get_gemm_utilization_or_default",
            mock_gemm_util,
        )

        # Small MoE: 16 experts, top-1 routing
        batch, seqlen, hidden_dim, n_experts, k = 1, 2048, 2048, 16, 1

        # Calculate individually
        gemm_time = calculate_router_gemm_time(
            batch, seqlen, hidden_dim, n_experts, h100_machine_spec, DType.BF16
        )
        topk_time = calculate_topk_time(batch, seqlen, n_experts, k, h100_machine_spec)
        perm_time = calculate_permutation_time(
            batch, seqlen, hidden_dim, h100_machine_spec, 2
        )

        # Calculate via combined function
        result = calculate_total_moe_router_overhead(
            batch, seqlen, hidden_dim, n_experts, k, h100_machine_spec, DType.BF16, 2
        )

        # Should match
        assert abs(result["Router GEMM"] - gemm_time) < 1e-12
        assert abs(result["Router TopK"] - topk_time) < 1e-12
        assert abs(result["Router Permutation"] - perm_time) < 1e-12

    def test_end_to_end_large_moe_model(self, a100_machine_spec: MachineSpec, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test complete router overhead calculation for large MoE model."""

        def mock_gemm_util(*args: object, **kwargs: object) -> float:
            return 0.75

        monkeypatch.setattr(
            "dlcalc.utils.moe_router_util.get_gemm_utilization_or_default",
            mock_gemm_util,
        )

        # Large MoE: 128 experts, top-2 routing
        result = calculate_total_moe_router_overhead(
            batch=2,
            seqlen=4096,
            hidden_dim=4096,
            n_experts=128,
            k=2,
            machine_spec=a100_machine_spec,
            dtype=DType.BF16,
            dtype_bytes=2,
        )

        # All components should be present and positive
        assert all(v > 0 for v in result.values())
        # Total should be in reasonable range (microseconds to milliseconds)
        assert 1e-6 < result["Total"] < 0.1  # Between 1μs and 100ms
