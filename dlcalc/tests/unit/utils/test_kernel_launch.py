"""Unit tests for kernel_launch.py utility functions.

This module tests kernel launch overhead modeling functions used to
estimate cumulative CPU-GPU synchronization overhead in training iterations.

Test Coverage:
- estimate_kernel_count_per_transformer_block: Kernel counting for different model types
- calculate_kernel_launch_overhead: Total launch overhead calculation
"""

import pytest

from dlcalc.utils.kernel_launch import (
    calculate_kernel_launch_overhead,
    estimate_kernel_count_per_transformer_block,
)


class TestEstimateKernelCountPerTransformerBlock:
    """Test cases for the estimate_kernel_count_per_transformer_block function."""

    def test_base_model_no_tp_no_moe(self) -> None:
        """Test kernel count for basic transformer block without TP or MoE."""
        count = estimate_kernel_count_per_transformer_block(
            has_tp=False,
            has_moe=False,
            n_experts_active=0,
        )
        # Base kernels: 4 GEMM + 2 norm + 3 activation + 1 SDPA = 10
        assert count == 10

    def test_model_with_tp_no_moe(self) -> None:
        """Test kernel count for transformer block with tensor parallelism."""
        count = estimate_kernel_count_per_transformer_block(
            has_tp=True,
            has_moe=False,
            n_experts_active=0,
        )
        # Base: 10 + TP collectives: 4 = 14
        assert count == 14

    def test_model_with_moe_no_tp_top2(self) -> None:
        """Test kernel count for MoE model without TP (top-2 routing)."""
        count = estimate_kernel_count_per_transformer_block(
            has_tp=False,
            has_moe=True,
            n_experts_active=2,
        )
        # Base: 10 + Router: 2 + Experts (2 * 2): 4 = 16
        assert count == 16

    def test_model_with_moe_no_tp_top1(self) -> None:
        """Test kernel count for MoE model without TP (top-1 routing)."""
        count = estimate_kernel_count_per_transformer_block(
            has_tp=False,
            has_moe=True,
            n_experts_active=1,
        )
        # Base: 10 + Router: 2 + Experts (1 * 2): 2 = 14
        assert count == 14

    def test_model_with_tp_and_moe_top2(self) -> None:
        """Test kernel count for MoE model with TP (top-2 routing)."""
        count = estimate_kernel_count_per_transformer_block(
            has_tp=True,
            has_moe=True,
            n_experts_active=2,
        )
        # Base: 10 + TP: 4 + Router: 2 + Experts (2 * 2): 4 = 20
        assert count == 20

    def test_model_with_moe_top4(self) -> None:
        """Test kernel count for MoE model with top-4 routing."""
        count = estimate_kernel_count_per_transformer_block(
            has_tp=False,
            has_moe=True,
            n_experts_active=4,
        )
        # Base: 10 + Router: 2 + Experts (4 * 2): 8 = 20
        assert count == 20

    def test_model_with_tp_and_moe_top4(self) -> None:
        """Test kernel count for MoE model with TP and top-4 routing."""
        count = estimate_kernel_count_per_transformer_block(
            has_tp=True,
            has_moe=True,
            n_experts_active=4,
        )
        # Base: 10 + TP: 4 + Router: 2 + Experts (4 * 2): 8 = 24
        assert count == 24

    def test_moe_with_zero_experts_active(self) -> None:
        """Test edge case where MoE is enabled but no experts active."""
        count = estimate_kernel_count_per_transformer_block(
            has_tp=False,
            has_moe=True,
            n_experts_active=0,
        )
        # Base: 10 + Router: 2 + Experts (0 * 2): 0 = 12
        assert count == 12

    def test_kernel_count_increases_with_experts(self) -> None:
        """Test that kernel count scales linearly with number of active experts."""
        counts = []
        for n_experts in [0, 1, 2, 3, 4, 8]:
            count = estimate_kernel_count_per_transformer_block(
                has_tp=False,
                has_moe=True,
                n_experts_active=n_experts,
            )
            counts.append(count)

        # Check that counts increase linearly (by 2 per additional expert)
        for i in range(1, len(counts)):
            diff = counts[i] - counts[i - 1]
            n_experts_diff = [0, 1, 2, 3, 4, 8][i] - [0, 1, 2, 3, 4, 8][i - 1]
            # Each expert adds 2 kernels (Up and Down)
            expected_diff = n_experts_diff * 2
            assert diff == expected_diff


class TestCalculateKernelLaunchOverhead:
    """Test cases for the calculate_kernel_launch_overhead function."""

    def test_h100_small_model(self) -> None:
        """Test kernel launch overhead for small model on H100."""
        # 12-layer model, 10 kernels/block, H100
        overhead = calculate_kernel_launch_overhead(
            n_layers=12,
            kernels_per_block=10,
            device_name="h100_80gb",
        )
        # Total kernels: 2 * 12 * 10 = 240 (forward + backward)
        # H100 latency: 5μs per kernel
        # Expected: 240 * 5μs = 1200μs = 0.0012s
        expected = 240 * 5.0 * 1e-6
        assert abs(overhead - expected) < 1e-9

    def test_a100_medium_model(self) -> None:
        """Test kernel launch overhead for medium model on A100."""
        # 40-layer model, 14 kernels/block (with TP), A100
        overhead = calculate_kernel_launch_overhead(
            n_layers=40,
            kernels_per_block=14,
            device_name="a100_40gb",
        )
        # Total kernels: 2 * 40 * 14 = 1120
        # A100 latency: 8μs per kernel
        # Expected: 1120 * 8μs = 8960μs = 0.00896s
        expected = 1120 * 8.0 * 1e-6
        assert abs(overhead - expected) < 1e-9

    def test_h200_large_model(self) -> None:
        """Test kernel launch overhead for large model on H200."""
        # 80-layer model, 16 kernels/block (MoE with top-2), H200
        overhead = calculate_kernel_launch_overhead(
            n_layers=80,
            kernels_per_block=16,
            device_name="h200",
        )
        # Total kernels: 2 * 80 * 16 = 2560
        # H200 latency: 5μs per kernel
        # Expected: 2560 * 5μs = 12800μs = 0.0128s
        expected = 2560 * 5.0 * 1e-6
        assert abs(overhead - expected) < 1e-9

    def test_b200_very_large_model(self) -> None:
        """Test kernel launch overhead for very large model on B200."""
        # 128-layer model, 20 kernels/block (TP + MoE), B200
        overhead = calculate_kernel_launch_overhead(
            n_layers=128,
            kernels_per_block=20,
            device_name="b200",
        )
        # Total kernels: 2 * 128 * 20 = 5120
        # B200 latency: 5μs per kernel
        # Expected: 5120 * 5μs = 25600μs = 0.0256s
        expected = 5120 * 5.0 * 1e-6
        assert abs(overhead - expected) < 1e-9

    def test_unknown_device_uses_default(self) -> None:
        """Test that unknown device uses default latency (7μs)."""
        # Unknown device should use 7μs default
        overhead = calculate_kernel_launch_overhead(
            n_layers=40,
            kernels_per_block=10,
            device_name="unknown_gpu",
        )
        # Total kernels: 2 * 40 * 10 = 800
        # Default latency: 7μs per kernel
        # Expected: 800 * 7μs = 5600μs = 0.0056s
        expected = 800 * 7.0 * 1e-6
        assert abs(overhead - expected) < 1e-9

    def test_device_name_parsing(self) -> None:
        """Test that device name parsing correctly extracts device type."""
        # Test various device name formats
        test_cases = [
            ("h100_80gb", 5.0),  # Standard format
            ("H100_80GB", 5.0),  # Uppercase
            ("a100", 8.0),  # No suffix
            ("h200_sxm", 5.0),  # Different suffix
            ("b200_pcie", 5.0),  # Another suffix
        ]

        for device_name, expected_latency_us in test_cases:
            overhead = calculate_kernel_launch_overhead(
                n_layers=10,
                kernels_per_block=10,
                device_name=device_name,
            )
            # Total kernels: 2 * 10 * 10 = 200
            expected = 200 * expected_latency_us * 1e-6
            assert abs(overhead - expected) < 1e-9

    def test_overhead_scales_with_layers(self) -> None:
        """Test that overhead scales linearly with number of layers."""
        device_name = "h100_80gb"
        kernels_per_block = 10

        overheads = []
        layer_counts = [10, 20, 40, 80]
        for n_layers in layer_counts:
            overhead = calculate_kernel_launch_overhead(
                n_layers=n_layers,
                kernels_per_block=kernels_per_block,
                device_name=device_name,
            )
            overheads.append(overhead)

        # Check linear scaling
        for i in range(1, len(overheads)):
            ratio = overheads[i] / overheads[i - 1]
            expected_ratio = layer_counts[i] / layer_counts[i - 1]
            assert abs(ratio - expected_ratio) < 1e-6

    def test_overhead_scales_with_kernels_per_block(self) -> None:
        """Test that overhead scales linearly with kernels per block."""
        device_name = "h100_80gb"
        n_layers = 40

        overheads = []
        kernel_counts = [10, 14, 16, 20]
        for kernels_per_block in kernel_counts:
            overhead = calculate_kernel_launch_overhead(
                n_layers=n_layers,
                kernels_per_block=kernels_per_block,
                device_name=device_name,
            )
            overheads.append(overhead)

        # Check linear scaling
        for i in range(1, len(overheads)):
            ratio = overheads[i] / overheads[i - 1]
            expected_ratio = kernel_counts[i] / kernel_counts[i - 1]
            assert abs(ratio - expected_ratio) < 1e-6

    def test_realistic_scenario_80layer_h100(self) -> None:
        """Test realistic scenario: 80-layer model with TP on H100."""
        # Realistic 80-layer model with TP (14 kernels/block)
        overhead = calculate_kernel_launch_overhead(
            n_layers=80,
            kernels_per_block=14,
            device_name="h100_80gb",
        )
        # Total kernels: 2 * 80 * 14 = 2240
        # H100 latency: 5μs
        # Expected: 2240 * 5μs = 11200μs = 11.2ms
        expected = 2240 * 5.0 * 1e-6
        assert abs(overhead - expected) < 1e-9
        # Verify it's in the ballpark of 10-15ms
        assert 0.010 < overhead < 0.015

    def test_realistic_scenario_128layer_moe_a100(self) -> None:
        """Test realistic scenario: 128-layer MoE model on A100."""
        # Large MoE model: 128 layers, top-2 routing, no TP (16 kernels/block)
        overhead = calculate_kernel_launch_overhead(
            n_layers=128,
            kernels_per_block=16,
            device_name="a100_80gb",
        )
        # Total kernels: 2 * 128 * 16 = 4096
        # A100 latency: 8μs
        # Expected: 4096 * 8μs = 32768μs = 32.768ms
        expected = 4096 * 8.0 * 1e-6
        assert abs(overhead - expected) < 1e-9
        # Verify it's in the ballpark of 30-35ms
        assert 0.030 < overhead < 0.035

    def test_minimal_model(self) -> None:
        """Test minimal model: 1 layer, base kernels."""
        overhead = calculate_kernel_launch_overhead(
            n_layers=1,
            kernels_per_block=10,
            device_name="h100_80gb",
        )
        # Total kernels: 2 * 1 * 10 = 20
        # H100 latency: 5μs
        # Expected: 20 * 5μs = 100μs = 0.0001s
        expected = 20 * 5.0 * 1e-6
        assert abs(overhead - expected) < 1e-9
        # Should be very small
        assert overhead < 0.001  # Less than 1ms


class TestIntegration:
    """Integration tests combining both functions."""

    def test_end_to_end_dense_model_h100(self) -> None:
        """Test complete workflow for dense model on H100."""
        # Dense model: 40 layers, TP enabled, no MoE
        kernels_per_block = estimate_kernel_count_per_transformer_block(
            has_tp=True,
            has_moe=False,
            n_experts_active=0,
        )
        assert kernels_per_block == 14

        overhead = calculate_kernel_launch_overhead(
            n_layers=40,
            kernels_per_block=kernels_per_block,
            device_name="h100_80gb",
        )
        # Total kernels: 2 * 40 * 14 = 1120
        # Expected: 1120 * 5μs = 5.6ms
        assert abs(overhead - 0.0056) < 1e-6

    def test_end_to_end_moe_model_a100(self) -> None:
        """Test complete workflow for MoE model on A100."""
        # MoE model: 80 layers, TP enabled, top-2 routing
        kernels_per_block = estimate_kernel_count_per_transformer_block(
            has_tp=True,
            has_moe=True,
            n_experts_active=2,
        )
        assert kernels_per_block == 20

        overhead = calculate_kernel_launch_overhead(
            n_layers=80,
            kernels_per_block=kernels_per_block,
            device_name="a100_80gb",
        )
        # Total kernels: 2 * 80 * 20 = 3200
        # Expected: 3200 * 8μs = 25.6ms
        assert abs(overhead - 0.0256) < 1e-6

    def test_end_to_end_minimal_model(self) -> None:
        """Test complete workflow for minimal model."""
        # Minimal model: 12 layers, no TP, no MoE
        kernels_per_block = estimate_kernel_count_per_transformer_block(
            has_tp=False,
            has_moe=False,
            n_experts_active=0,
        )
        assert kernels_per_block == 10

        overhead = calculate_kernel_launch_overhead(
            n_layers=12,
            kernels_per_block=kernels_per_block,
            device_name="h100_80gb",
        )
        # Total kernels: 2 * 12 * 10 = 240
        # Expected: 240 * 5μs = 1.2ms
        assert abs(overhead - 0.0012) < 1e-6
