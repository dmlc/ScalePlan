"""Unit tests for overlap.py utility functions.

This module tests communication-computation overlap modeling functions
used to calculate exposed DP communication time in distributed training.

Test Coverage:
- calculate_dp_overlap_efficiency: Overlap efficiency for different scenarios
- calculate_exposed_dp_time: Exposed communication time with pipeline parallelism
"""

import pytest

from dlcalc.utils.overlap import (
    calculate_dp_overlap_efficiency,
    calculate_exposed_dp_time,
)


class TestCalculateDpOverlapEfficiency:
    """Test cases for the calculate_dp_overlap_efficiency function."""

    def test_first_pp_stage_fully_exposed(self) -> None:
        """Test that first PP stage has no overlap (efficiency = 0.0)."""
        efficiency = calculate_dp_overlap_efficiency(
            microbatch_compute_time=1.0,
            comm_time_per_bucket=0.1,
            n_buckets=10,
            is_first_pp_stage=True,
        )
        assert efficiency == 0.0

    def test_perfect_overlap_non_first_stage(self) -> None:
        """Test perfect overlap when compute window exceeds comm time."""
        # Compute window per bucket = 1.0 / 10 = 0.1s
        # Comm time per bucket = 0.05s
        # Since compute window >= comm time, full overlap
        efficiency = calculate_dp_overlap_efficiency(
            microbatch_compute_time=1.0,
            comm_time_per_bucket=0.05,
            n_buckets=10,
            is_first_pp_stage=False,
        )
        assert efficiency == 1.0

    def test_partial_overlap_non_first_stage(self) -> None:
        """Test partial overlap when comm time exceeds compute window."""
        # Compute window per bucket = 1.0 / 10 = 0.1s
        # Comm time per bucket = 0.15s
        # Exposed time per bucket = 0.05s
        # Total exposed = 0.05 * 10 = 0.5s
        # Total comm = 0.15 * 10 = 1.5s
        # Efficiency = 1.0 - (0.5 / 1.5) = 0.667
        efficiency = calculate_dp_overlap_efficiency(
            microbatch_compute_time=1.0,
            comm_time_per_bucket=0.15,
            n_buckets=10,
            is_first_pp_stage=False,
        )
        expected_efficiency = 1.0 - (0.5 / 1.5)
        assert abs(efficiency - expected_efficiency) < 1e-6

    def test_no_overlap_comm_dominates(self) -> None:
        """Test case where comm time far exceeds compute time."""
        # Compute window per bucket = 0.1 / 10 = 0.01s
        # Comm time per bucket = 1.0s
        # Exposed time per bucket = 0.99s
        # Total exposed = 9.9s
        # Total comm = 10.0s
        # Efficiency = 1.0 - (9.9 / 10.0) = 0.01
        efficiency = calculate_dp_overlap_efficiency(
            microbatch_compute_time=0.1,
            comm_time_per_bucket=1.0,
            n_buckets=10,
            is_first_pp_stage=False,
        )
        expected_efficiency = 0.01
        assert abs(efficiency - expected_efficiency) < 1e-6

    def test_single_bucket(self) -> None:
        """Test with single bucket."""
        # Compute window = 1.0s, comm time = 0.5s
        # Exposed time = 0, full overlap
        efficiency = calculate_dp_overlap_efficiency(
            microbatch_compute_time=1.0,
            comm_time_per_bucket=0.5,
            n_buckets=1,
            is_first_pp_stage=False,
        )
        assert efficiency == 1.0

    def test_zero_comm_time(self) -> None:
        """Test edge case with zero communication time."""
        efficiency = calculate_dp_overlap_efficiency(
            microbatch_compute_time=1.0,
            comm_time_per_bucket=0.0,
            n_buckets=10,
            is_first_pp_stage=False,
        )
        assert efficiency == 1.0

    def test_many_buckets(self) -> None:
        """Test with many buckets (realistic gradient bucketing)."""
        # Realistic scenario: 100 buckets, 1s compute, 0.02s comm per bucket
        # Compute window per bucket = 1.0 / 100 = 0.01s
        # Comm time per bucket = 0.02s
        # Exposed per bucket = 0.01s
        # Total exposed = 1.0s, total comm = 2.0s
        # Efficiency = 1.0 - (1.0 / 2.0) = 0.5
        efficiency = calculate_dp_overlap_efficiency(
            microbatch_compute_time=1.0,
            comm_time_per_bucket=0.02,
            n_buckets=100,
            is_first_pp_stage=False,
        )
        assert abs(efficiency - 0.5) < 1e-6

    def test_efficiency_bounds(self) -> None:
        """Test that efficiency is always between 0.0 and 1.0."""
        test_cases = [
            (1.0, 0.1, 10, False),
            (1.0, 1.0, 5, False),
            (0.5, 0.2, 20, False),
            (2.0, 0.5, 8, False),
            (1.0, 0.0, 10, False),
        ]
        for compute_time, comm_time, n_buckets, is_first in test_cases:
            efficiency = calculate_dp_overlap_efficiency(
                microbatch_compute_time=compute_time,
                comm_time_per_bucket=comm_time,
                n_buckets=n_buckets,
                is_first_pp_stage=is_first,
            )
            assert 0.0 <= efficiency <= 1.0


class TestCalculateExposedDpTime:
    """Test cases for the calculate_exposed_dp_time function."""

    def test_pp_degree_1_fully_exposed(self) -> None:
        """Test that PP=1 results in fully exposed communication."""
        # With PP=1, treated as first stage, so fully exposed
        exposed_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=0.1,
            n_buckets=10,
            microbatch_compute_time=1.0,
            pp_degree=1,
        )
        # Total comm time = 0.1 * 10 = 1.0s
        # PP=1 means avg_overlap_efficiency = 0.0
        # Exposed time = 1.0 * (1.0 - 0.0) = 1.0s
        assert exposed_time == 1.0

    def test_pp_degree_2_half_exposed(self) -> None:
        """Test PP=2 with perfect overlap on second stage."""
        # PP=2: stage 0 (efficiency=0), stage 1 (depends on compute/comm ratio)
        # Compute window per bucket = 1.0 / 10 = 0.1s
        # Comm time per bucket = 0.05s (perfect overlap for stage 1)
        # Stage 1 efficiency = 1.0
        # Avg efficiency = (0.0 + 1.0) / 2 = 0.5
        # Total comm = 0.05 * 10 = 0.5s
        # Exposed = 0.5 * (1 - 0.5) = 0.25s
        exposed_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=0.05,
            n_buckets=10,
            microbatch_compute_time=1.0,
            pp_degree=2,
        )
        assert abs(exposed_time - 0.25) < 1e-6

    def test_pp_degree_4_with_partial_overlap(self) -> None:
        """Test PP=4 with partial overlap on non-first stages."""
        # Total comm = 0.1 * 10 = 1.0s
        # Compute window per bucket = 1.0 / 10 = 0.1s
        # Comm per bucket = 0.1s (perfect overlap for non-first stages)
        # Stage 0: efficiency = 0.0
        # Stages 1-3: efficiency = 1.0
        # Avg efficiency = (0.0 + 1.0 + 1.0 + 1.0) / 4 = 0.75
        # Exposed = 1.0 * (1 - 0.75) = 0.25s
        exposed_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=0.1,
            n_buckets=10,
            microbatch_compute_time=1.0,
            pp_degree=4,
        )
        assert abs(exposed_time - 0.25) < 1e-6

    def test_pp_degree_8_large_pipeline(self) -> None:
        """Test large pipeline degree reduces exposed time."""
        # Total comm = 0.2 * 10 = 2.0s
        # Compute window per bucket = 1.0 / 10 = 0.1s
        # Comm per bucket = 0.2s
        # Exposed per bucket for non-first stages = 0.1s
        # Non-first stage efficiency = 1.0 - (1.0 / 2.0) = 0.5
        # Stage 0: efficiency = 0.0
        # Stages 1-7: efficiency = 0.5
        # Avg efficiency = (0.0 + 7 * 0.5) / 8 = 3.5 / 8 = 0.4375
        # Exposed = 2.0 * (1 - 0.4375) = 1.125s
        exposed_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=0.2,
            n_buckets=10,
            microbatch_compute_time=1.0,
            pp_degree=8,
        )
        assert abs(exposed_time - 1.125) < 1e-6

    def test_zero_comm_time(self) -> None:
        """Test edge case with zero communication time."""
        exposed_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=0.0,
            n_buckets=10,
            microbatch_compute_time=1.0,
            pp_degree=4,
        )
        assert exposed_time == 0.0

    def test_comm_dominates_compute(self) -> None:
        """Test scenario where communication far exceeds compute."""
        # Total comm = 1.0 * 10 = 10.0s
        # Compute window per bucket = 0.1 / 10 = 0.01s
        # Comm per bucket = 1.0s
        # Exposed per bucket for non-first = 0.99s
        # Non-first efficiency = 1.0 - (9.9 / 10.0) = 0.01
        # With PP=4: avg efficiency = (0.0 + 3 * 0.01) / 4 = 0.0075
        # Exposed = 10.0 * (1 - 0.0075) = 9.925s
        exposed_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=1.0,
            n_buckets=10,
            microbatch_compute_time=0.1,
            pp_degree=4,
        )
        expected_exposed = 9.925
        assert abs(exposed_time - expected_exposed) < 1e-6

    def test_single_bucket_realistic(self) -> None:
        """Test with single large bucket."""
        # Total comm = 1.0s
        # Compute window = 2.0s (full overlap possible for non-first stages)
        # Non-first efficiency = 1.0
        # With PP=2: avg efficiency = (0.0 + 1.0) / 2 = 0.5
        # Exposed = 1.0 * 0.5 = 0.5s
        exposed_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=1.0,
            n_buckets=1,
            microbatch_compute_time=2.0,
            pp_degree=2,
        )
        assert abs(exposed_time - 0.5) < 1e-6

    def test_increasing_pp_degree_reduces_exposed_time(self) -> None:
        """Test that increasing PP degree monotonically reduces exposed time."""
        # Fixed scenario with partial overlap
        prev_exposed = float("inf")
        for pp_degree in [1, 2, 4, 8, 16]:
            exposed_time = calculate_exposed_dp_time(
                dp_comm_time_per_bucket=0.15,
                n_buckets=10,
                microbatch_compute_time=1.0,
                pp_degree=pp_degree,
            )
            # Should decrease or stay same as PP increases
            assert exposed_time <= prev_exposed
            prev_exposed = exposed_time

    def test_exposed_time_bounds(self) -> None:
        """Test that exposed time is within valid bounds."""
        test_cases = [
            (0.1, 10, 1.0, 2),
            (0.2, 5, 2.0, 4),
            (0.05, 20, 0.5, 8),
            (1.0, 1, 1.0, 1),
        ]
        for comm_per_bucket, n_buckets, compute_time, pp_degree in test_cases:
            exposed_time = calculate_exposed_dp_time(
                dp_comm_time_per_bucket=comm_per_bucket,
                n_buckets=n_buckets,
                microbatch_compute_time=compute_time,
                pp_degree=pp_degree,
            )
            total_comm_time = comm_per_bucket * n_buckets
            # Exposed time should be between 0 and total comm time
            assert 0.0 <= exposed_time <= total_comm_time
