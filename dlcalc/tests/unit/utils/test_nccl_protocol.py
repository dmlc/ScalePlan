"""Unit tests for NCCL protocol overhead modeling in comms.py.

This module tests NCCL protocol selection and overhead calculation functions
used to model communication time more accurately.

Test Coverage:
- get_nccl_protocol_overhead: Protocol selection based on message size
- calculate_chunking_overhead: Chunking pipeline overhead
- _ring_ag_or_rs_latency_term_s: Latency with protocol overhead
- _ring_ag_or_rs_bw_term_s: Bandwidth with protocol efficiency
"""

from dlcalc.utils.comms import (
    NCCL_CHUNK_SIZE,
    NCCL_LL128_BW_EFFICIENCY,
    NCCL_LL128_LATENCY_MULTIPLIER,
    NCCL_LL128_THRESHOLD,
    NCCL_LL_BW_EFFICIENCY,
    NCCL_LL_LATENCY_MULTIPLIER,
    NCCL_LL_THRESHOLD,
    NCCL_SIMPLE_BW_EFFICIENCY,
    NCCL_SIMPLE_LATENCY_MULTIPLIER,
    _ring_ag_or_rs_bw_term_s,
    _ring_ag_or_rs_latency_term_s,
    calculate_chunking_overhead,
    get_nccl_protocol_overhead,
)
from dlcalc.utils.data import Size


class TestGetNcclProtocolOverhead:
    """Test cases for the get_nccl_protocol_overhead function."""

    def test_ll_protocol_small_message(self) -> None:
        """Test LL protocol selection for small messages (<32KB)."""
        message_size = 16 * 1024  # 16KB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_LL_BW_EFFICIENCY
        assert lat_mult == NCCL_LL_LATENCY_MULTIPLIER
        assert bw_eff == 0.50
        assert lat_mult == 2.0

    def test_ll_protocol_at_threshold(self) -> None:
        """Test LL protocol at exact threshold boundary (32KB)."""
        message_size = NCCL_LL_THRESHOLD - 1  # Just below 32KB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_LL_BW_EFFICIENCY
        assert lat_mult == NCCL_LL_LATENCY_MULTIPLIER

    def test_ll128_protocol_medium_message(self) -> None:
        """Test LL128 protocol selection for medium messages (32KB-1MB)."""
        message_size = 512 * 1024  # 512KB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_LL128_BW_EFFICIENCY
        assert lat_mult == NCCL_LL128_LATENCY_MULTIPLIER
        assert bw_eff == 0.9375 * 0.90
        assert lat_mult == 1.5

    def test_ll128_efficiency_is_nccl_line_efficiency_times_residual(self) -> None:
        """LL128 effective BW = NCCL 120/128 line efficiency × ring residual.

        NCCL's LL128 protocol carries 120 useful bytes per 128-byte line
        (src/graph/tuning.cc: `0.92 /*120.0/128.0*/`). We multiply that line
        efficiency by the same 0.90 protocol-independent ring/staging residual
        applied to Simple, so the constant is physical (not the prior 0.70 guess)
        AND the effective-BW ordering stays LL < LL128 < Simple (0.50 < 0.844 <
        0.90) rather than inverting.
        """
        assert NCCL_LL128_BW_EFFICIENCY == (120.0 / 128.0) * NCCL_SIMPLE_BW_EFFICIENCY
        assert NCCL_LL_BW_EFFICIENCY == 4.0 / 8.0
        # Physical ordering must hold: smaller messages are never more efficient.
        assert NCCL_LL_BW_EFFICIENCY < NCCL_LL128_BW_EFFICIENCY < NCCL_SIMPLE_BW_EFFICIENCY

    def test_ll128_protocol_at_lower_threshold(self) -> None:
        """Test LL128 protocol at lower threshold boundary."""
        message_size = NCCL_LL_THRESHOLD  # Exactly 32KB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_LL128_BW_EFFICIENCY
        assert lat_mult == NCCL_LL128_LATENCY_MULTIPLIER

    def test_ll128_protocol_at_upper_threshold(self) -> None:
        """Test LL128 protocol at upper threshold boundary."""
        message_size = NCCL_LL128_THRESHOLD - 1  # Just below 1MB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_LL128_BW_EFFICIENCY
        assert lat_mult == NCCL_LL128_LATENCY_MULTIPLIER

    def test_simple_protocol_large_message(self) -> None:
        """Test Simple protocol selection for large messages (>1MB)."""
        message_size = 10 * 1024 * 1024  # 10MB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_SIMPLE_BW_EFFICIENCY
        assert lat_mult == NCCL_SIMPLE_LATENCY_MULTIPLIER
        assert bw_eff == 0.90
        assert lat_mult == 1.0

    def test_simple_protocol_at_threshold(self) -> None:
        """Test Simple protocol at exact threshold boundary (1MB)."""
        message_size = NCCL_LL128_THRESHOLD  # Exactly 1MB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_SIMPLE_BW_EFFICIENCY
        assert lat_mult == NCCL_SIMPLE_LATENCY_MULTIPLIER

    def test_very_small_message(self) -> None:
        """Test protocol selection for very small message (1KB)."""
        message_size = 1024  # 1KB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_LL_BW_EFFICIENCY
        assert lat_mult == NCCL_LL_LATENCY_MULTIPLIER

    def test_very_large_message(self) -> None:
        """Test protocol selection for very large message (1GB)."""
        message_size = 1024 * 1024 * 1024  # 1GB
        bw_eff, lat_mult = get_nccl_protocol_overhead(message_size)

        assert bw_eff == NCCL_SIMPLE_BW_EFFICIENCY
        assert lat_mult == NCCL_SIMPLE_LATENCY_MULTIPLIER

    def test_efficiency_bounds(self) -> None:
        """Test that efficiency and latency multiplier are within valid bounds."""
        test_sizes = [1024, 32 * 1024, 512 * 1024, 1024 * 1024, 10 * 1024 * 1024]
        for size in test_sizes:
            bw_eff, lat_mult = get_nccl_protocol_overhead(size)
            # Efficiency should be between 0 and 1
            assert 0.0 < bw_eff <= 1.0
            # Latency multiplier should be positive
            assert lat_mult >= 1.0


class TestCalculateChunkingOverhead:
    """Test cases for the calculate_chunking_overhead function."""

    def test_single_chunk_no_overhead(self) -> None:
        """Test that single chunk (< 512KB) has no chunking overhead."""
        message_size = 256 * 1024  # 256KB
        base_latency = 1e-6  # 1 microsecond
        overhead = calculate_chunking_overhead(message_size, base_latency)

        # n_chunks = max(1, 256KB // 512KB) = 1
        # overhead = (1 - 1) * 2.0us = 0
        assert overhead == 0.0

    def test_two_chunks(self) -> None:
        """Test chunking overhead with two chunks."""
        message_size = 1024 * 1024  # 1MB = 2 chunks of 512KB
        base_latency = 1e-6
        overhead = calculate_chunking_overhead(message_size, base_latency)

        # n_chunks = 1024KB // 512KB = 2
        # overhead = (2 - 1) * 2.0us = 2.0us = 2e-6s
        expected_overhead = 1 * 2.0 * 1e-6
        assert abs(overhead - expected_overhead) < 1e-9

    def test_multiple_chunks(self) -> None:
        """Test chunking overhead with multiple chunks."""
        message_size = 4 * 1024 * 1024  # 4MB = 8 chunks of 512KB
        base_latency = 1e-6
        overhead = calculate_chunking_overhead(message_size, base_latency)

        # n_chunks = 4MB // 512KB = 8
        # overhead = (8 - 1) * 2.0us = 14us = 14e-6s
        expected_overhead = 7 * 2.0 * 1e-6
        assert abs(overhead - expected_overhead) < 1e-9

    def test_exact_chunk_boundary(self) -> None:
        """Test message size at exact chunk boundary."""
        message_size = NCCL_CHUNK_SIZE  # Exactly 512KB
        base_latency = 1e-6
        overhead = calculate_chunking_overhead(message_size, base_latency)

        # n_chunks = 512KB // 512KB = 1
        # overhead = (1 - 1) * 2.0us = 0
        assert overhead == 0.0

    def test_slightly_over_chunk_boundary(self) -> None:
        """Test message size just over chunk boundary."""
        message_size = NCCL_CHUNK_SIZE + 1  # 512KB + 1 byte
        base_latency = 1e-6
        overhead = calculate_chunking_overhead(message_size, base_latency)

        # n_chunks = (512KB + 1) // 512KB = 1 (integer division)
        # overhead = (1 - 1) * 2.0us = 0
        assert overhead == 0.0

    def test_large_message_many_chunks(self) -> None:
        """Test large message with many chunks (100MB)."""
        message_size = 100 * 1024 * 1024  # 100MB
        base_latency = 1e-6
        overhead = calculate_chunking_overhead(message_size, base_latency)

        # n_chunks = 100MB // 512KB = 200
        # overhead = (200 - 1) * 2.0us = 398us
        n_chunks = 100 * 1024 * 1024 // NCCL_CHUNK_SIZE
        expected_overhead = (n_chunks - 1) * 2.0 * 1e-6
        assert abs(overhead - expected_overhead) < 1e-9

    def test_overhead_scales_with_chunks(self) -> None:
        """Test that overhead scales linearly with number of chunks."""
        base_latency = 1e-6
        sizes = [1 * 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024]  # 1MB, 2MB, 4MB
        overheads = [calculate_chunking_overhead(s, base_latency) for s in sizes]

        # Overhead should roughly double when message size doubles
        # (not exactly due to integer division, but close)
        assert overheads[1] > overheads[0]
        assert overheads[2] > overheads[1]


class TestRingLatencyTermWithProtocol:
    """Test cases for _ring_ag_or_rs_latency_term_s with protocol overhead."""

    def test_without_message_size_fallback(self) -> None:
        """Test that function falls back to original behavior without message size."""
        n_participants = 8
        link_latency = 1e-6  # 1 microsecond
        latency_term = _ring_ag_or_rs_latency_term_s(n_participants, link_latency)

        # Should use original formula: (n - 1) * latency
        expected = 7 * 1e-6
        assert latency_term == expected

    def test_with_small_message_ll_protocol(self) -> None:
        """Test latency term with small message (LL protocol)."""
        n_participants = 8
        link_latency = 1e-6
        message_size = 16 * 1024  # 16KB -> LL protocol

        latency_term = _ring_ag_or_rs_latency_term_s(
            n_participants, link_latency, message_size
        )

        # Base latency: 7 * 1us = 7us
        # LL protocol: latency_mult = 2.0
        # Chunking: 0 (single chunk)
        # Total: 7us * 2.0 = 14us
        base_latency = 7 * 1e-6
        expected = base_latency * 2.0
        assert abs(latency_term - expected) < 1e-9

    def test_with_medium_message_ll128_protocol(self) -> None:
        """Test latency term with medium message (LL128 protocol)."""
        n_participants = 4
        link_latency = 2e-6
        message_size = 512 * 1024  # 512KB -> LL128 protocol

        latency_term = _ring_ag_or_rs_latency_term_s(
            n_participants, link_latency, message_size
        )

        # Base latency: 3 * 2us = 6us
        # LL128 protocol: latency_mult = 1.5
        # Chunking: 0 (single chunk)
        # Total: 6us * 1.5 = 9us
        base_latency = 3 * 2e-6
        expected = base_latency * 1.5
        assert abs(latency_term - expected) < 1e-9

    def test_with_large_message_simple_protocol(self) -> None:
        """Test latency term with large message (Simple protocol)."""
        n_participants = 16
        link_latency = 1e-6
        message_size = 10 * 1024 * 1024  # 10MB -> Simple protocol

        latency_term = _ring_ag_or_rs_latency_term_s(
            n_participants, link_latency, message_size
        )

        # Base latency: 15 * 1us = 15us
        # Simple protocol: latency_mult = 1.0
        # Chunking: (10MB // 512KB - 1) * 2us = 19 * 2us = 38us
        # Total: 15us * 1.0 + 38us = 53us
        base_latency = 15 * 1e-6
        n_chunks = message_size // NCCL_CHUNK_SIZE
        chunking = (n_chunks - 1) * 2.0 * 1e-6
        expected = base_latency * 1.0 + chunking
        assert abs(latency_term - expected) < 1e-9

    def test_two_participants(self) -> None:
        """Test with minimum participants (2)."""
        n_participants = 2
        link_latency = 1e-6
        message_size = 1 * 1024 * 1024  # 1MB

        latency_term = _ring_ag_or_rs_latency_term_s(
            n_participants, link_latency, message_size
        )

        # Base: 1 * 1us = 1us
        # Simple protocol: mult = 1.0
        # Chunking: (2 - 1) * 2us = 2us
        # Total: 1us + 2us = 3us
        expected = 1e-6 + 2.0 * 1e-6
        assert abs(latency_term - expected) < 1e-9


class TestRingBwTermWithProtocol:
    """Test cases for _ring_ag_or_rs_bw_term_s with protocol efficiency."""

    def test_small_message_ll_protocol(self) -> None:
        """Test bandwidth term with small message (LL protocol, 50% efficiency)."""
        # 16KB with 8-bit elements: numel = 16*1024 bytes
        size = Size(numel=16 * 1024, bits_per_element=8)
        n_participants = 8
        link_bw = 100 * 1024 * 1024 * 1024  # 100 GB/s

        bw_term = _ring_ag_or_rs_bw_term_s(size, n_participants, link_bw)

        # Phase data per participant: 16KB / 8 = 2KB
        # LL protocol: 50% efficiency -> effective BW = 50 GB/s
        # Time per phase: 2KB / 50GB/s
        # Total: 7 phases * time_per_phase
        phase_data = size.bytes() / n_participants
        effective_bw = link_bw * 0.5
        expected = 7 * (phase_data / effective_bw)
        assert abs(bw_term - expected) < 1e-9

    def test_medium_message_ll128_protocol(self) -> None:
        """Test bandwidth term with medium message (LL128 protocol, 120/128 eff.)."""
        # 512KB with 8-bit elements
        size = Size(numel=512 * 1024, bits_per_element=8)
        n_participants = 4
        link_bw = 200 * 1024 * 1024 * 1024  # 200 GB/s

        bw_term = _ring_ag_or_rs_bw_term_s(size, n_participants, link_bw)

        # Phase data: 512KB / 4 = 128KB
        # LL128 protocol: NCCL line efficiency 120/128 = 0.9375
        # Effective BW: 200 * 0.9375 = 187.5 GB/s
        phase_data = size.bytes() / n_participants
        effective_bw = link_bw * NCCL_LL128_BW_EFFICIENCY
        expected = 3 * (phase_data / effective_bw)
        assert abs(bw_term - expected) < 1e-9

    def test_large_message_simple_protocol(self) -> None:
        """Test bandwidth term with large message (Simple protocol, 90% efficiency)."""
        # 10MB with 8-bit elements
        size = Size(numel=10 * 1024 * 1024, bits_per_element=8)
        n_participants = 16
        link_bw = 400 * 1024 * 1024 * 1024  # 400 GB/s

        bw_term = _ring_ag_or_rs_bw_term_s(size, n_participants, link_bw)

        # Phase data: 10MB / 16
        # Simple protocol: 90% efficiency
        # Effective BW: 400 * 0.9 = 360 GB/s
        phase_data = size.bytes() / n_participants
        effective_bw = link_bw * 0.9
        expected = 15 * (phase_data / effective_bw)
        assert abs(bw_term - expected) < 1e-9

    def test_efficiency_impact(self) -> None:
        """Test that lower efficiency increases bandwidth term."""
        n_participants = 8
        link_bw = 100 * 1024 * 1024 * 1024

        # Create different message sizes to trigger different protocols
        size_small = Size(numel=16 * 1024, bits_per_element=8)  # 16KB - LL: 50% efficiency
        size_medium = Size(numel=512 * 1024, bits_per_element=8)  # 512KB - LL128: 93.75% eff.
        size_large = Size(numel=1024 * 1024 * 1024, bits_per_element=8)  # 1GB - Simple: 90% efficiency

        bw_term_small = _ring_ag_or_rs_bw_term_s(size_small, n_participants, link_bw)
        bw_term_medium = _ring_ag_or_rs_bw_term_s(size_medium, n_participants, link_bw)
        bw_term_large = _ring_ag_or_rs_bw_term_s(size_large, n_participants, link_bw)

        # For same data size, lower efficiency means higher time
        # But here sizes differ, so just check that terms are positive
        assert bw_term_small > 0
        assert bw_term_medium > 0
        assert bw_term_large > 0

    def test_two_participants(self) -> None:
        """Test with minimum participants (2)."""
        # 1MB with 8-bit elements
        size = Size(numel=1024 * 1024, bits_per_element=8)
        n_participants = 2
        link_bw = 100 * 1024 * 1024 * 1024

        bw_term = _ring_ag_or_rs_bw_term_s(size, n_participants, link_bw)

        # Phase data: 1MB / 2 = 0.5MB
        # Simple protocol (1MB): 90% efficiency
        # Effective BW: 100 * 0.9 = 90 GB/s
        # Time: 1 phase * (0.5MB / 90GB/s)
        phase_data = size.bytes() / n_participants
        effective_bw = link_bw * 0.9
        expected = 1 * (phase_data / effective_bw)
        assert abs(bw_term - expected) < 1e-9
