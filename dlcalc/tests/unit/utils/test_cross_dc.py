"""Unit tests for cross-DC configuration and communication functions."""

from dlcalc.utils.comms import (
    get_cross_dc_dp_all_gather_comm_time_s,
    get_cross_dc_dp_reduce_scatter_comm_time_s,
)
from dlcalc.utils.configurations import CrossDCConfig
from dlcalc.utils.data import Size
from dlcalc.utils.hardware import MachineSpec
from dlcalc.utils.model_3d import ParallelConfig


class TestCrossDCConfig:
    """Test CrossDCConfig dataclass."""

    def test_initialization(self):
        """Test CrossDCConfig can be initialized with required parameters."""
        config = CrossDCConfig(
            n_dcs=3,
            interconnect_bandwidth_gbps=800,
            interconnect_latency_s=0.0035,
        )
        assert config.n_dcs == 3
        assert config.interconnect_bandwidth_gbps == 800
        assert config.interconnect_latency_s == 0.0035


class TestCrossDCCommunication:
    """Test cross-DC communication functions."""

    def test_cross_dc_dp_reduce_scatter(self):
        """Test cross-DC data parallel reduce-scatter calculation."""
        # Setup test configuration
        cross_dc_config = CrossDCConfig(
            n_dcs=3,
            interconnect_bandwidth_gbps=800,
            interconnect_latency_s=0.0035,
        )

        parallel_config = ParallelConfig(
            tp=8,
            cp=1,
            pp=2,
            dp=64,
            expert_mesh=None,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        machine_spec = MachineSpec.from_str("p5.48xlarge")

        # Size of data to communicate (e.g., gradient bucket)
        size = Size(numel=250_000_000, bits_per_element=32)  # 250M params, 32-bit grads

        # Calculate cross-DC reduce-scatter time
        cross_dc_time = get_cross_dc_dp_reduce_scatter_comm_time_s(
            size=size,
            parallel_config=parallel_config,
            machine_spec=machine_spec,
            cross_dc_config=cross_dc_config,
        )

        # Should be significantly higher than baseline due to cross-DC bottleneck
        assert cross_dc_time > 0
        # Rough check that it's in a reasonable range (milliseconds to seconds)
        assert 0.001 < cross_dc_time < 10.0

    def test_cross_dc_dp_all_gather(self):
        """Test cross-DC data parallel all-gather calculation."""
        cross_dc_config = CrossDCConfig(
            n_dcs=3,
            interconnect_bandwidth_gbps=800,
            interconnect_latency_s=0.0035,
        )

        parallel_config = ParallelConfig(
            tp=8,
            cp=1,
            pp=2,
            dp=64,
            expert_mesh=None,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        machine_spec = MachineSpec.from_str("p5.48xlarge")

        # Size of data to communicate (e.g., parameter bucket)
        size = Size(numel=250_000_000, bits_per_element=16)  # 250M params, 16-bit

        # Calculate cross-DC all-gather time
        cross_dc_time = get_cross_dc_dp_all_gather_comm_time_s(
            size=size,
            parallel_config=parallel_config,
            machine_spec=machine_spec,
            cross_dc_config=cross_dc_config,
        )

        # Should be significantly higher than baseline due to cross-DC bottleneck
        assert cross_dc_time > 0
        # Rough check that it's in a reasonable range
        assert 0.001 < cross_dc_time < 10.0

    def test_cross_dc_vs_baseline_degradation(self):
        """Test that cross-DC communication is slower than baseline."""
        from dlcalc.utils.comms import (
            get_dp_all_gather_comm_time_s,
            get_dp_reduce_scatter_comm_time_s,
        )

        cross_dc_config = CrossDCConfig(
            n_dcs=3,
            interconnect_bandwidth_gbps=800,  # Limited cross-DC bandwidth
            interconnect_latency_s=0.0035,  # 3.5ms latency
        )

        parallel_config = ParallelConfig(
            tp=8,
            cp=1,
            pp=2,
            dp=64,
            expert_mesh=None,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        machine_spec = MachineSpec.from_str("p5.48xlarge")
        size = Size(numel=250_000_000, bits_per_element=32)

        # Calculate baseline (single DC) times
        baseline_rs = get_dp_reduce_scatter_comm_time_s(
            size=size,
            parallel_config=parallel_config,
            machine_spec=machine_spec,
        )

        baseline_ag = get_dp_all_gather_comm_time_s(
            size=Size(numel=250_000_000, bits_per_element=16),
            parallel_config=parallel_config,
            machine_spec=machine_spec,
        )

        # Calculate cross-DC times
        cross_dc_rs = get_cross_dc_dp_reduce_scatter_comm_time_s(
            size=size,
            parallel_config=parallel_config,
            machine_spec=machine_spec,
            cross_dc_config=cross_dc_config,
        )

        cross_dc_ag = get_cross_dc_dp_all_gather_comm_time_s(
            size=Size(numel=250_000_000, bits_per_element=16),
            parallel_config=parallel_config,
            machine_spec=machine_spec,
            cross_dc_config=cross_dc_config,
        )

        # Cross-DC should be slower than baseline
        assert cross_dc_rs > baseline_rs
        assert cross_dc_ag > baseline_ag

        # Degradation should be significant but not unreasonable
        # Cross-DC can be quite a bit slower due to bandwidth and latency constraints
        assert 1.1 < (cross_dc_rs / baseline_rs) < 100
        assert 1.1 < (cross_dc_ag / baseline_ag) < 100
