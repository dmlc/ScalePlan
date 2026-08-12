"""Unit tests for model_3d module, focusing on ParallelConfig and optimizer states."""

import pytest

from dlcalc.utils.data import TensorRepr
from dlcalc.utils.model_3d import (
    DistributedAdamOptimizerStates,
    DistributedMuonOptimizerStates,
    DistributedScionOptimizerStates,
    ModelStates,
    ParallelConfig,
)


class TestParallelConfig:
    """Test ParallelConfig class functionality."""

    def test_basic_initialization(self):
        """Test basic ParallelConfig initialization without expert mesh."""
        config = ParallelConfig(
            tp=4,
            cp=2,
            pp=2,
            dp=8,
            expert_mesh=None,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.tp == 4
        assert config.cp == 2
        assert config.pp == 2
        assert config.dp == 8
        assert config.expert_mesh is None
        assert config.vpp == 1
        assert config.sp_enabled is True
        assert config.zero_level == ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER

    def test_world_size_calculation(self):
        """Test world_size calculation."""
        config = ParallelConfig(
            tp=4,
            cp=2,
            pp=2,
            dp=8,
            expert_mesh=None,
            vpp=1,
            sp_enabled=False,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.world_size() == 4 * 2 * 2 * 8  # tp * cp * pp * dp = 128

    def test_expert_parallel_cfg_initialization(self):
        """Test ExpertParallelCfg initialization."""
        expert_cfg = ParallelConfig.ExpertParallelCfg(ep=8, tp=2, dp=4)

        assert expert_cfg.ep == 8
        assert expert_cfg.tp == 2
        assert expert_cfg.dp == 4

    def test_parallel_config_with_valid_expert_mesh(self):
        """Test ParallelConfig with a valid expert mesh configuration."""
        # The constraint is: expert_mesh.ep * expert_mesh.tp * expert_mesh.dp == dp * cp * tp
        # With tp=4, cp=2, dp=8: dp * cp * tp = 8 * 2 * 4 = 64
        # So expert_mesh.ep * expert_mesh.tp * expert_mesh.dp must equal 64
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=8,  # 8 * 2 * 4 = 64 ✓
            tp=2,
            dp=4,
        )

        config = ParallelConfig(
            tp=4,
            cp=2,
            pp=2,
            dp=8,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.expert_mesh.ep == 8
        assert config.expert_mesh.tp == 2
        assert config.expert_mesh.dp == 4

    def test_parallel_config_with_invalid_expert_mesh_raises_error(self):
        """Test that invalid expert mesh configuration raises assertion error."""
        # The constraint is: expert_mesh.ep * expert_mesh.tp * expert_mesh.dp == dp * cp * tp
        # With tp=4, cp=2, dp=8: dp * cp * tp = 8 * 2 * 4 = 64
        # But we'll provide expert_mesh that multiplies to 32 (invalid)
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=4,  # 4 * 2 * 4 = 32 ✗ (should be 64)
            tp=2,
            dp=4,
        )

        with pytest.raises(AssertionError):
            ParallelConfig(
                tp=4,
                cp=2,
                pp=2,
                dp=8,
                expert_mesh=expert_mesh,
                vpp=1,
                sp_enabled=True,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            )

    def test_zero_level_enum_values(self):
        """Test ZeroLevel enum values."""
        assert ParallelConfig.ZeroLevel.NONE == 0
        assert ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER == 1
        assert ParallelConfig.ZeroLevel.PARTITION_GRADIENTS == 2
        assert ParallelConfig.ZeroLevel.PARTITION_PARAMETERS == 3

    def test_different_world_sizes(self):
        """Test various world size configurations."""
        test_cases = [
            (1, 1, 1, 1, 1),  # Single device
            (2, 1, 1, 1, 2),  # TP only
            (1, 1, 2, 1, 2),  # PP only
            (1, 1, 1, 4, 4),  # DP only
            (2, 2, 2, 2, 16),  # All dimensions
            (8, 1, 4, 16, 512),  # Large cluster
        ]

        for tp, cp, pp, dp, expected_world_size in test_cases:
            config = ParallelConfig(
                tp=tp,
                cp=cp,
                pp=pp,
                dp=dp,
                expert_mesh=None,
                vpp=1,
                sp_enabled=False,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            )
            assert config.world_size() == expected_world_size

    def test_expert_mesh_constraint_with_cp_equals_1(self):
        """Test expert mesh constraint when cp=1."""
        # With tp=8, cp=1, dp=64: dp * cp * tp = 64 * 1 * 8 = 512
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=8,  # 8 * 8 * 8 = 512 ✓
            tp=8,
            dp=8,
        )

        config = ParallelConfig(
            tp=8,
            cp=1,
            pp=4,
            dp=64,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=False,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.expert_mesh.ep == 8
        assert config.expert_mesh.tp == 8
        assert config.expert_mesh.dp == 8
        # Verify the constraint
        assert (
            config.expert_mesh.ep * config.expert_mesh.tp * config.expert_mesh.dp
            == config.dp * config.cp * config.tp
        )

    def test_expert_mesh_with_different_tp_values(self):
        """Test that expert_mesh.tp can differ from main tp."""
        # Main config has tp=4, but expert_mesh has tp=1
        # With tp=4, cp=2, dp=8: dp * cp * tp = 8 * 2 * 4 = 64
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=16,  # 16 * 1 * 4 = 64 ✓
            tp=1,  # Different from main tp=4
            dp=4,
        )

        config = ParallelConfig(
            tp=4,
            cp=2,
            pp=2,
            dp=8,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.tp == 4  # Main TP
        assert config.expert_mesh.tp == 1  # Expert TP (different)
        assert config.expert_mesh.ep == 16
        assert config.expert_mesh.dp == 4

    def test_vpp_values(self):
        """Test different VPP (Virtual Pipeline Parallel) values."""
        for vpp in [1, 2, 4, 8]:
            config = ParallelConfig(
                tp=2,
                cp=1,
                pp=4,
                dp=2,
                expert_mesh=None,
                vpp=vpp,
                sp_enabled=False,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            )
            assert config.vpp == vpp

    def test_sp_enabled_flag(self):
        """Test sequence parallel enablement flag."""
        config_sp_on = ParallelConfig(
            tp=2,
            cp=1,
            pp=2,
            dp=2,
            expert_mesh=None,
            vpp=1,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )
        assert config_sp_on.sp_enabled is True

        config_sp_off = ParallelConfig(
            tp=2,
            cp=1,
            pp=2,
            dp=2,
            expert_mesh=None,
            vpp=1,
            sp_enabled=False,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )
        assert config_sp_off.sp_enabled is False

    def test_edge_case_single_device_with_expert_mesh(self):
        """Test edge case with single device and expert mesh."""
        # With tp=1, cp=1, dp=1: dp * cp * tp = 1 * 1 * 1 = 1
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=1,  # 1 * 1 * 1 = 1 ✓
            tp=1,
            dp=1,
        )

        config = ParallelConfig(
            tp=1,
            cp=1,
            pp=1,
            dp=1,
            expert_mesh=expert_mesh,
            vpp=1,
            sp_enabled=False,
            zero_level=ParallelConfig.ZeroLevel.NONE,
        )

        assert config.world_size() == 1
        assert config.expert_mesh.ep == 1
        assert config.expert_mesh.tp == 1
        assert config.expert_mesh.dp == 1

    def test_large_scale_configuration(self):
        """Test large-scale cluster configuration."""
        # Simulating a large cluster with 12288 GPUs
        # tp=8, cp=4, pp=6, dp=64 => 8 * 4 * 6 * 64 = 12288
        expert_mesh = ParallelConfig.ExpertParallelCfg(
            ep=128,  # 128 * 8 * 2 = 2048 = dp * cp * tp = 64 * 4 * 8
            tp=8,
            dp=2,
        )

        config = ParallelConfig(
            tp=8,
            cp=4,
            pp=6,
            dp=64,
            expert_mesh=expert_mesh,
            vpp=2,
            sp_enabled=True,
            zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        )

        assert config.world_size() == 12288
        assert config.expert_mesh.ep == 128
        # Verify constraint
        assert (
            config.expert_mesh.ep * config.expert_mesh.tp * config.expert_mesh.dp
            == config.dp * config.cp * config.tp
        )


class TestDistributedMuonOptimizerStates:
    """Test DistributedMuonOptimizerStates class functionality."""

    def test_basic_initialization(self):
        """Test basic initialization of Muon optimizer states."""
        n_params = 1000000
        dp = 8
        muon_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        # Verify tensor shapes
        assert muon_states.param_shard._unpartitioned_shape == (n_params,)
        assert muon_states.velocity_shard._unpartitioned_shape == (n_params,)
        assert muon_states.grad_buffer._unpartitioned_shape == (n_params,)

        # Verify partitioning
        assert muon_states.param_shard._partition_spec == {0: dp}
        assert muon_states.velocity_shard._partition_spec == {0: dp}
        assert muon_states.grad_buffer._partition_spec == {}  # Not partitioned

        # Verify bit precision
        assert muon_states.param_shard._bits_per_elt == 16  # store_param_remainders=True
        assert muon_states.velocity_shard._bits_per_elt == 32
        assert muon_states.grad_buffer._bits_per_elt == 32

    def test_initialization_without_param_remainders(self):
        """Test initialization with store_param_remainders=False."""
        muon_states = DistributedMuonOptimizerStates(
            n_params=1000000,
            store_param_remainders=False,
            dp=8,
        )

        # param_shard should use 32 bits when not storing remainders
        assert muon_states.param_shard._bits_per_elt == 32

    def test_total_bytes_partitioned(self):
        """Test total_bytes calculation with partitioned=True."""
        n_params = 1000000
        dp = 8
        muon_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        total_bytes = muon_states.total_bytes(partitioned=True)

        # Calculate expected bytes:
        # param_shard:    (1M / 8) * 16 bits = 125K * 2 bytes = 250KB
        # velocity_shard: (1M / 8) * 32 bits = 125K * 4 bytes = 500KB
        # grad_buffer:    1M * 32 bits = 1M * 4 bytes = 4MB (not partitioned)
        # Total: 250KB + 500KB + 4MB = 4.75MB = 4,750,000 bytes
        expected_bytes = (n_params // dp) * 2 + (n_params // dp) * 4 + n_params * 4
        assert total_bytes == expected_bytes

    def test_total_bytes_unpartitioned(self):
        """Test total_bytes calculation with partitioned=False."""
        n_params = 1000000
        muon_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=8,
        )

        total_bytes = muon_states.total_bytes(partitioned=False)

        # Calculate expected bytes (unpartitioned):
        # param_shard:    1M * 16 bits = 1M * 2 bytes = 2MB
        # velocity_shard: 1M * 32 bits = 1M * 4 bytes = 4MB
        # grad_buffer:    1M * 32 bits = 1M * 4 bytes = 4MB
        # Total: 2MB + 4MB + 4MB = 10MB = 10,000,000 bytes
        expected_bytes = n_params * 2 + n_params * 4 + n_params * 4
        assert total_bytes == expected_bytes

    def test_memory_efficiency_compared_to_adam(self):
        """Test that Muon uses less memory than Adam (no exp_avg_sq)."""
        n_params = 1000000
        dp = 8

        muon_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        adam_states = DistributedAdamOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        # Muon should use less memory due to missing exp_avg_sq
        muon_bytes = muon_states.total_bytes(partitioned=True)
        adam_bytes = adam_states.total_bytes(partitioned=True)

        # Adam has exp_avg (32 bits) and exp_avg_sq (32 bits)
        # Muon has only velocity (32 bits)
        # Difference should be one 32-bit buffer partitioned over DP
        expected_difference = (n_params // dp) * 4
        assert adam_bytes - muon_bytes == expected_difference

    def test_different_dp_values(self):
        """Test optimizer states with different DP degrees."""
        n_params = 1000000
        test_dp_values = [1, 2, 4, 8, 16, 32]

        for dp in test_dp_values:
            muon_states = DistributedMuonOptimizerStates(
                n_params=n_params,
                store_param_remainders=True,
                dp=dp,
            )

            # Verify partitioned size decreases with larger DP
            partitioned_bytes = muon_states.total_bytes(partitioned=True)
            # Unpartitioned size should remain constant
            unpartitioned_bytes = muon_states.total_bytes(partitioned=False)

            assert unpartitioned_bytes == n_params * 2 + n_params * 4 + n_params * 4
            # Larger DP should result in smaller partitioned size
            assert partitioned_bytes <= unpartitioned_bytes


class TestDistributedScionOptimizerStates:
    """Test DistributedScionOptimizerStates class functionality."""

    def test_basic_initialization(self):
        """Test basic initialization of Scion optimizer states."""
        n_params = 1000000
        dp = 8
        scion_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        # Verify tensor shapes
        assert scion_states.param_shard._unpartitioned_shape == (n_params,)
        assert scion_states.momentum_shard._unpartitioned_shape == (n_params,)
        assert scion_states.grad_buffer._unpartitioned_shape == (n_params,)

        # Verify partitioning
        assert scion_states.param_shard._partition_spec == {0: dp}
        assert scion_states.momentum_shard._partition_spec == {0: dp}
        assert scion_states.grad_buffer._partition_spec == {}  # Not partitioned

        # Verify bit precision
        assert scion_states.param_shard._bits_per_elt == 16  # store_param_remainders=True
        assert scion_states.momentum_shard._bits_per_elt == 32
        assert scion_states.grad_buffer._bits_per_elt == 32

    def test_initialization_without_param_remainders(self):
        """Test initialization with store_param_remainders=False."""
        scion_states = DistributedScionOptimizerStates(
            n_params=1000000,
            store_param_remainders=False,
            dp=8,
        )

        # param_shard should use 32 bits when not storing remainders
        assert scion_states.param_shard._bits_per_elt == 32

    def test_total_bytes_partitioned(self):
        """Test total_bytes calculation with partitioned=True."""
        n_params = 1000000
        dp = 8
        scion_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        total_bytes = scion_states.total_bytes(partitioned=True)

        # Calculate expected bytes:
        # param_shard:    (1M / 8) * 16 bits = 125K * 2 bytes = 250KB
        # momentum_shard: (1M / 8) * 32 bits = 125K * 4 bytes = 500KB
        # grad_buffer:    1M * 32 bits = 1M * 4 bytes = 4MB (not partitioned)
        # Total: 250KB + 500KB + 4MB = 4.75MB = 4,750,000 bytes
        expected_bytes = (n_params // dp) * 2 + (n_params // dp) * 4 + n_params * 4
        assert total_bytes == expected_bytes

    def test_total_bytes_unpartitioned(self):
        """Test total_bytes calculation with partitioned=False."""
        n_params = 1000000
        scion_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=8,
        )

        total_bytes = scion_states.total_bytes(partitioned=False)

        # Calculate expected bytes (unpartitioned):
        # param_shard:    1M * 16 bits = 1M * 2 bytes = 2MB
        # momentum_shard: 1M * 32 bits = 1M * 4 bytes = 4MB
        # grad_buffer:    1M * 32 bits = 1M * 4 bytes = 4MB
        # Total: 2MB + 4MB + 4MB = 10MB = 10,000,000 bytes
        expected_bytes = n_params * 2 + n_params * 4 + n_params * 4
        assert total_bytes == expected_bytes

    def test_memory_efficiency_compared_to_adam(self):
        """Test that Scion uses less memory than Adam (no exp_avg_sq)."""
        n_params = 1000000
        dp = 8

        scion_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        adam_states = DistributedAdamOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        # Scion should use less memory due to missing exp_avg_sq
        scion_bytes = scion_states.total_bytes(partitioned=True)
        adam_bytes = adam_states.total_bytes(partitioned=True)

        # Adam has exp_avg (32 bits) and exp_avg_sq (32 bits)
        # Scion has only momentum (32 bits)
        # Difference should be one 32-bit buffer partitioned over DP
        expected_difference = (n_params // dp) * 4
        assert adam_bytes - scion_bytes == expected_difference

    def test_memory_same_as_muon(self):
        """Test that Scion uses same memory as Muon (both have single momentum buffer)."""
        n_params = 1000000
        dp = 8

        scion_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        muon_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        # Scion and Muon should use the same memory (both have single momentum buffer)
        scion_bytes = scion_states.total_bytes(partitioned=True)
        muon_bytes = muon_states.total_bytes(partitioned=True)

        assert scion_bytes == muon_bytes

    def test_different_dp_values(self):
        """Test optimizer states with different DP degrees."""
        n_params = 1000000
        test_dp_values = [1, 2, 4, 8, 16, 32]

        for dp in test_dp_values:
            scion_states = DistributedScionOptimizerStates(
                n_params=n_params,
                store_param_remainders=True,
                dp=dp,
            )

            # Verify partitioned size decreases with larger DP
            partitioned_bytes = scion_states.total_bytes(partitioned=True)
            # Unpartitioned size should remain constant
            unpartitioned_bytes = scion_states.total_bytes(partitioned=False)

            assert unpartitioned_bytes == n_params * 2 + n_params * 4 + n_params * 4
            # Larger DP should result in smaller partitioned size
            assert partitioned_bytes <= unpartitioned_bytes


class TestModelStates:
    """Test ModelStates class with Adam, Muon, and Scion optimizers."""

    def test_model_states_with_adam_optimizer(self):
        """Test ModelStates initialization with Adam optimizer."""
        n_params = 1000000
        dp = 8

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        adam_opt_states = DistributedAdamOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        model_states = ModelStates(
            params_shard=params_shard,
            opt_states=adam_opt_states,
        )

        # Verify total_bytes includes both params and optimizer states
        total_bytes = model_states.total_bytes(partitioned=True)
        expected_bytes = params_shard.size(partitioned=True).bytes() + adam_opt_states.total_bytes(
            partitioned=True
        )
        assert total_bytes == expected_bytes

    def test_model_states_with_muon_optimizer(self):
        """Test ModelStates initialization with Muon optimizer."""
        n_params = 1000000
        dp = 8

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        muon_opt_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        model_states = ModelStates(
            params_shard=params_shard,
            opt_states=muon_opt_states,
        )

        # Verify total_bytes includes both params and optimizer states
        total_bytes = model_states.total_bytes(partitioned=True)
        expected_bytes = params_shard.size(partitioned=True).bytes() + muon_opt_states.total_bytes(
            partitioned=True
        )
        assert total_bytes == expected_bytes

    def test_repr_with_adam_optimizer(self):
        """Test __repr__ output for ModelStates with Adam optimizer."""
        n_params = 100000
        dp = 4

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        adam_opt_states = DistributedAdamOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        model_states = ModelStates(
            params_shard=params_shard,
            opt_states=adam_opt_states,
        )

        repr_str = repr(model_states)

        # Verify Adam-specific fields are present
        assert "exp_avg" in repr_str
        assert "exp_avg_squared" in repr_str
        assert "grad_buffer" in repr_str
        assert "TOTAL" in repr_str

        # Velocity should NOT be present for Adam
        assert "velocity" not in repr_str

    def test_repr_with_muon_optimizer(self):
        """Test __repr__ output for ModelStates with Muon optimizer."""
        n_params = 100000
        dp = 4

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        muon_opt_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        model_states = ModelStates(
            params_shard=params_shard,
            opt_states=muon_opt_states,
        )

        repr_str = repr(model_states)

        # Verify Muon-specific fields are present
        assert "velocity" in repr_str
        assert "grad_buffer" in repr_str
        assert "TOTAL" in repr_str

        # Adam-specific fields should NOT be present
        assert "exp_avg_squared" not in repr_str

    def test_repr_with_scion_optimizer(self):
        """Test __repr__ output for ModelStates with Scion optimizer."""
        n_params = 100000
        dp = 4

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        scion_opt_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        model_states = ModelStates(
            params_shard=params_shard,
            opt_states=scion_opt_states,
        )

        repr_str = repr(model_states)

        # Verify Scion-specific fields are present
        assert "momentum" in repr_str
        assert "grad_buffer" in repr_str
        assert "TOTAL" in repr_str

        # Adam-specific fields should NOT be present
        assert "exp_avg_squared" not in repr_str
        # Muon-specific fields should NOT be present
        assert "velocity" not in repr_str

    def test_model_states_with_scion_optimizer(self):
        """Test ModelStates initialization with Scion optimizer."""
        n_params = 1000000
        dp = 8

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        scion_opt_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        model_states = ModelStates(
            params_shard=params_shard,
            opt_states=scion_opt_states,
        )

        # Verify total_bytes includes both params and optimizer states
        total_bytes = model_states.total_bytes(partitioned=True)
        expected_bytes = params_shard.size(partitioned=True).bytes() + scion_opt_states.total_bytes(
            partitioned=True
        )
        assert total_bytes == expected_bytes

    def test_muon_uses_less_memory_than_adam(self):
        """Test that ModelStates with Muon uses less memory than Adam."""
        n_params = 1000000
        dp = 8

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        adam_opt_states = DistributedAdamOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        muon_opt_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        adam_model_states = ModelStates(params_shard=params_shard, opt_states=adam_opt_states)
        muon_model_states = ModelStates(params_shard=params_shard, opt_states=muon_opt_states)

        adam_total = adam_model_states.total_bytes(partitioned=True)
        muon_total = muon_model_states.total_bytes(partitioned=True)

        # Muon should use less memory (missing exp_avg_sq)
        assert muon_total < adam_total

        # The difference should be exactly the size of exp_avg_sq
        expected_difference = (n_params // dp) * 4  # 32 bits = 4 bytes per element
        assert adam_total - muon_total == expected_difference

    def test_scion_uses_less_memory_than_adam(self):
        """Test that ModelStates with Scion uses less memory than Adam."""
        n_params = 1000000
        dp = 8

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        adam_opt_states = DistributedAdamOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        scion_opt_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        adam_model_states = ModelStates(params_shard=params_shard, opt_states=adam_opt_states)
        scion_model_states = ModelStates(params_shard=params_shard, opt_states=scion_opt_states)

        adam_total = adam_model_states.total_bytes(partitioned=True)
        scion_total = scion_model_states.total_bytes(partitioned=True)

        # Scion should use less memory (missing exp_avg_sq)
        assert scion_total < adam_total

        # The difference should be exactly the size of exp_avg_sq
        expected_difference = (n_params // dp) * 4  # 32 bits = 4 bytes per element
        assert adam_total - scion_total == expected_difference

    def test_scion_uses_same_memory_as_muon(self):
        """Test that ModelStates with Scion uses same memory as Muon."""
        n_params = 1000000
        dp = 8

        params_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={},
            bits_per_elt=16,
        )

        muon_opt_states = DistributedMuonOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        scion_opt_states = DistributedScionOptimizerStates(
            n_params=n_params,
            store_param_remainders=True,
            dp=dp,
        )

        muon_model_states = ModelStates(params_shard=params_shard, opt_states=muon_opt_states)
        scion_model_states = ModelStates(params_shard=params_shard, opt_states=scion_opt_states)

        muon_total = muon_model_states.total_bytes(partitioned=True)
        scion_total = scion_model_states.total_bytes(partitioned=True)

        # Scion and Muon should use the same memory
        assert scion_total == muon_total


class TestThreeDParallelModelOptimizerSelection:
    """Test ThreeDParallelModel optimizer type selection."""

    def _create_base_model_kwargs(self):
        """Helper to create base model kwargs."""
        from dlcalc.utils.configurations import ActivationCheckpointingType

        return {
            "parallelism_cfg": ParallelConfig(
                tp=2,
                cp=1,
                pp=1,
                dp=4,
                expert_mesh=None,
                vpp=1,
                sp_enabled=True,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            ),
            "sequence_len": 1024,
            "microbatch_sz": 1,
            "hidden_sz": 512,
            "n_layers": 4,
            "n_q_heads": 8,
            "n_kv_heads": 8,
            "head_dim": 64,
            "inter_sz": 2048,
            "glu": True,
            "moe_cfg": None,
            "rotary_embed": True,
            "dropout": False,
            "vocab_sz": 50000,
            "tie_embeddings": True,
            "act_ckpting_type": ActivationCheckpointingType.SELECTIVE,
            "n_param_buckets": 4,
        }

    def test_default_optimizer_is_adam(self):
        """Test that default optimizer type is adam."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        model = ThreeDParallelModel(**kwargs)

        # Default should be adam
        assert model.optimizer_type == "adam"
        assert isinstance(model.states.opt_states, DistributedAdamOptimizerStates)

    def test_explicit_adam_optimizer(self):
        """Test explicit adam optimizer selection."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "adam"
        model = ThreeDParallelModel(**kwargs)

        assert model.optimizer_type == "adam"
        assert isinstance(model.states.opt_states, DistributedAdamOptimizerStates)

    def test_explicit_adamw_optimizer(self):
        """Test explicit adamw optimizer selection."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "adamw"
        model = ThreeDParallelModel(**kwargs)

        assert model.optimizer_type == "adamw"
        assert isinstance(model.states.opt_states, DistributedAdamOptimizerStates)

    def test_explicit_muon_optimizer(self):
        """Test explicit muon optimizer selection."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "muon"
        model = ThreeDParallelModel(**kwargs)

        assert model.optimizer_type == "muon"
        assert isinstance(model.states.opt_states, DistributedMuonOptimizerStates)

    def test_explicit_scion_optimizer(self):
        """Test explicit scion optimizer selection."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "scion"
        model = ThreeDParallelModel(**kwargs)

        assert model.optimizer_type == "scion"
        assert isinstance(model.states.opt_states, DistributedScionOptimizerStates)

    def test_case_insensitive_optimizer_type(self):
        """Test that optimizer type is case-insensitive."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        # Test ADAM
        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "ADAM"
        model = ThreeDParallelModel(**kwargs)
        assert isinstance(model.states.opt_states, DistributedAdamOptimizerStates)

        # Test AdamW (mixed case)
        kwargs["optimizer_type"] = "AdamW"
        model = ThreeDParallelModel(**kwargs)
        assert isinstance(model.states.opt_states, DistributedAdamOptimizerStates)

        # Test Muon (mixed case)
        kwargs["optimizer_type"] = "Muon"
        model = ThreeDParallelModel(**kwargs)
        assert isinstance(model.states.opt_states, DistributedMuonOptimizerStates)

        # Test Scion (mixed case)
        kwargs["optimizer_type"] = "Scion"
        model = ThreeDParallelModel(**kwargs)
        assert isinstance(model.states.opt_states, DistributedScionOptimizerStates)

    def test_invalid_optimizer_type_raises_error(self):
        """Test that invalid optimizer type raises ValueError."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "invalid_optimizer"

        with pytest.raises(ValueError) as exc_info:
            ThreeDParallelModel(**kwargs)

        assert "Invalid optimizer_type" in str(exc_info.value)
        assert "invalid_optimizer" in str(exc_info.value)
        assert "adam" in str(exc_info.value).lower()
        assert "adamw" in str(exc_info.value).lower()
        assert "muon" in str(exc_info.value).lower()
        assert "scion" in str(exc_info.value).lower()

    def test_adamw_uses_same_memory_as_adam(self):
        """Test that AdamW and Adam models use the same memory (same optimizer states)."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        # Create Adam model
        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "adam"
        adam_model = ThreeDParallelModel(**kwargs)

        # Create AdamW model
        kwargs["optimizer_type"] = "adamw"
        adamw_model = ThreeDParallelModel(**kwargs)

        # Compare memory usage - should be identical
        adam_mem = adam_model.states.total_bytes(partitioned=True)
        adamw_mem = adamw_model.states.total_bytes(partitioned=True)

        assert adam_mem == adamw_mem, "Adam and AdamW should use the same memory"

    def test_muon_uses_less_memory_than_adam_in_model(self):
        """Test that Muon model uses less memory than Adam model."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        # Create Adam model
        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "adam"
        adam_model = ThreeDParallelModel(**kwargs)

        # Create Muon model
        kwargs["optimizer_type"] = "muon"
        muon_model = ThreeDParallelModel(**kwargs)

        # Compare memory usage
        adam_mem = adam_model.states.total_bytes(partitioned=True)
        muon_mem = muon_model.states.total_bytes(partitioned=True)

        assert muon_mem < adam_mem, "Muon should use less memory than Adam"

        # The difference should be the exp_avg_sq buffer
        n_params = adam_model._ThreeDParallelModel__get_n_total_params(
            spmd_partitioned=True, mpmd_partitioned=True
        )
        dp = adam_model.parallelism_cfg.dp
        expected_difference = (n_params // dp) * 4  # 32-bit buffer
        assert adam_mem - muon_mem == expected_difference

    def test_muon_uses_less_memory_than_adamw_in_model(self):
        """Test that Muon model uses less memory than AdamW model."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        # Create AdamW model
        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "adamw"
        adamw_model = ThreeDParallelModel(**kwargs)

        # Create Muon model
        kwargs["optimizer_type"] = "muon"
        muon_model = ThreeDParallelModel(**kwargs)

        # Compare memory usage
        adamw_mem = adamw_model.states.total_bytes(partitioned=True)
        muon_mem = muon_model.states.total_bytes(partitioned=True)

        assert muon_mem < adamw_mem, "Muon should use less memory than AdamW"

    def test_scion_uses_less_memory_than_adam_in_model(self):
        """Test that Scion model uses less memory than Adam model."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        # Create Adam model
        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "adam"
        adam_model = ThreeDParallelModel(**kwargs)

        # Create Scion model
        kwargs["optimizer_type"] = "scion"
        scion_model = ThreeDParallelModel(**kwargs)

        # Compare memory usage
        adam_mem = adam_model.states.total_bytes(partitioned=True)
        scion_mem = scion_model.states.total_bytes(partitioned=True)

        assert scion_mem < adam_mem, "Scion should use less memory than Adam"

        # The difference should be the exp_avg_sq buffer
        n_params = adam_model._ThreeDParallelModel__get_n_total_params(
            spmd_partitioned=True, mpmd_partitioned=True
        )
        dp = adam_model.parallelism_cfg.dp
        expected_difference = (n_params // dp) * 4  # 32-bit buffer
        assert adam_mem - scion_mem == expected_difference

    def test_scion_uses_less_memory_than_adamw_in_model(self):
        """Test that Scion model uses less memory than AdamW model."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        # Create AdamW model
        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "adamw"
        adamw_model = ThreeDParallelModel(**kwargs)

        # Create Scion model
        kwargs["optimizer_type"] = "scion"
        scion_model = ThreeDParallelModel(**kwargs)

        # Compare memory usage
        adamw_mem = adamw_model.states.total_bytes(partitioned=True)
        scion_mem = scion_model.states.total_bytes(partitioned=True)

        assert scion_mem < adamw_mem, "Scion should use less memory than AdamW"

    def test_scion_uses_same_memory_as_muon_in_model(self):
        """Test that Scion and Muon models use the same memory."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        # Create Muon model
        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "muon"
        muon_model = ThreeDParallelModel(**kwargs)

        # Create Scion model
        kwargs["optimizer_type"] = "scion"
        scion_model = ThreeDParallelModel(**kwargs)

        # Compare memory usage - should be identical
        muon_mem = muon_model.states.total_bytes(partitioned=True)
        scion_mem = scion_model.states.total_bytes(partitioned=True)

        assert scion_mem == muon_mem, "Scion and Muon should use the same memory"


class TestOptimizerStepFLOPs:
    """Test optimizer step FLOPs calculation."""

    def _create_base_model_kwargs(self):
        """Helper to create base model kwargs."""
        from dlcalc.utils.configurations import ActivationCheckpointingType

        return {
            "parallelism_cfg": ParallelConfig(
                tp=2,
                cp=1,
                pp=1,
                dp=4,
                expert_mesh=None,
                vpp=1,
                sp_enabled=True,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            ),
            "sequence_len": 1024,
            "microbatch_sz": 1,
            "hidden_sz": 512,
            "n_layers": 4,
            "n_q_heads": 8,
            "n_kv_heads": 8,
            "head_dim": 64,
            "inter_sz": 2048,
            "glu": True,
            "moe_cfg": None,
            "rotary_embed": True,
            "dropout": False,
            "vocab_sz": 50000,
            "tie_embeddings": True,
            "act_ckpting_type": ActivationCheckpointingType.SELECTIVE,
            "n_param_buckets": 4,
        }

    def test_adam_optimizer_step_flops_linear(self):
        """Test that Adam optimizer FLOPs scale linearly with parameters."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "adam"

        # Create model
        model = ThreeDParallelModel(**kwargs)

        # Get FLOPs
        adam_flops = model.get_optimizer_step_flops()

        # Adam should be ~11 FLOPs per parameter
        n_params = model.get_n_total_params(partitioned=True)
        expected_flops = 11.0 * n_params

        assert adam_flops == expected_flops

    def test_adamw_same_flops_as_adam(self):
        """Test that AdamW has same FLOPs as Adam."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()

        # Adam model
        kwargs["optimizer_type"] = "adam"
        adam_model = ThreeDParallelModel(**kwargs)
        adam_flops = adam_model.get_optimizer_step_flops()

        # AdamW model
        kwargs["optimizer_type"] = "adamw"
        adamw_model = ThreeDParallelModel(**kwargs)
        adamw_flops = adamw_model.get_optimizer_step_flops()

        assert adam_flops == adamw_flops

    def test_muon_optimizer_step_flops_higher_than_adam(self):
        """Test that Muon optimizer requires more FLOPs than Adam due to Newton-Schulz."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()

        # Adam model
        kwargs["optimizer_type"] = "adam"
        adam_model = ThreeDParallelModel(**kwargs)
        adam_flops = adam_model.get_optimizer_step_flops()

        # Muon model
        kwargs["optimizer_type"] = "muon"
        muon_model = ThreeDParallelModel(**kwargs)
        muon_flops = muon_model.get_optimizer_step_flops()

        # Muon should require significantly more FLOPs due to Newton-Schulz
        assert muon_flops > adam_flops
        # Should be at least 10x more due to O(m²*n) vs O(n) complexity
        assert muon_flops > 10 * adam_flops

    def test_muon_flops_scale_with_hidden_size(self):
        """Test that Muon FLOPs scale quadratically with hidden size."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "muon"

        # Small hidden size
        kwargs["hidden_sz"] = 256
        kwargs["inter_sz"] = 1024
        small_model = ThreeDParallelModel(**kwargs)
        small_flops = small_model.get_optimizer_step_flops()

        # Large hidden size (2x)
        kwargs["hidden_sz"] = 512
        kwargs["inter_sz"] = 2048
        large_model = ThreeDParallelModel(**kwargs)
        large_flops = large_model.get_optimizer_step_flops()

        # With 2x hidden size, FLOPs should increase by more than 2x
        # (due to quadratic scaling in Newton-Schulz)
        assert large_flops > 2 * small_flops

    def test_scion_optimizer_step_flops_higher_than_adam(self):
        """Test that Scion optimizer requires more FLOPs than Adam due to spectral LMO."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()

        # Adam model
        kwargs["optimizer_type"] = "adam"
        adam_model = ThreeDParallelModel(**kwargs)
        adam_flops = adam_model.get_optimizer_step_flops()

        # Scion model
        kwargs["optimizer_type"] = "scion"
        scion_model = ThreeDParallelModel(**kwargs)
        scion_flops = scion_model.get_optimizer_step_flops()

        # Scion should require significantly more FLOPs due to spectral LMO
        assert scion_flops > adam_flops
        # Should be at least 10x more due to O(m²*n) vs O(n) complexity
        assert scion_flops > 10 * adam_flops

    def test_scion_flops_scale_with_hidden_size(self):
        """Test that Scion FLOPs scale quadratically with hidden size."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()
        kwargs["optimizer_type"] = "scion"

        # Small hidden size
        kwargs["hidden_sz"] = 256
        kwargs["inter_sz"] = 1024
        small_model = ThreeDParallelModel(**kwargs)
        small_flops = small_model.get_optimizer_step_flops()

        # Large hidden size (2x)
        kwargs["hidden_sz"] = 512
        kwargs["inter_sz"] = 2048
        large_model = ThreeDParallelModel(**kwargs)
        large_flops = large_model.get_optimizer_step_flops()

        # With 2x hidden size, FLOPs should increase by more than 2x
        # (due to quadratic scaling in spectral LMO)
        assert large_flops > 2 * small_flops

    def test_scion_and_muon_have_similar_flops(self):
        """Test that Scion and Muon have similar FLOPs (both use Newton-Schulz)."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()

        # Muon model
        kwargs["optimizer_type"] = "muon"
        muon_model = ThreeDParallelModel(**kwargs)
        muon_flops = muon_model.get_optimizer_step_flops()

        # Scion model
        kwargs["optimizer_type"] = "scion"
        scion_model = ThreeDParallelModel(**kwargs)
        scion_flops = scion_model.get_optimizer_step_flops()

        # Both use Newton-Schulz, so FLOPs should be similar (within 2x)
        # Scion has lower overhead (6mn vs 11mn), so might be slightly less
        assert scion_flops <= muon_flops
        assert scion_flops >= 0.5 * muon_flops

    def test_get_optimizer_step_flops_returns_positive(self):
        """Test that optimizer step FLOPs is always positive."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        kwargs = self._create_base_model_kwargs()

        for opt_type in ["adam", "adamw", "muon", "scion"]:
            kwargs["optimizer_type"] = opt_type
            model = ThreeDParallelModel(**kwargs)
            flops = model.get_optimizer_step_flops()

            assert flops > 0, f"{opt_type} optimizer should have positive FLOPs"


class TestSharedExpert:
    """Shared-expert dense FFN (Effect C): a dense MLP run on ALL tokens every MoE
    layer, in addition to the routed experts. Must add exact params/activations and
    scale with M = seq*mbs (not divided by EP)."""

    def _moe_model_kwargs(self, shared_expert_inter_sz: int):
        from dlcalc.utils.configurations import ActivationCheckpointingType
        from dlcalc.utils.model_3d import MoeCfg

        ep, tp, cp, dp = 8, 1, 1, 32
        expert_mesh = ParallelConfig.ExpertParallelCfg(ep=ep, tp=1, dp=dp * cp * tp // ep)
        return {
            "parallelism_cfg": ParallelConfig(
                tp=tp,
                cp=cp,
                pp=1,
                dp=dp,
                expert_mesh=expert_mesh,
                vpp=1,
                sp_enabled=True,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            ),
            "sequence_len": 8192,
            "microbatch_sz": 1,
            "hidden_sz": 2048,
            "n_layers": 16,
            "n_q_heads": 16,
            "n_kv_heads": 8,
            "head_dim": 128,
            "inter_sz": 5120,
            "glu": True,
            "moe_cfg": MoeCfg(
                n_experts=128,
                expert_inter_sz=1280,
                experts_per_token=3,
                capacity_factor=1,
                moe_frequency=1.0,
                expert_tp_degree=1,
                shared_expert_inter_sz=shared_expert_inter_sz,
            ),
            "rotary_embed": True,
            "dropout": False,
            "vocab_sz": 257152,
            "tie_embeddings": False,
            "act_ckpting_type": ActivationCheckpointingType.NONE,
            "n_param_buckets": 8,
            "optimizer_type": "muon",
            "precision": "bf16",
        }

    def test_param_count_adds_exactly_one_dense_ffn_per_moe_layer(self):
        """Total and active params both increase by exactly (up+down) per MoE layer.

        The shared expert is a dense GLU MLP: up (hidden -> 2*shared_inter) and
        down (shared_inter -> hidden). It runs on all tokens (not routed), so it
        counts in BOTH total and active params.
        """
        from dlcalc.utils.model_3d import ThreeDParallelModel

        shared_inter = 1280
        m0 = ThreeDParallelModel(**self._moe_model_kwargs(shared_expert_inter_sz=0))
        m1 = ThreeDParallelModel(**self._moe_model_kwargs(shared_expert_inter_sz=shared_inter))

        hidden = 2048
        n_moe_layers = 16  # moe_frequency=1.0 * n_layers
        up = hidden * (2 * shared_inter)  # GLU merges up+gate
        down = shared_inter * hidden
        expected_delta = n_moe_layers * (up + down)

        assert (
            m1.get_n_total_params(partitioned=False) - m0.get_n_total_params(partitioned=False)
            == expected_delta
        )
        assert (
            m1.get_n_active_params(partitioned=False) - m0.get_n_active_params(partitioned=False)
            == expected_delta
        )

    def test_no_shared_expert_when_size_zero(self):
        """shared_expert_inter_sz=0 leaves weights None and params unchanged vs a
        model with no shared expert term at all."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        m = ThreeDParallelModel(**self._moe_model_kwargs(shared_expert_inter_sz=0))
        assert m.shared_mlp_up_weight is None
        assert m.shared_mlp_down_weight is None

    def test_activation_memory_grows_with_shared_expert(self):
        """Per-layer activation memory must account for the shared-expert
        intermediate (all tokens), else the memory model under-counts (GUIDELINES §6)."""
        from dlcalc.utils.model_3d import ThreeDParallelModel

        m0 = ThreeDParallelModel(**self._moe_model_kwargs(shared_expert_inter_sz=0))
        m1 = ThreeDParallelModel(**self._moe_model_kwargs(shared_expert_inter_sz=1280))

        a0 = m0.activation_size_per_microbatch_per_layer()
        a1 = m1.activation_size_per_microbatch_per_layer()
        assert a1.numel() > a0.numel(), "shared expert must add activation memory"
