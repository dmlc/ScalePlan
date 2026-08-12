import dataclasses
from enum import Enum

from .configurations import ActivationCheckpointingType
from .data import Size, TensorRepr
from .math import safe_divide


@dataclasses.dataclass
class ParallelConfig:
    class ZeroLevel(int, Enum):
        NONE = 0
        PARTITION_OPTIMIZER = 1
        PARTITION_GRADIENTS = 2
        PARTITION_PARAMETERS = 3

    @dataclasses.dataclass
    class ExpertParallelCfg:
        ep: int  # Expert Parallel (EP) degree
        tp: int
        dp: int

    tp: int  # Tensor Parallel (TP) degree
    cp: int  # Context Parallel (CP) degree
    pp: int  # Pipeline Parallel (PP) degree
    dp: int  # Data Parallel (DP) degree

    expert_mesh: ExpertParallelCfg | None

    vpp: int  # Virtual Pipeline Parallel (VPP) degree

    sp_enabled: bool  # Sequence Parallel (SP) enablement

    zero_level: ZeroLevel

    def __post_init__(self) -> None:
        if self.expert_mesh is not None:
            assert (
                self.expert_mesh.ep * self.expert_mesh.tp * self.expert_mesh.dp
                == self.dp * self.cp * self.tp
            )

    def world_size(self) -> int:
        return self.tp * self.cp * self.pp * self.dp


class DistributedAdamOptimizerStates:
    """Optimizer states from Apex DistributedFusedAdam.
    see: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md
    and: https://github.com/NVIDIA/apex/blob/master/apex/contrib/optimizers/distributed_fused_adam.py

    the distributed optimizer recieves a set of parameters to manage that are already
    model parallel partitioned. Internally, it will additionally partition states
    over DP, so optimizer states end up being partitioned over MP * DP, but
    the diestributed optimizer doesn't have any concept of model parallelism.

    NOTE: shards will be larger in reality due to alignment requirements and
    unfilled buckets.
    """

    # NOTE: n_params here is meant to be the parameters in a pipeline stage.
    def __init__(self, n_params: int, store_param_remainders: bool, dp: int) -> None:
        self.param_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            # apex has a optimization to avoid storing information that's redundant
            # between fp32 and fp16 weights. it can instead store an extra 16
            # bits of precision, which can be combined with bf16 weights to yield
            # fp32 weights
            bits_per_elt=16 if store_param_remainders else 32,
            enforce_evenly_partitionable=False,
        )
        self.exp_avg_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )
        self.exp_avg_sq_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )
        # FIX: grad_buffer should be DP-partitioned after reduce-scatter
        # Previous comment was misleading - in ZeRO-1 (PARTITION_OPTIMIZER), the
        # grad_buffer is reduce-scattered after each microbatch, so it's partitioned.
        # Only during gradient accumulation within a microbatch is it temporarily full,
        # but peak memory usage is after reduce-scatter when it's DP-partitioned.
        self.grad_buffer = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},  # DP-partitioned after reduce-scatter
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )

    def total_bytes(self, partitioned: bool) -> int:
        return _sum(
            self.param_shard.size(partitioned=partitioned).bytes(),
            self.exp_avg_shard.size(partitioned=partitioned).bytes(),
            self.exp_avg_sq_shard.size(partitioned=partitioned).bytes(),
            self.grad_buffer.size(partitioned=partitioned).bytes(),
        )


class DistributedMuonOptimizerStates:
    """Optimizer states for Muon optimizer with distributed parameter sharding.

    Muon uses Newton-Schulz orthogonalization with momentum for optimization.
    Compared to Adam:
    - Uses a single velocity (momentum) buffer instead of exp_avg and exp_avg_sq
    - Newton-Schulz orthogonalization is computed on-the-fly (no persistent state)
    - Simpler state management with lower memory overhead

    State partitioning follows the same pattern as DistributedFusedAdam:
    - Parameters and optimizer states are partitioned over DP dimension
    - Gradient buffer holds full (non-DP-partitioned) gradients during accumulation

    NOTE: shards will be larger in reality due to alignment requirements and
    unfilled buckets.
    """

    def __init__(self, n_params: int, store_param_remainders: bool, dp: int) -> None:
        """Initialize Muon optimizer states.

        Args:
            n_params: Number of parameters in a pipeline stage
            store_param_remainders: If True, store 16-bit remainders instead of full fp32 params
            dp: Data parallel degree for partitioning optimizer states
        """
        self.param_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            # apex has a optimization to avoid storing information that's redundant
            # between fp32 and fp16 weights. it can instead store an extra 16
            # bits of precision, which can be combined with bf16 weights to yield
            # fp32 weights
            bits_per_elt=16 if store_param_remainders else 32,
            enforce_evenly_partitionable=False,
        )
        self.velocity_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )
        # FIX: grad_buffer should be DP-partitioned after reduce-scatter
        # Previous comment was misleading - in ZeRO-1 (PARTITION_OPTIMIZER), the
        # grad_buffer is reduce-scattered after each microbatch, so it's partitioned.
        # Only during gradient accumulation within a microbatch is it temporarily full,
        # but peak memory usage is after reduce-scatter when it's DP-partitioned.
        self.grad_buffer = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},  # DP-partitioned after reduce-scatter
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )

    def total_bytes(self, partitioned: bool) -> int:
        """Calculate total memory usage for all optimizer states.

        Args:
            partitioned: If True, return size after DP partitioning; otherwise unpartitioned size

        Returns:
            Total bytes across all optimizer state tensors
        """
        return _sum(
            self.param_shard.size(partitioned=partitioned).bytes(),
            self.velocity_shard.size(partitioned=partitioned).bytes(),
            self.grad_buffer.size(partitioned=partitioned).bytes(),
        )


class DistributedScionOptimizerStates:
    """Optimizer states for Scion optimizer with distributed parameter sharding.

    Scion (Spectral Constrained Ion) uses spectral norm constraints with linear minimization
    oracles (LMOs) and momentum for optimization.
    Compared to Adam:
    - Uses a single momentum buffer instead of exp_avg and exp_avg_sq
    - Spectral LMO is computed on-the-fly using Newton-Schulz iteration (no persistent state)
    - Simpler state management with lower memory overhead

    State partitioning follows the same pattern as DistributedFusedAdam:
    - Parameters and optimizer states are partitioned over DP dimension
    - Gradient buffer holds full (non-DP-partitioned) gradients during accumulation

    NOTE: shards will be larger in reality due to alignment requirements and
    unfilled buckets.
    """

    def __init__(self, n_params: int, store_param_remainders: bool, dp: int) -> None:
        """Initialize Scion optimizer states.

        Args:
            n_params: Number of parameters in a pipeline stage
            store_param_remainders: If True, store 16-bit remainders instead of full fp32 params
            dp: Data parallel degree for partitioning optimizer states
        """
        self.param_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            # apex has a optimization to avoid storing information that's redundant
            # between fp32 and fp16 weights. it can instead store an extra 16
            # bits of precision, which can be combined with bf16 weights to yield
            # fp32 weights
            bits_per_elt=16 if store_param_remainders else 32,
            enforce_evenly_partitionable=False,
        )
        self.momentum_shard = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )
        # FIX: grad_buffer should be DP-partitioned after reduce-scatter
        # Previous comment was misleading - in ZeRO-1 (PARTITION_OPTIMIZER), the
        # grad_buffer is reduce-scattered after each microbatch, so it's partitioned.
        # Only during gradient accumulation within a microbatch is it temporarily full,
        # but peak memory usage is after reduce-scatter when it's DP-partitioned.
        self.grad_buffer = TensorRepr(
            unpartitioned_shape=(n_params,),
            partition_spec={0: dp},  # DP-partitioned after reduce-scatter
            bits_per_elt=32,
            enforce_evenly_partitionable=False,
        )

    def total_bytes(self, partitioned: bool) -> int:
        """Calculate total memory usage for all optimizer states.

        Args:
            partitioned: If True, return size after DP partitioning; otherwise unpartitioned size

        Returns:
            Total bytes across all optimizer state tensors
        """
        return _sum(
            self.param_shard.size(partitioned=partitioned).bytes(),
            self.momentum_shard.size(partitioned=partitioned).bytes(),
            self.grad_buffer.size(partitioned=partitioned).bytes(),
        )

@dataclasses.dataclass
class ModelStates:
    """Tracks persistent tensor allocations kept throughput training."""

    params_shard: TensorRepr
    opt_states: (
        DistributedAdamOptimizerStates
        | DistributedMuonOptimizerStates
        | DistributedScionOptimizerStates
    )

    def total_bytes(self, partitioned: bool) -> int:
        return self.params_shard.size(
            partitioned=partitioned
        ).bytes() + self.opt_states.total_bytes(partitioned=partitioned)

    def __repr__(self) -> str:
        params_bytes = self.params_shard.size(partitioned=True).bytes()
        opt_states_bytes = self.opt_states.total_bytes(partitioned=True)

        base_info = [
            f"params          : {self.params_shard.size(partitioned=True)}",
            f"params (opt)    : {self.opt_states.param_shard.size(partitioned=True)}",
        ]

        # Add optimizer-specific state information
        if isinstance(self.opt_states, DistributedAdamOptimizerStates):
            opt_specific = [
                f"exp_avg         : {self.opt_states.exp_avg_shard.size(partitioned=True)}",
                f"exp_avg_squared : {self.opt_states.exp_avg_sq_shard.size(partitioned=True)}",
            ]
        elif isinstance(self.opt_states, DistributedMuonOptimizerStates):
            opt_specific = [
                f"velocity        : {self.opt_states.velocity_shard.size(partitioned=True)}",
            ]
        elif isinstance(self.opt_states, DistributedScionOptimizerStates):
            opt_specific = [
                f"momentum        : {self.opt_states.momentum_shard.size(partitioned=True)}",
            ]
        else:
            opt_specific = []

        footer = [
            f"grad_buffer     : {self.opt_states.grad_buffer.size(partitioned=True)}",
            f"TOTAL           : {(params_bytes + opt_states_bytes) / (1024**3):.2f}GiB",
        ]

        return "\n".join(base_info + opt_specific + footer)


def _sum(*summands: int) -> int:
    return sum(list(summands))


@dataclasses.dataclass
class MoeCfg:
    n_experts: int
    expert_inter_sz: int
    experts_per_token: int
    capacity_factor: float
    moe_frequency: float
    expert_tp_degree: int
    # Dense "shared expert" FFN run on ALL tokens every MoE layer (in addition to
    # the routed experts), à la DeepSeek-MoE / Qwen-MoE. 0 = no shared expert.
    # MLflow logs this as `moe_shared_expert_intermediate_size` (== moe_ffn_hidden_size
    # for every validated arch). It is NOT expert-parallel-sharded: it is a plain
    # dense MLP, TP-sharded like the non-MoE MLP. See ALL_MODELS_VALIDATION_FINDINGS.md
    # "Effect C".
    shared_expert_inter_sz: int = 0


@dataclasses.dataclass
class ThreeDParallelModel:
    """Representation of a 3D parallel transformer model."""

    parallelism_cfg: ParallelConfig

    # Instance variables set in __post_init__
    router_weight: TensorRepr | None = dataclasses.field(init=False, default=None)
    mlp_up_exp_weight: TensorRepr | None = dataclasses.field(init=False, default=None)
    mlp_down_exp_weight: TensorRepr | None = dataclasses.field(init=False, default=None)
    shared_mlp_up_weight: TensorRepr | None = dataclasses.field(init=False, default=None)
    shared_mlp_down_weight: TensorRepr | None = dataclasses.field(init=False, default=None)
    vocab_sz_padded: int = dataclasses.field(init=False, default=0)

    sequence_len: int
    microbatch_sz: int

    hidden_sz: int

    n_layers: int

    n_q_heads: int
    n_kv_heads: int  # num_query_groups if GQA
    head_dim: int

    inter_sz: int
    glu: bool
    moe_cfg: MoeCfg | None

    rotary_embed: bool

    dropout: bool

    vocab_sz: int  # Original vocabulary size
    tie_embeddings: bool

    act_ckpting_type: ActivationCheckpointingType

    n_param_buckets: int

    # Vocabulary padding configuration
    vocab_padding_size: int = 64  # Default padding size for hardware efficiency

    # Optimizer configuration
    optimizer_type: str = "adam"  # Options: "adam", "adamw", "muon", "scion"

    # Precision configuration
    # Can be overridden by passing bits_per_parameter directly, or will be inferred from precision string
    precision: str | None = None  # Options: "fp8", "fp16", "bf16", "fp32"
    bits_per_parameter: int = 16  # Default to 16-bit, can be overridden
    bits_per_grad: int = 32
    bits_per_optim_state: int = 32

    def __post_init__(self) -> None:
        # Convert precision string to bits if provided
        if self.precision is not None:
            precision_map = {
                "fp8": 8,
                "fp16": 16,
                "bf16": 16,
                "fp32": 32,
            }
            precision_lower = self.precision.lower()
            if precision_lower in precision_map:
                self.bits_per_parameter = precision_map[precision_lower]
            else:
                raise ValueError(
                    f"Invalid precision '{self.precision}'. "
                    f"Supported options: {list(precision_map.keys())}"
                )

        if self.n_layers % (self.parallelism_cfg.pp * self.parallelism_cfg.vpp) != 0:
            raise ValueError(
                f"number of layers {self.n_layers} is not divisible by the product "
                f"of PP={self.parallelism_cfg.pp} and VPP={self.parallelism_cfg.vpp}"
            )

        if self.moe_cfg:
            if not float(self.moe_cfg.moe_frequency * self.n_layers).is_integer():
                raise ValueError(
                    f"invalid moe frequency {self.moe_cfg.moe_frequency} for layer number {self.n_layers}"
                )
            self.n_moe_layers = int(self.moe_cfg.moe_frequency * self.n_layers)
            self.n_nml_layers = self.n_layers - self.n_moe_layers
        else:
            self.n_moe_layers = 0
            self.n_nml_layers = self.n_layers

        n_experts = self.moe_cfg.n_experts if self.moe_cfg else 1

        # Calculate padded vocabulary size for hardware efficiency
        self.vocab_sz_padded = self._pad_vocab_size(self.vocab_sz, self.vocab_padding_size)

        self.embed_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz, self.vocab_sz_padded),
            partition_spec={1: self.parallelism_cfg.tp},  # vocab-parallel
            bits_per_elt=self.bits_per_parameter,
        )

        self.pre_attn_norm_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz,),
            partition_spec={},  # replicated
            bits_per_elt=self.bits_per_parameter,
        )
        self.qkv_weight = TensorRepr(
            unpartitioned_shape=(
                self.hidden_sz,
                # following common practice of merging Q + K + V matmuls.
                (self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim,
            ),
            partition_spec={1: self.parallelism_cfg.tp},  # col parallel
            bits_per_elt=self.bits_per_parameter,
        )
        self.attn_out_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz, self.hidden_sz),
            partition_spec={0: self.parallelism_cfg.tp},  # row parallel
            bits_per_elt=self.bits_per_parameter,
        )

        self.pre_mlp_norm_weight = TensorRepr(
            unpartitioned_shape=(self.hidden_sz,),
            partition_spec={},
            bits_per_elt=self.bits_per_parameter,
        )
        self.mlp_up_weight = TensorRepr(
            unpartitioned_shape=(
                self.hidden_sz,
                # following common practice of merging up + gate matmuls in the event
                # we're using GLU.
                (self.inter_sz * 2) if self.glu else self.inter_sz,
            ),
            partition_spec={1: self.parallelism_cfg.tp},  # col parallel
            bits_per_elt=self.bits_per_parameter,
        )
        self.mlp_down_weight = TensorRepr(
            unpartitioned_shape=(self.inter_sz, self.hidden_sz),
            partition_spec={0: self.parallelism_cfg.tp},  # row parallel
            bits_per_elt=self.bits_per_parameter,
        )

        if self.moe_cfg is not None:
            assert self.parallelism_cfg.expert_mesh is not None
            assert self.moe_cfg is not None

            self.router_weight = TensorRepr(
                unpartitioned_shape=(self.hidden_sz, self.moe_cfg.n_experts),
                partition_spec={},
                bits_per_elt=self.bits_per_parameter,
            )
            self.mlp_up_exp_weight = (
                TensorRepr(
                    # following common practice of merging up + gate matmuls in the event
                    # we're using GLU.
                    unpartitioned_shape=(
                        n_experts,
                        self.hidden_sz,
                        (self.moe_cfg.expert_inter_sz * 2)
                        if self.glu
                        else self.moe_cfg.expert_inter_sz,
                    ),
                    partition_spec={
                        0: self.parallelism_cfg.expert_mesh.ep,
                        2: self.parallelism_cfg.expert_mesh.tp,
                    },  # col parallel
                    bits_per_elt=self.bits_per_parameter,
                )
                if self.moe_cfg is not None
                else None
            )
            self.mlp_down_exp_weight = (
                TensorRepr(
                    unpartitioned_shape=(
                        n_experts,
                        self.moe_cfg.expert_inter_sz,
                        self.hidden_sz,
                    ),
                    partition_spec={
                        0: self.parallelism_cfg.expert_mesh.ep,
                        1: self.parallelism_cfg.expert_mesh.tp,
                    },  # row parallel
                    bits_per_elt=self.bits_per_parameter,
                )
                if self.moe_cfg is not None
                else None
            )
            # Shared-expert dense MLP (run on all tokens every MoE layer). Sharded
            # like the non-MoE MLP: up col-parallel over TP, down row-parallel over
            # TP. NOT expert-parallel. None when shared_expert_inter_sz == 0.
            if self.moe_cfg.shared_expert_inter_sz > 0:
                self.shared_mlp_up_weight = TensorRepr(
                    unpartitioned_shape=(
                        self.hidden_sz,
                        (self.moe_cfg.shared_expert_inter_sz * 2)
                        if self.glu
                        else self.moe_cfg.shared_expert_inter_sz,
                    ),
                    partition_spec={1: self.parallelism_cfg.tp},  # col parallel
                    bits_per_elt=self.bits_per_parameter,
                )
                self.shared_mlp_down_weight = TensorRepr(
                    unpartitioned_shape=(
                        self.moe_cfg.shared_expert_inter_sz,
                        self.hidden_sz,
                    ),
                    partition_spec={0: self.parallelism_cfg.tp},  # row parallel
                    bits_per_elt=self.bits_per_parameter,
                )
            else:
                self.shared_mlp_up_weight = None
                self.shared_mlp_down_weight = None
        else:
            self.router_weight = None
            self.mlp_up_exp_weight = None
            self.mlp_down_exp_weight = None
            self.shared_mlp_up_weight = None
            self.shared_mlp_down_weight = None

        if self.parallelism_cfg.zero_level not in (
            ParallelConfig.ZeroLevel.NONE,
            ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
        ):
            raise NotImplementedError(
                f"zero_level={self.parallelism_cfg.zero_level} not supported "
                "(only NONE and PARTITION_OPTIMIZER are implemented)"
            )

        # Create optimizer states based on optimizer_type
        n_params = self.__get_n_total_params(
            spmd_partitioned=True,
            mpmd_partitioned=True,
        )
        store_param_remainders = True  # TODO. should be configurable

        # When zero_level=NONE, optimizer states and gradient buffer are
        # replicated across DP ranks (no partitioning). Passing dp=1 makes
        # TensorRepr partition by 1 (i.e. full size).
        opt_state_dp = (
            self.parallelism_cfg.dp
            if self.parallelism_cfg.zero_level == ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER
            else 1
        )

        optimizer_type_lower = self.optimizer_type.lower()
        opt_states: (
            DistributedAdamOptimizerStates
            | DistributedMuonOptimizerStates
            | DistributedScionOptimizerStates
        )
        if optimizer_type_lower in ("adam", "adamw"):
            # Both Adam and AdamW use the same optimizer states (exp_avg, exp_avg_sq)
            # AdamW differs only in weight decay implementation (decoupled)
            opt_states = DistributedAdamOptimizerStates(
                n_params=n_params,
                store_param_remainders=store_param_remainders,
                dp=opt_state_dp,
            )
        elif optimizer_type_lower == "muon":
            opt_states = DistributedMuonOptimizerStates(
                n_params=n_params,
                store_param_remainders=store_param_remainders,
                dp=opt_state_dp,
            )
        elif optimizer_type_lower == "scion":
            opt_states = DistributedScionOptimizerStates(
                n_params=n_params,
                store_param_remainders=store_param_remainders,
                dp=opt_state_dp,
            )
        else:
            raise ValueError(
                f"Invalid optimizer_type '{self.optimizer_type}'. "
                f"Supported options: 'adam', 'adamw', 'muon', 'scion'"
            )

        self.states = ModelStates(
            params_shard=TensorRepr(
                unpartitioned_shape=(n_params,),
                partition_spec={},
                bits_per_elt=self.bits_per_parameter,
                enforce_evenly_partitionable=False,
            ),
            opt_states=opt_states,
        )

    def get_single_microbatch_fwd_flops(self) -> float:
        return (
            2  # FLOPs/MAC
            * 1  # factor for forward only (1 GEMMs per op)
            * self.microbatch_sz
            * self.sequence_len
            * self.__get_n_active_params(partitioned=False)
        )

    def get_single_microbatch_bwd_flops(self) -> float:
        return (
            2  # FLOPs/MAC
            * 2  # factor for backward only (2 GEMMs per op)
            * self.microbatch_sz
            * self.sequence_len
            * self.__get_n_active_params(partitioned=False)
        )

    def get_n_total_params(self, partitioned: bool) -> int:
        return self.__get_n_total_params(
            spmd_partitioned=partitioned,
            mpmd_partitioned=partitioned,
        )

    def get_n_expert_params_per_stage(self, partitioned: bool) -> int:
        """Routed-expert MLP params on the most-loaded PP stage.

        These are the ONLY params reduced over the expert-DP (`expert_mesh.dp`)
        group during the DP gradient reduction; every other param (embed/lm-head,
        attention, dense MLP, shared expert, router, norms) is replicated over the
        full `dp` group and reduced there. Splitting the two is required for an
        honest DP-comm model: lumping all params into the expert-DP reduction
        under-counts (and, when expert_dp==1, zeroes) the dense-param reduction.
        Returns 0 for a dense (non-MoE) model.
        """
        if self.moe_cfg is None:
            return 0
        assert self.mlp_up_exp_weight is not None
        assert self.mlp_down_exp_weight is not None
        expert_params_per_moe_layer = _sum(
            self.mlp_up_exp_weight.numel(partitioned=partitioned),
            self.mlp_down_exp_weight.numel(partitioned=partitioned),
        )
        return self.__n_layers(mpmd_partitioned=partitioned, moe=True) * expert_params_per_moe_layer

    def get_n_dense_params_per_stage(self, partitioned: bool) -> int:
        """All non-routed-expert params on the most-loaded PP stage (see
        get_n_expert_params_per_stage). Reduced over the full `dp` group."""
        n_total = self.get_n_total_params(partitioned=partitioned)
        n_expert = self.get_n_expert_params_per_stage(partitioned=partitioned)
        return n_total - n_expert

    def get_n_active_params(self, partitioned: bool) -> int:
        return self.__get_n_active_params(partitioned=partitioned)

    def get_optimizer_step_flops(self) -> float:
        """Calculate FLOPs required for optimizer step.

        Adam/AdamW: O(n) - approximately 11 FLOPs per parameter
        Muon: O(m² * n) - Newton-Schulz orthogonalization dominates
        Scion: O(m² * n) - Spectral LMO via Newton-Schulz iteration dominates

        Returns:
            Total FLOPs for one optimizer step
        """
        optimizer_type_lower = self.optimizer_type.lower()

        if optimizer_type_lower in ("adam", "adamw"):
            # Adam/AdamW: ~11 FLOPs per parameter
            # - exp_avg update: 3 FLOPs (beta1*exp_avg + (1-beta1)*grad)
            # - exp_avg_sq update: 4 FLOPs (beta2*exp_avg_sq + (1-beta2)*grad^2)
            # - param update: 4 FLOPs (param - lr*exp_avg/(sqrt(exp_avg_sq)+eps))
            n_params = self.__get_n_total_params(
                spmd_partitioned=True,
                mpmd_partitioned=True,
            )
            return 11.0 * n_params

        elif optimizer_type_lower == "muon":
            # Muon: Newton-Schulz orthogonalization is O(m² * n) per weight matrix
            # For each 2D weight matrix of shape (m, n) where m <= n after transpose:
            # - Momentum + normalization: ~3*m*n FLOPs
            # - Newton-Schulz (5 iterations): ~10*m²*n FLOPs per iteration
            # - Parameter update: ~m*n FLOPs
            # Total: ~(3 + 50 + 1)*m*n = 54*m*n for small m, but ~50*m²*n when m is large
            #
            # We'll calculate for each major weight matrix:

            total_flops = 0.0
            n_layers_per_stage = self.__n_layers(mpmd_partitioned=True, moe=False)
            n_moe_layers_per_stage = self.__n_layers(mpmd_partitioned=True, moe=True)

            # For each regular layer
            for _ in range(n_layers_per_stage):
                # QKV projection: (hidden_sz, (n_q_heads + 2*n_kv_heads) * head_dim)
                m = self.hidden_sz
                n = (self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim
                m_small = min(m, n)
                n_large = max(m, n)
                total_flops += self.__muon_weight_matrix_flops(m_small, n_large)

                # Attention output: (hidden_sz, hidden_sz)
                m = n = self.hidden_sz
                total_flops += self.__muon_weight_matrix_flops(m, n)

                # MLP up/gate: (hidden_sz, inter_sz * 2) for GLU
                m = self.hidden_sz
                n = self.inter_sz * 2 if self.glu else self.inter_sz
                m_small = min(m, n)
                n_large = max(m, n)
                total_flops += self.__muon_weight_matrix_flops(m_small, n_large)

                # MLP down: (inter_sz, hidden_sz)
                m = self.inter_sz
                n = self.hidden_sz
                m_small = min(m, n)
                n_large = max(m, n)
                total_flops += self.__muon_weight_matrix_flops(m_small, n_large)

                # Layer norms (small, treat as O(n))
                total_flops += 11.0 * 2 * self.hidden_sz

            # For each MoE layer
            if self.moe_cfg is not None:
                for _ in range(n_moe_layers_per_stage):
                    # Router is small, treat as O(n)
                    total_flops += 11.0 * self.hidden_sz * self.moe_cfg.n_experts

                    # Expert weights - divided by EP degree
                    n_experts_per_device = safe_divide(
                        self.moe_cfg.n_experts,
                        self.parallelism_cfg.expert_mesh.ep
                        if self.parallelism_cfg.expert_mesh
                        else 1,
                    )

                    # Up/gate per expert: (hidden_sz, expert_inter_sz * 2)
                    m = self.hidden_sz
                    n = self.moe_cfg.expert_inter_sz * 2 if self.glu else self.moe_cfg.expert_inter_sz
                    m_small = min(m, n)
                    n_large = max(m, n)
                    total_flops += n_experts_per_device * self.__muon_weight_matrix_flops(
                        m_small, n_large
                    )

                    # Down per expert: (expert_inter_sz, hidden_sz)
                    m = self.moe_cfg.expert_inter_sz
                    n = self.hidden_sz
                    m_small = min(m, n)
                    n_large = max(m, n)
                    total_flops += n_experts_per_device * self.__muon_weight_matrix_flops(
                        m_small, n_large
                    )

                    # Layer norms
                    total_flops += 11.0 * 2 * self.hidden_sz

            # Embedding and LM head (large vocab, treat as O(n) for simplicity)
            # As vocab is very large compared to hidden_sz, Newton-Schulz would be expensive
            # but we'll approximate as O(n) for now
            n_embed_params = (
                1 if (self.parallelism_cfg.pp > 1) or self.tie_embeddings else 2
            ) * self.embed_weight.numel(partitioned=True)
            total_flops += 11.0 * n_embed_params

            return total_flops

        elif optimizer_type_lower == "scion":
            # Scion: Spectral constrained optimizer using LMO with Newton-Schulz iteration
            # For each 2D weight matrix of shape (m, n) where m <= n after transpose:
            # - Momentum update: buf = (1-momentum)*buf + momentum*grad → 3*m*n FLOPs
            # - Spectral LMO (Newton-Schulz, 5 iterations): ~20*m²*n FLOPs per iteration
            # - Constrained parameter update: p = (1-lr)*p - lr*update → 3*m*n FLOPs
            # Total: ~6*m*n + 20*m²*n for small m, but ~20*m²*n when m is large
            #
            # Similar structure to Muon but with different constants

            total_flops = 0.0
            n_layers_per_stage = self.__n_layers(mpmd_partitioned=True, moe=False)
            n_moe_layers_per_stage = self.__n_layers(mpmd_partitioned=True, moe=True)

            # For each regular layer
            for _ in range(n_layers_per_stage):
                # QKV projection: (hidden_sz, (n_q_heads + 2*n_kv_heads) * head_dim)
                m = self.hidden_sz
                n = (self.n_q_heads + 2 * self.n_kv_heads) * self.head_dim
                m_small = min(m, n)
                n_large = max(m, n)
                total_flops += self.__scion_weight_matrix_flops(m_small, n_large)

                # Attention output: (hidden_sz, hidden_sz)
                m = n = self.hidden_sz
                total_flops += self.__scion_weight_matrix_flops(m, n)

                # MLP up/gate: (hidden_sz, inter_sz * 2) for GLU
                m = self.hidden_sz
                n = self.inter_sz * 2 if self.glu else self.inter_sz
                m_small = min(m, n)
                n_large = max(m, n)
                total_flops += self.__scion_weight_matrix_flops(m_small, n_large)

                # MLP down: (inter_sz, hidden_sz)
                m = self.inter_sz
                n = self.hidden_sz
                m_small = min(m, n)
                n_large = max(m, n)
                total_flops += self.__scion_weight_matrix_flops(m_small, n_large)

                # Layer norms (small, treat as O(n))
                total_flops += 11.0 * 2 * self.hidden_sz

            # For each MoE layer
            if self.moe_cfg is not None:
                for _ in range(n_moe_layers_per_stage):
                    # Router is small, treat as O(n)
                    total_flops += 11.0 * self.hidden_sz * self.moe_cfg.n_experts

                    # Expert weights - divided by EP degree
                    n_experts_per_device = safe_divide(
                        self.moe_cfg.n_experts,
                        self.parallelism_cfg.expert_mesh.ep
                        if self.parallelism_cfg.expert_mesh
                        else 1,
                    )

                    # Up/gate per expert: (hidden_sz, expert_inter_sz * 2)
                    m = self.hidden_sz
                    n = self.moe_cfg.expert_inter_sz * 2 if self.glu else self.moe_cfg.expert_inter_sz
                    m_small = min(m, n)
                    n_large = max(m, n)
                    total_flops += n_experts_per_device * self.__scion_weight_matrix_flops(
                        m_small, n_large
                    )

                    # Down per expert: (expert_inter_sz, hidden_sz)
                    m = self.moe_cfg.expert_inter_sz
                    n = self.hidden_sz
                    m_small = min(m, n)
                    n_large = max(m, n)
                    total_flops += n_experts_per_device * self.__scion_weight_matrix_flops(
                        m_small, n_large
                    )

                    # Layer norms
                    total_flops += 11.0 * 2 * self.hidden_sz

            # Embedding and LM head (large vocab, treat as O(n) for simplicity)
            # As vocab is very large compared to hidden_sz, Newton-Schulz would be expensive
            # but we'll approximate as O(n) for now
            n_embed_params = (
                1 if (self.parallelism_cfg.pp > 1) or self.tie_embeddings else 2
            ) * self.embed_weight.numel(partitioned=True)
            total_flops += 11.0 * n_embed_params

            return total_flops

        else:
            # Fallback for unknown optimizers
            n_params = self.__get_n_total_params(
                spmd_partitioned=True,
                mpmd_partitioned=True,
            )
            return 11.0 * n_params

    def __muon_weight_matrix_flops(self, m: int, n: int) -> float:
        """Calculate FLOPs for Muon optimizer on a single weight matrix.

        Based on the reference implementation:
        1. Momentum update: velocity = momentum*velocity + grad → 2mn FLOPs
        2. Nesterov lookahead: U = grad + momentum*velocity → 2mn FLOPs
        3. Normalization: fro_norm + division → 3mn FLOPs
        4. Newton-Schulz (5 iterations): → 20m²n + 10m³ FLOPs
        5. Weight decay: W = (1 - lr*wd)*W → 2mn FLOPs
        6. Parameter update: W = W - lr*U → 2mn FLOPs

        Total: 11mn + 20m²n + 10m³ FLOPs

        Args:
            m: Smaller dimension (after transpose if needed)
            n: Larger dimension

        Returns:
            Total FLOPs for updating this weight matrix with Muon
        """
        # Newton-Schulz orthogonalization (5 iterations)
        # Each iteration:
        #   - U @ U.T: 2*m²*n FLOPs (m×n @ n×m = m×m)
        #   - A @ A: 2*m³ FLOPs (m×m @ m×m = m×m)
        #   - (bA + cA²) @ U: 2*m²*n FLOPs (m×m @ m×n = m×n)
        # Total per iteration: 4*m²*n + 2*m³ FLOPs
        ns_steps = 5
        flops_per_iteration = 4 * m * m * n + 2 * m * m * m
        newton_schulz_flops = ns_steps * flops_per_iteration

        # Overhead operations (all O(mn)):
        # - Momentum update: 2mn
        # - Nesterov lookahead: 2mn
        # - Normalization: 3mn
        # - Weight decay: 2mn
        # - Parameter update: 2mn
        # Total overhead: 11mn FLOPs
        overhead_flops = 11 * m * n

        return newton_schulz_flops + overhead_flops

    def __scion_weight_matrix_flops(self, m: int, n: int) -> float:
        """Calculate FLOPs for Scion optimizer on a single weight matrix.

        Based on the reference implementation and paper:
        1. Momentum update: buf = (1-momentum)*buf + momentum*grad → 3mn FLOPs
        2. Spectral LMO (Newton-Schulz, 5 iterations): → 20m²n + 10m³ FLOPs
        3. Constrained parameter update: p = (1-lr)*p - lr*update → 3mn FLOPs

        Total: 6mn + 20m²n + 10m³ FLOPs

        Args:
            m: Smaller dimension (after transpose if needed)
            n: Larger dimension

        Returns:
            Total FLOPs for updating this weight matrix with Scion
        """
        # Spectral LMO using Newton-Schulz iteration (5 iterations)
        # Each iteration:
        #   - G @ G.T: 2*m²*n FLOPs (m×n @ n×m = m×m)
        #   - A @ A: 2*m³ FLOPs (m×m @ m×m = m×m)
        #   - (I - c*A) @ G: 2*m²*n FLOPs (m×m @ m×n = m×n)
        # Total per iteration: 4*m²*n + 2*m³ FLOPs
        ns_steps = 5
        flops_per_iteration = 4 * m * m * n + 2 * m * m * m
        newton_schulz_flops = ns_steps * flops_per_iteration

        # Overhead operations (all O(mn)):
        # - Momentum update: 3mn (buf.mul_ + add_)
        # - Constrained parameter update: 3mn (p.mul_ + add_)
        # Total overhead: 6mn FLOPs
        overhead_flops = 6 * m * n

        return newton_schulz_flops + overhead_flops

    def activation_size_per_microbatch_per_layer(self) -> Size:
        activation_dict = self.__activation_numel_per_microbatch_per_layer()
        return Size(
            numel=sum(activation_dict.values()),
            bits_per_element=self.bits_per_parameter,
        )

    def activation_breakdown_per_microbatch_per_layer(self) -> dict[str, int]:
        """Returns dictionary of activation name to numel for a single microbatch per layer."""
        return self.__activation_numel_per_microbatch_per_layer()

    def vpp_penalty(self) -> float:
        """interleaved schedule requires storing activations for (1 + (p - 1)/pm)
        more layers."""
        if self.parallelism_cfg.vpp == 1:
            return 1.0

        return 1 + (self.parallelism_cfg.pp - 1) / (
            self.parallelism_cfg.pp * self.parallelism_cfg.vpp
        )

    def layers_per_pp_stage(self) -> int:
        return sum(self.__n_layers(mpmd_partitioned=True, moe=moe) for moe in [False, True])

    def get_vocab_padding_overhead(self) -> float:
        """Returns the memory overhead due to vocabulary padding as a ratio."""
        if self.vocab_sz == 0:
            return 0.0
        return (self.vocab_sz_padded - self.vocab_sz) / self.vocab_sz

    def get_vocab_padding_info(self) -> dict[str, int]:
        """Returns vocabulary padding information."""
        return {
            "original_vocab_size": self.vocab_sz,
            "padded_vocab_size": self.vocab_sz_padded,
            "padding_added": self.vocab_sz_padded - self.vocab_sz,
            "padding_size": self.vocab_padding_size,
            "tp_degree": self.parallelism_cfg.tp,
        }

    def _pad_vocab_size(self, vocab_size: int, padding_size: int) -> int:
        """Pad vocabulary size to be divisible by padding_size and tp degree.
        
        This ensures:
        1. Better hardware utilization by aligning to padding boundaries
        2. Even distribution across tensor parallel devices
        
        Args:
            vocab_size: Original vocabulary size
            padding_size: Padding boundary (e.g., 64 for hardware efficiency)
            
        Returns:
            Padded vocabulary size divisible by both padding_size and tp degree
        """
        # First pad to padding_size boundary for hardware efficiency
        padded_size = ((vocab_size + padding_size - 1) // padding_size) * padding_size
        
        # Then ensure divisible by tensor parallel degree for even distribution
        tp_degree = self.parallelism_cfg.tp
        if padded_size % tp_degree != 0:
            # Round up to next multiple that's divisible by both padding_size and tp_degree
            lcm = (padding_size * tp_degree) // self._gcd(padding_size, tp_degree)
            padded_size = ((padded_size + lcm - 1) // lcm) * lcm
        
        return padded_size

    def _gcd(self, a: int, b: int) -> int:
        """Calculate greatest common divisor using Euclidean algorithm."""
        while b:
            a, b = b, a % b
        return a

    def __get_n_total_params(self, spmd_partitioned: bool, mpmd_partitioned: bool) -> int:
        return _sum(
            # we'll give the number of parameters on the most heavily loaded pipeline stage
            # if PP=1 then the only pipeline stage must store both embedding and LM head.
            (1 if (mpmd_partitioned and self.parallelism_cfg.pp > 1) or self.tie_embeddings else 2)
            * self.__get_embedding_or_lm_head_size(spmd_partitioned=spmd_partitioned),
            # add in the transformer blocks
            sum(
                self.__n_layers(mpmd_partitioned=mpmd_partitioned, moe=moe)
                * self.__get_transformer_block_n_params(
                    spmd_partitioned=spmd_partitioned,
                    moe=moe,
                    active=False,
                    experts_per_token=None,
                )
                # TODO. clean this up
                for moe in ([False, True] if self.moe_cfg is not None else [False])
            ),
        )

    def __get_n_active_params(self, partitioned: bool) -> int:
        if not self.moe_cfg:
            return self.__get_n_total_params(
                spmd_partitioned=partitioned,
                mpmd_partitioned=partitioned,
            )

        # we'll give the number of parameters on the most heavily loaded pipeline stage
        # if PP=1 then the only pipeline stage must store both embedding and LM head.
        return _sum(
            # embedding/lmhead
            (1 if (partitioned and self.parallelism_cfg.pp > 1) or self.tie_embeddings else 2)
            * self.__get_embedding_or_lm_head_size(spmd_partitioned=partitioned),
            # transformer blocks
            sum(
                self.__n_layers(mpmd_partitioned=partitioned, moe=moe)
                * self.__get_transformer_block_n_params(
                    spmd_partitioned=partitioned,
                    moe=moe,
                    active=True,
                    experts_per_token=self.moe_cfg.experts_per_token,
                )
                for moe in [False, True]
            ),
        )

    def __get_embedding_or_lm_head_size(self, spmd_partitioned: bool) -> int:
        return self.embed_weight.size(partitioned=spmd_partitioned).numel()

    def __get_transformer_block_n_params(
        self,
        spmd_partitioned: bool,
        moe: bool,
        # TODO. would rather not expose these.
        active: bool,
        experts_per_token: int | None,
    ) -> int:
        assert active == (experts_per_token is not None)

        if moe:
            assert self.moe_cfg is not None
            assert self.mlp_up_exp_weight is not None
            assert self.mlp_down_exp_weight is not None
            mlp_params = (
                # if we're trying to compute active params, then we account for the
                # fact that we'll apply topk mlps per token.
                experts_per_token  # type: ignore[operator]
                * _sum(
                    self.mlp_up_exp_weight.numel(partitioned=spmd_partitioned)
                    // self.moe_cfg.n_experts,
                    self.mlp_down_exp_weight.numel(partitioned=spmd_partitioned)
                    // self.moe_cfg.n_experts,
                )
                if active
                else _sum(
                    self.mlp_up_exp_weight.numel(partitioned=spmd_partitioned),
                    self.mlp_down_exp_weight.numel(partitioned=spmd_partitioned),
                )
            )
            # Shared expert is a dense FFN run on ALL tokens every MoE layer, so it
            # contributes to BOTH total and active params (it is not routed/gated).
            if self.shared_mlp_up_weight is not None and self.shared_mlp_down_weight is not None:
                mlp_params += _sum(
                    self.shared_mlp_up_weight.numel(partitioned=spmd_partitioned),
                    self.shared_mlp_down_weight.numel(partitioned=spmd_partitioned),
                )
        else:
            mlp_params = _sum(
                self.mlp_up_weight.numel(partitioned=spmd_partitioned),
                self.mlp_down_weight.numel(partitioned=spmd_partitioned),
            )

        return _sum(
            self.pre_attn_norm_weight.numel(partitioned=spmd_partitioned),
            self.qkv_weight.numel(partitioned=spmd_partitioned),
            self.attn_out_weight.numel(partitioned=spmd_partitioned),
            self.pre_mlp_norm_weight.numel(partitioned=spmd_partitioned),
            mlp_params,
        )

    def __n_layers(self, mpmd_partitioned: bool, moe: bool) -> int:
        # TODO. we're making the assumption that MoE and non-MoE layers
        # can be evenly partitioned.
        n_layers = self.n_moe_layers if moe else self.n_nml_layers
        if mpmd_partitioned:
            n_layers = safe_divide(n_layers, self.parallelism_cfg.pp)

        return n_layers

    def __activation_numel_per_microbatch_per_layer(self) -> dict[str, int]:
        """
        See: Reducing Activation Recomputation in Large Transformer Models
        https://arxiv.org/pdf/2205.05198.pdf
        """
        sbh = self.sequence_len * self.microbatch_sz * self.hidden_sz
        sbq = self.sequence_len * self.microbatch_sz * self.n_q_heads * self.head_dim
        sbk = self.sequence_len * self.microbatch_sz * self.n_kv_heads * self.head_dim
        sbv = self.sequence_len * self.microbatch_sz * self.n_kv_heads * self.head_dim
        sbi = self.sequence_len * self.microbatch_sz * self.inter_sz

        is_moe = self.moe_cfg is not None
        moe_n_local_experts = (
            safe_divide(self.moe_cfg.n_experts, self.parallelism_cfg.expert_mesh.ep)
            if self.moe_cfg is not None and self.parallelism_cfg.expert_mesh is not None
            else 0
        )
        moe_nh = self.expert_capacity() * self.hidden_sz if self.moe_cfg is not None else 0
        moe_ni = (
            self.expert_capacity() * self.moe_cfg.expert_inter_sz if self.moe_cfg is not None else 0
        )
        # Shared-expert intermediate activation (all tokens, dense; TP-sharded).
        shared_expert_inter_sz = (
            self.moe_cfg.shared_expert_inter_sz if self.moe_cfg is not None else 0
        )
        has_shared_expert = shared_expert_inter_sz > 0
        sbi_shared = self.sequence_len * self.microbatch_sz * shared_expert_inter_sz

        if self.act_ckpting_type == ActivationCheckpointingType.FULL:
            return {"Block Input": self.__sp_partition_if_on(sbh)}
        elif self.act_ckpting_type in (
            ActivationCheckpointingType.NONE,
            ActivationCheckpointingType.SELECTIVE,  # basically obsolete w/ flash attention
            ActivationCheckpointingType.SUPER_SELECTIVE,
        ):
            return {
                # LAYERNORM 1
                "Pre Attn Norm": self.__deallocate_for_ssc(self.__sp_partition_if_on(sbh)),
                # QKV PROJ (col parallel linear)
                "Query": self.__tp_partition(sbq),
                "Key": self.__tp_partition(sbk),
                "Value": self.__tp_partition(sbv),
                # ROTARY EMBEDDINGS
                "Query Rotary": self.__tp_partition(sbq) if self.rotary_embed else 0,
                "Key Rotary": self.__tp_partition(sbk) if self.rotary_embed else 0,
                # SELF ATTENTION
                "Attention Output": self.__tp_partition(sbh),
                # DROPOUT
                "Post Attention Dropout Mask": self.__deallocate_for_ssc(
                    self.__sp_partition_if_on(int(0.5 * sbh)) if self.dropout else 0
                ),
                # RESIDUAL
                "Post Attention Residual": self.__sp_partition_if_on(sbh),
                # LAYERNORM 2
                "Pre MLP Norm": self.__sp_partition_if_on(sbh),
                # Permuted Input
                **(
                    {}
                    if not is_moe
                    else {
                        "Permuted Input": moe_n_local_experts * self.__expert_tp_partition(moe_nh)
                    }
                ),
                # MLP Input (EP)
                **(
                    {}
                    if not is_moe
                    else {
                        "MLP Input (EP)": moe_n_local_experts * self.__expert_tp_partition(moe_nh)
                    }
                ),
                # MLP UP/GATE (col parallel linear)
                "Up/Gate": (
                    self.__tp_partition(2 * sbi if self.glu else sbi)
                    if not is_moe
                    else moe_n_local_experts
                    * self.__expert_tp_partition(2 * moe_ni if self.glu else moe_ni)
                ),
                # SwiGLU
                "SiLU": self.__deallocate_for_ssc(
                    self.__tp_partition(sbi)
                    if not is_moe
                    else moe_n_local_experts * self.__expert_tp_partition(moe_ni)
                ),
                "Gate": self.__deallocate_for_ssc(
                    self.__tp_partition(sbi)
                    if not is_moe
                    else moe_n_local_experts * self.__expert_tp_partition(moe_ni)
                ),
                # Unpermuted Input
                **(
                    {}
                    if not is_moe
                    else {
                        "Unpermuted Output": moe_n_local_experts
                        * self.__expert_tp_partition(moe_nh)
                    }
                ),
                # Shared-expert dense MLP intermediate (all tokens, TP-sharded).
                # Mirrors the dense/routed MLP convention: up/gate (2x if GLU) plus
                # the SiLU and Gate saved activations (both deallocatable for SSC).
                **(
                    {
                        "Shared Expert Up/Gate": self.__tp_partition(
                            2 * sbi_shared if self.glu else sbi_shared
                        ),
                        "Shared Expert SiLU": self.__deallocate_for_ssc(
                            self.__tp_partition(sbi_shared)
                        ),
                        "Shared Expert Gate": self.__deallocate_for_ssc(
                            self.__tp_partition(sbi_shared)
                        ),
                    }
                    if has_shared_expert
                    else {}
                ),
                # DROPOUT
                "Post MLP Dropout Mask": self.__deallocate_for_ssc(
                    self.__sp_partition_if_on(int(0.5 * sbh)) if self.dropout else 0
                ),
                # RESIDUAL
                "Post MLP Residual": self.__sp_partition_if_on(sbh),
            }
        else:
            raise ValueError(f"unhandled checkpointing_type={self.act_ckpting_type}")

    def __tp_partition(self, unpartitoned_numel: int) -> int:
        x = unpartitoned_numel
        for parallelism_degree in [self.parallelism_cfg.cp, self.parallelism_cfg.tp]:
            x = safe_divide(x, parallelism_degree)

        return x

    def __expert_tp_partition(self, unpartitioned_numel: int) -> int:
        assert self.parallelism_cfg.expert_mesh is not None
        return safe_divide(unpartitioned_numel, self.parallelism_cfg.expert_mesh.tp)

    def __deallocate_for_ssc(self, numel: int) -> int:
        if self.act_ckpting_type == ActivationCheckpointingType.SUPER_SELECTIVE:
            return 0
        return numel

    def __sp_partition_if_on(self, unpartitioned_numel: int) -> int:
        parallelism_degrees = [self.parallelism_cfg.cp]
        if self.parallelism_cfg.sp_enabled:
            parallelism_degrees.append(self.parallelism_cfg.tp)

        x = unpartitioned_numel
        for parallelism_degree in parallelism_degrees:
            x = safe_divide(x, parallelism_degree)

        return x

    def expert_capacity(self) -> int:
        """Returns the number of tokens that can be processed by each expert.

        Dropless routing (moe_expert_capacity_factor=None in Megatron) processes
        every token with no drop, so the modeled per-expert load is the MEAN load:
        capacity_factor=1 gives exactly seq*mbs*dp*top_k / (expert_dp * n_experts),
        i.e. total routed token-rows spread evenly over experts. This is the honest
        mean-load approximation for dropless (Effect C). The measured busiest expert
        runs ~13% above mean (tokens_per_expert max ≈104k vs mean ≈92k on the 700M
        runs); modeling that worst-case would need an imbalance factor, but the
        golden harness shows the compute-bound residual is dominated by exposed comm
        (Effect B), not this ~13% capacity gap — so we keep the mean-load model and
        do not introduce an unused knob (no fudge factor; GUIDELINES §1).
        """
        if self.moe_cfg is None:
            raise RuntimeError
        assert self.parallelism_cfg.expert_mesh is not None

        # parallelisms (with token partitioning dimensions denoted by *)
        # nonexpert: [DP*, CP*, TP]
        # expert:    [EP, eDP*, eTP]

        n_tokens_unpartitioned = self.sequence_len * self.microbatch_sz * self.parallelism_cfg.dp
        n_expert_region_tokens_unpartitioned = int(
            n_tokens_unpartitioned * self.moe_cfg.experts_per_token * self.moe_cfg.capacity_factor
        )
        n_expert_region_tokens_partitioned = safe_divide(
            n_expert_region_tokens_unpartitioned,
            self.parallelism_cfg.expert_mesh.dp,
        )

        return safe_divide(n_expert_region_tokens_partitioned, self.moe_cfg.n_experts)

    def get_moe_workspace_memory(self) -> Size:
        """Calculate MoE-specific memory overhead based on Megatron-LM implementation.

        By examining the Megatron-LM source code, we identified the actual sources:
        1. GlobalMemoryBuffer for allgather (persists, scales with EP!)
        2. Expert intermediate activations (in-flight during forward/backward)
        3. All-to-all send/recv buffers
        4. Token permutation buffers
        5. Expert gradient buffers (NOT DP-partitioned during gradient accumulation!)
        6. Router overhead

        Returns:
            Size object representing total MoE workspace memory per device
        """
        if self.moe_cfg is None:
            return Size(numel=0, bits_per_element=self.bits_per_parameter)

        assert self.parallelism_cfg.expert_mesh is not None

        # Number of experts on this device after EP partitioning
        n_local_experts = safe_divide(
            self.moe_cfg.n_experts,
            self.parallelism_cfg.expert_mesh.ep,
        )

        # Tokens processed by each expert on this device
        expert_cap = self.expert_capacity()

        # Tokens per device (before routing to experts)
        n_tokens_per_device = safe_divide(
            self.sequence_len * self.microbatch_sz,
            self.parallelism_cfg.cp,
        )

        # ===== 1. GlobalMemoryBuffer for allgather (persists, never shrinks!) =====
        # Location: megatron/core/tensor_parallel/mappings.py, line 140
        # This buffer is allocated once and stays at max size for entire training
        # Size scales with EP because allgather group size is TP*EP
        allgather_buffer_numel = (
            self.sequence_len
            * self.microbatch_sz
            * self.parallelism_cfg.expert_mesh.ep  # ⚠️ SCALES WITH EP!
            * self.hidden_sz
        )

        # ===== 2. Expert intermediate activations (in-flight during forward/backward) =====
        # These are the actual expert MLP activations that exist during computation
        expert_activations_numel = (
            n_local_experts
            * expert_cap
            * self.moe_cfg.expert_inter_sz
            * 2  # GLU doubles the intermediate size
        )

        # ===== 3. All-to-all send/recv buffers (EP > 1 only) =====
        # Location: megatron/core/transformer/moe/token_dispatcher.py
        # Tokens are routed between expert groups via all-to-all collectives
        a2a_buffer_numel = 0
        if self.parallelism_cfg.expert_mesh.ep > 1:
            # Need both send and recv buffers, each same size as allgather
            a2a_buffer_numel = 2 * allgather_buffer_numel

        # ===== 4. Token permutation buffers =====
        # Tokens need to be permuted before and unpermuted after expert processing
        permute_buffer_numel = (
            2  # permute + unpermute
            * n_tokens_per_device
            * self.hidden_sz
        )

        # ===== 5. Expert gradient buffers (NOT DP-partitioned!) =====
        # ⚠️ THIS IS THE BIG ONE! Expert gradients are accumulated WITHOUT
        # DP partitioning until the final microbatch, then reduce-scattered.
        # This is similar to grad_buffer but for expert parameters only.
        expert_grad_buffer_numel = (
            n_local_experts
            * (
                self.hidden_sz * self.moe_cfg.expert_inter_sz * 2  # up proj (GLU)
                + self.moe_cfg.expert_inter_sz * self.hidden_sz  # down proj
            )
        )

        # ===== 6. Router overhead =====
        # Router logits, indices, and scores
        router_overhead_numel = (
            n_tokens_per_device * self.moe_cfg.n_experts  # logits (FP32, so 2x FP16)
            + n_tokens_per_device * self.moe_cfg.experts_per_token * 2  # indices + scores
        )

        # Total MoE workspace per layer
        per_layer_workspace_numel = (
            expert_activations_numel
            + permute_buffer_numel
            + expert_grad_buffer_numel
            + router_overhead_numel
        )

        # Determine layer multiplier based on activation checkpointing strategy
        # - SELECTIVE: Layers overlap during recomputation, all buffers can exist simultaneously
        # - NONE/FULL: Layers process sequentially, only one layer's buffers at a time
        n_moe_layers = int(self.n_layers * self.moe_cfg.moe_frequency)

        if self.act_ckpting_type == ActivationCheckpointingType.SELECTIVE:
            # With selective checkpointing, all layers' buffers can exist simultaneously
            layer_multiplier = n_moe_layers
        else:
            # With NONE or FULL, layers process sequentially - only one layer's buffers at a time
            layer_multiplier = 1

        # Global buffers (allgather, a2a) are shared across layers
        global_buffers_numel = (
            allgather_buffer_numel
            + a2a_buffer_numel
        )

        # Total MoE workspace
        total_workspace_numel = (
            global_buffers_numel
            + per_layer_workspace_numel * layer_multiplier
        )

        return Size(
            numel=total_workspace_numel,
            bits_per_element=self.bits_per_parameter,
        )
