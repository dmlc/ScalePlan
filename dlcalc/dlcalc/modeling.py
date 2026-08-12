import abc
from abc import ABC
from datetime import timedelta

from dlcalc.utils.hardware import DType, MachineSpec
from dlcalc.utils.math import safe_divide


class Op(ABC):
    @abc.abstractmethod
    def compute_runtime_s(self, machine_spec: MachineSpec) -> timedelta:
        raise NotImplementedError

    @abc.abstractmethod
    def get_n_params(self, partitioned: bool) -> int:
        raise NotImplementedError


class Transformer(Op):
    pass


class TransformerBlock(Op):
    pass


class Attention(Op):
    pass


class MLP(Op):
    pass


class ExpertMLP(Op):
    def __init__(
        self,
        n_tokens: int,
        n_experts: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        glu: bool,
        ep: int,
        tp: int,
        dtype: DType = DType.FP16,
        capacity_factor: float = 1.25,
        top_k: int = 2,
    ) -> None:
        # Calculate expert capacity: capacity_factor * n_tokens * top_k / n_experts
        # This ensures load balancing across experts with some buffer for imbalanced routing
        self.expert_capacity = max(1, int(capacity_factor * n_tokens * top_k / n_experts))
        
        # Store configuration for parameter calculations
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.glu = glu
        self.ep = ep
        self.tp = tp

        # Router: maps tokens to experts
        self.router = GEMM(
            m=n_tokens,
            k=hidden_dim,
            n=n_experts,
            m_partition_degree=1,  # No partitioning for router typically
            k_partition_degree=tp,  # Partition along hidden dimension with tensor parallelism
            n_partition_degree=1,  # Expert selection not partitioned
            dtype=dtype,
        )

        # MLP up projection for all experts (batched)
        self.mlp_up = BatchedGEMM(
            b=n_experts // ep,  # Number of experts per device after expert parallelism
            m=self.expert_capacity,
            k=hidden_dim // tp,  # Hidden dim partitioned by tensor parallelism
            # GLU doubles the intermediate dimension
            n=(mlp_hidden_dim * 2) if glu else mlp_hidden_dim,
        )
        
        # MLP down projection for all experts (batched)
        self.mlp_down = BatchedGEMM(
            b=n_experts // ep,  # Number of experts per device after expert parallelism
            m=self.expert_capacity,
            k=(mlp_hidden_dim * 2) if glu else mlp_hidden_dim,
            n=hidden_dim // tp,  # Output partitioned by tensor parallelism
        )

    def compute_runtime_s(self, machine_spec: MachineSpec) -> timedelta:
        """Compute total runtime for expert MLP operations."""
        router_time = self.router.compute_runtime_s(machine_spec)
        up_time = self.mlp_up.compute_runtime_s(machine_spec)
        down_time = self.mlp_down.compute_runtime_s(machine_spec)
        
        # All operations are sequential
        total_time = router_time + up_time + down_time
        return total_time

    def get_n_params(self, partitioned: bool) -> int:
        """Calculate total parameters for expert MLP."""
        router_params = self.router.get_n_params(partitioned)
        up_params = self.mlp_up.get_n_params(partitioned)
        down_params = self.mlp_down.get_n_params(partitioned)
        
        return router_params + up_params + down_params


class Norm(Op):
    def __init__(self, n_tokens: int, hidden_dim: int, n_tokens_partition_degree: int) -> None:
        self.__n_tokens = n_tokens
        self.__hidden_dim = hidden_dim
        self.__n_tokens_partition_degree = n_tokens_partition_degree

    def compute_runtime_s(self, machine_spec: MachineSpec) -> timedelta:
        """Compute runtime for layer normalization.
        
        Layer norm involves:
        1. Computing mean and variance (2 passes over data)
        2. Normalization and scaling (1 pass)
        Total: ~3 * n_tokens * hidden_dim element-wise operations
        """
        # Partition tokens across devices
        n_tokens_partitioned = safe_divide(self.__n_tokens, self.__n_tokens_partition_degree)
        
        # Element-wise operations are much faster than matrix multiplications
        # Approximating as 3 ops per element (mean, variance, normalize+scale)
        total_operations = 3 * n_tokens_partitioned * self.__hidden_dim
        
        # Use memory bandwidth as bottleneck rather than compute for elementwise ops
        # Assume each element is read/written twice (input + output)
        bytes_per_element = 2  # FP16
        total_bytes = 2 * n_tokens_partitioned * self.__hidden_dim * bytes_per_element
        
        # Use memory bandwidth as limiting factor (more realistic for norm ops)
        memory_bandwidth_bytes_per_sec = machine_spec.device_spec.mem_bandwidth_bytes_per_sec
        runtime_seconds = total_bytes / memory_bandwidth_bytes_per_sec
        
        return timedelta(seconds=runtime_seconds)

    def get_n_params(self, partitioned: bool) -> int:
        """Layer normalization has 2 parameters per hidden dimension: weight and bias."""
        if partitioned:
            # If partitioned, only count parameters for the local partition
            # Typically normalization parameters are replicated, not partitioned
            return 2 * self.__hidden_dim
        else:
            # Total parameters: weight vector + bias vector
            return 2 * self.__hidden_dim


class SDPA(Op):
    pass


class GEMM(Op):
    def __init__(
        self,
        m: int,
        k: int,
        n: int,
        m_partition_degree: int,
        k_partition_degree: int,
        n_partition_degree: int,
        dtype: DType,
    ) -> None:
        self.__m = m
        self.__k = k
        self.__n = n
        self.__m_partition_degree = m_partition_degree
        self.__k_partition_degree = k_partition_degree
        self.__n_partition_degree = n_partition_degree
        self.__dtype = dtype

    def compute_runtime_s(self, machine_spec: MachineSpec) -> timedelta:
        m_partitioned = safe_divide(self.__m, self.__m_partition_degree)
        n_partitioned = safe_divide(self.__n, self.__n_partition_degree)
        k_partitioned = safe_divide(self.__k, self.__k_partition_degree)

        n_flops_partitioned = m_partitioned * n_partitioned * k_partitioned

        n_secs = n_flops_partitioned / machine_spec.device_spec.peak_flops(self.__dtype)

        return timedelta(seconds=n_secs)

    def get_n_params(self, partitioned: bool) -> int:
        if partitioned:
            k_partitioned = safe_divide(self.__k, self.__k_partition_degree)
            n_partitioned = safe_divide(self.__n, self.__n_partition_degree)
            return k_partitioned * n_partitioned
        else:
            return self.__k * self.__n


class BatchedGEMM(Op):
    def __init__(self, b: int, m: int, k: int, n: int, dtype: DType = DType.FP16) -> None:
        self.b = b  # batch size (number of matrices)
        self.m = m  # rows of each matrix A
        self.k = k  # cols of A, rows of B
        self.n = n  # cols of each matrix B
        self.dtype = dtype

    def compute_runtime_s(self, machine_spec: MachineSpec) -> timedelta:
        """Compute runtime for batched GEMM: b matrices of size (m,k) @ (k,n)."""
        # Total FLOPs: batch_size * (2 * m * k * n) 
        # The factor of 2 comes from multiply-add operations
        total_flops = self.b * 2 * self.m * self.k * self.n
        
        # Compute time based on peak FLOPS of the device
        peak_flops = machine_spec.device_spec.peak_flops(self.dtype)
        runtime_seconds = total_flops / peak_flops
        
        return timedelta(seconds=runtime_seconds)

    def get_n_params(self, partitioned: bool) -> int:
        """Calculate parameters for batched GEMM (weight matrices)."""
        # Each batch element has a weight matrix of size (k, n)
        # Total parameters = b * k * n
        return self.b * self.k * self.n
