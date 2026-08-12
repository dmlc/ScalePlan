import json
from dataclasses import dataclass
from enum import Enum

EFA_LATENCY_S = 30e-6


class DType(Enum):
    FP4 = "fp4"
    FP8 = "fp8"
    FP8_E4M3 = "fp8_e4m3"
    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"
    BF16 = "bf16"
    INT8 = "int8"

    def size_bytes(self) -> float:
        """Return the size of this dtype in bytes."""
        size_map = {
            DType.FP4: 0.5,  # 4 bits
            DType.FP8: 1,
            DType.FP8_E4M3: 1,
            DType.INT8: 1,
            DType.FP16: 2,
            DType.BF16: 2,
            DType.FP32: 4,
            DType.TF32: 4,
            DType.FP64: 8,
            DType.TF64: 8,
        }
        return size_map.get(self, 2)  # Default to 2 bytes
    TF32 = "tf32"
    TF64 = "tf64"


@dataclass
class DeviceSpec:
    dtype_to_peak_flops: dict[DType, float]
    mem_bandwidth_bytes_per_sec: int
    mem_capacity_bytes: int

    def peak_flops(self, dtype: DType) -> float:
        """Get the peak FLOPS for a specific data type.

        Args:
            dtype: The data type to get peak FLOPS for.

        Returns:
            The peak FLOPS for the specified data type.

        Raises:
            KeyError: If the dtype is not supported by this device.
        """
        return self.dtype_to_peak_flops[dtype]


@dataclass
class LinkSpec:
    unidirectional_bw_bytes_per_sec: int
    latency_sec: float

    def __repr__(self) -> str:
        return json.dumps(
            {
                "unidirectional bw": f"{self.unidirectional_bw_bytes_per_sec * 1e-9:.2f} GBps",
                "latency": f"{self.latency_sec * 1e6} us",
            }
        )


@dataclass
class MachineSpec:
    name: str
    n_devices: int
    device_spec: DeviceSpec
    intra_node_connect: LinkSpec
    inter_node_connect: LinkSpec

    @staticmethod
    def from_str(str: str) -> "MachineSpec":
        return {
            # A100 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
            # https://aws.amazon.com/ec2/instance-types/p4/
            "p4d.24xlarge": MachineSpec(
                name="p4d.24xlarge",
                n_devices=8,
                # A100-40GiB SXM
                device_spec=DeviceSpec(
                    dtype_to_peak_flops={
                        DType.FP32: int(19.5e12),
                        DType.FP16: int(312e12),
                        DType.FP64: int(9.7e12),
                        DType.BF16: int(312e12),
                        DType.INT8: int(624e12),
                        DType.TF32: int(156e12),
                        DType.TF64: int(19.5e12)
                    },
                    # 40GiB HBM2
                    mem_bandwidth_bytes_per_sec=int(1555e9),
                    mem_capacity_bytes=40 * (1024**3),
                ),
                # NVLink 3
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(300e9),
                    latency_sec=3e-6,
                ),
                # EFA v1 - 4 x 100 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(50e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # https://aws.amazon.com/ec2/instance-types/p4/
            "p4de.24xlarge": MachineSpec(
                name="p4de.24xlarge",
                n_devices=8,
                # A100-80GiB SXM
                device_spec=DeviceSpec(
                    dtype_to_peak_flops={
                        DType.FP32: int(19.5e12),
                        DType.FP16: int(312e12),
                        DType.FP64: int(9.7e12),
                        DType.BF16: int(312e12),
                        DType.INT8: int(624e12),
                        DType.TF32: int(156e12),
                        DType.TF64: int(19.5e12)
                    },
                    # 80GiB HBM2e
                    mem_bandwidth_bytes_per_sec=int(2039e9),
                    mem_capacity_bytes=80 * (1024**3),
                ),
                # NVLink 3
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(300e9),
                    latency_sec=3e-6,
                ),
                # EFA v1 - 4 x 100 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(50e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # H100 datasheet: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
            # https://aws.amazon.com/ec2/instance-types/p5/
            "p5.48xlarge": MachineSpec(
                name="p5.48xlarge",
                n_devices=8,
                # H100 SXM
                device_spec=DeviceSpec(
                    dtype_to_peak_flops={
                        DType.FP8: int(1978.9e12),
                        DType.FP8_E4M3: int(1978.9e12),
                        DType.FP16: int(989.4e12),  # (ignore sparsity numbers)
                        DType.FP32: int(67e12),
                        DType.FP64: int(34e12),
                        DType.BF16: int(989.4e12),  # (ignore sparsity numbers)
                        DType.INT8: int(1978.9e12)
                    },
                    # 80 GiB HBM3
                    mem_bandwidth_bytes_per_sec=int(3350e9),
                    mem_capacity_bytes=80 * (1024**3),
                ),
                # NVLink 4
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(450e9),
                    latency_sec=3e-6,
                ),
                # EFA v2 - 32 x 100 Gbps
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(400e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            ),
            # The Technical Specifications for NVIDIA GB200 GPU: https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf
            "p6-b200.48xlarge": MachineSpec(
                name="p6-b200.48xlarge",
                n_devices=8,
                device_spec=DeviceSpec(
                    dtype_to_peak_flops={
                        DType.FP4: int(20e15),
                        DType.FP8: int(4.5e15),
                        DType.FP8_E4M3: int(4.5e15),
                        DType.FP16: int(2.25e15),
                        DType.FP32: int(80e12),
                        DType.FP64: int(40e12),
                        DType.TF32: int(1.25e15),
                        DType.BF16: int(2.25e15),
                        DType.INT8: int(10e15)
                    },
                    # 180GiB HBM3e
                    mem_bandwidth_bytes_per_sec = int(8e12),
                    mem_capacity_bytes = 186 * (1024**3),
                ),
                # NVLink inside the NVL72 baseboard (aggregate headline bandwidth).
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec = int(0.9e12),  # 1.8 TB/s for NVLINK
                    latency_sec = 3e-6,  # placeholder; update if you have a measured value
                ),
                # https://aws.amazon.com/ec2/instance-types/p6/
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec = int(400e9),   # 400 GB/s (placeholder, e.g., EFA/IB link scale)
                    latency_sec = EFA_LATENCY_S,
                ),
            ),
            # https://aws.amazon.com/ec2/instance-types/trn1/
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trn1-arch.html
            "trn1n.32xlarge": MachineSpec(
                name="trn1n.32xlarge",
                n_devices=32,  # technically 16, but treating neuron cores as devices.
                # NeuronCore v2
                device_spec=DeviceSpec(
                    dtype_to_peak_flops={
                        DType.FP16: int(95e12),
                    },
                    # 16 GiB HBM2e
                    mem_bandwidth_bytes_per_sec=int(410e9),
                    mem_capacity_bytes=16 * (1024**3),
                ),
                # NeuronLink v2
                intra_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(384e9),
                    latency_sec=float("inf"),  # TODO. not sure, figure out empirically
                ),
                # EFA v2
                inter_node_connect=LinkSpec(
                    unidirectional_bw_bytes_per_sec=int(200e9),
                    latency_sec=EFA_LATENCY_S,
                ),
            )
        }[str]

    def total_flops(self, dtype: DType) -> float:
        """Get the total peak FLOPS across all devices in the machine.

        Args:
            dtype: The data type to get peak FLOPS for.

        Returns:
            The total peak FLOPS across all devices for the specified data type.
        """
        return self.device_spec.peak_flops(dtype) * self.n_devices
