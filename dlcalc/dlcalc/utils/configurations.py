import dataclasses
from enum import Enum


class ActivationCheckpointingType(Enum):
    NONE = 1
    SELECTIVE = 2
    SUPER_SELECTIVE = 3
    FULL = 4

    @staticmethod
    def from_str(str_key: str) -> "ActivationCheckpointingType":
        return {
            "none": ActivationCheckpointingType.NONE,
            "selective": ActivationCheckpointingType.SELECTIVE,
            "super-selective": ActivationCheckpointingType.SUPER_SELECTIVE,
            "full": ActivationCheckpointingType.FULL,
        }[str_key]


@dataclasses.dataclass
class CrossDCConfig:
    """Configuration for cross-DC training.

    Attributes:
        n_dcs: Number of data centers in the ring topology
        interconnect_bandwidth_gbps: Total cross-DC bandwidth in Gbps
        interconnect_latency_s: Latency between adjacent DCs in seconds
    """

    n_dcs: int
    interconnect_bandwidth_gbps: float
    interconnect_latency_s: float

    def interconnect_bandwidth_bytes_per_sec(self) -> float:
        """Convert bandwidth from Gbps to bytes per second.

        Returns:
            Bandwidth in bytes per second
        """
        return self.interconnect_bandwidth_gbps * 1e9 / 8  # Gbps to bytes/sec
