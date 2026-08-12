from .math import ceil_divide, product, safe_divide


class Size:
    """representation of how much space something takes, which is derived from
    what the datatype is, and how many elements of that datatype there are.
    """

    def __init__(self, numel: int, bits_per_element: int) -> None:
        self._numel = numel
        self._bits_per_element = bits_per_element

    def numel(self) -> int:
        return self._numel

    def bits(self) -> int:
        return self._numel * self._bits_per_element

    def bytes(self) -> int:
        return self._numel * safe_divide(self._bits_per_element, 8)

    def __add__(self, other: "Size") -> "Size":
        if self._bits_per_element != other._bits_per_element:
            raise ValueError(
                f"attempted addition between two sizes with different bits per element "
                f"{self._bits_per_element} vs {other._bits_per_element}"
            )

        return Size(numel=self._numel + other._numel, bits_per_element=self._bits_per_element)

    def __mul__(self, multiplicand: int) -> "Size":
        return Size(self._numel * multiplicand, self._bits_per_element)

    def __floordiv__(self, divisor: int) -> "Size":
        return Size(self._numel // divisor, self._bits_per_element)

    def __rmul__(self, multiplicand: int) -> "Size":
        return Size(self._numel * multiplicand, self._bits_per_element)

    def __repr__(self) -> str:
        return (
            f"numel: {self._numel * 1e-9:.3f} B, "
            f"bytes: {self._numel * safe_divide(self._bits_per_element, 8) / (1024**3):.3f} GiB"
        )


class TensorRepr:
    """Representation of a (possibly distributed) Tensor,
    has an unpartitioned shape, along with partitioning axes and degree.
    """

    def __init__(
        self,
        unpartitioned_shape: tuple[int, ...],
        partition_spec: dict[int, int],  # axis -> degree
        bits_per_elt: int,
        enforce_evenly_partitionable: bool = True,
    ) -> None:
        if enforce_evenly_partitionable:
            for partition_dim, partition_degree in partition_spec.items():
                if unpartitioned_shape[partition_dim] % partition_degree != 0:
                    raise RuntimeError(
                        f"dim {partition_dim} of {unpartitioned_shape} not divisible by {partition_degree}"
                    )

        self._unpartitioned_shape = unpartitioned_shape
        self._partition_spec = partition_spec
        self._bits_per_elt = bits_per_elt

    def shape(self, partitioned: bool) -> tuple[int, ...]:
        return self.__get_shape(partitioned=partitioned)

    def numel(self, partitioned: bool) -> int:
        return self.__get_numel(partitioned=partitioned)

    def size(self, partitioned: bool) -> Size:
        return Size(
            numel=self.__get_numel(partitioned=partitioned),
            bits_per_element=self._bits_per_elt,
        )

    def __get_shape(self, partitioned: bool) -> tuple[int, ...]:
        if partitioned:
            shape = list(self._unpartitioned_shape)
            for partition_dim, partition_degree in self._partition_spec.items():
                shape[partition_dim] = safe_divide(shape[partition_dim], partition_degree)
            return tuple(shape)
        else:
            return self._unpartitioned_shape

    def __get_numel(self, partitioned: bool) -> int:
        unpartitioned_numel = product(*self._unpartitioned_shape)
        if partitioned and self._partition_spec:
            total_partitioning_degree = product(*self._partition_spec.values())
            return ceil_divide(unpartitioned_numel, total_partitioning_degree)
        else:
            return unpartitioned_numel
