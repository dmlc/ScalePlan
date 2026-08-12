"""Unit tests for data.py classes.

This module tests the Size and TensorRepr classes which are fundamental
data structures for representing memory sizes and distributed tensor shapes
in the dlcalc package.

Test Coverage:
- Size class: Basic operations, arithmetic, edge cases
- TensorRepr class: Partitioning, shape calculations, size methods
- Integration: Combined usage of Size and TensorRepr for realistic scenarios
"""

import pytest

from dlcalc.utils.data import Size, TensorRepr


class TestSize:
    """Test cases for the Size class."""

    def test_basic_initialization(self):
        """Test basic Size initialization."""
        size = Size(numel=1000, bits_per_element=16)
        assert size.numel() == 1000
        assert size._bits_per_element == 16

    def test_bits_calculation(self):
        """Test bits calculation."""
        size = Size(numel=1000, bits_per_element=16)
        assert size.bits() == 16000

    def test_bytes_calculation(self):
        """Test bytes calculation."""
        # 16 bits = 2 bytes per element
        size = Size(numel=1000, bits_per_element=16)
        assert size.bytes() == 2000

        # 32 bits = 4 bytes per element
        size = Size(numel=500, bits_per_element=32)
        assert size.bytes() == 2000

        # 8 bits = 1 byte per element
        size = Size(numel=100, bits_per_element=8)
        assert size.bytes() == 100

    def test_addition_same_bits(self):
        """Test addition of Size objects with same bits per element."""
        size1 = Size(numel=1000, bits_per_element=16)
        size2 = Size(numel=500, bits_per_element=16)
        result = size1 + size2

        assert result.numel() == 1500
        assert result._bits_per_element == 16
        assert result.bytes() == 3000

    def test_addition_different_bits_raises_error(self):
        """Test that adding sizes with different bits per element raises error."""
        size1 = Size(numel=1000, bits_per_element=16)
        size2 = Size(numel=500, bits_per_element=32)

        with pytest.raises(ValueError, match="different bits per element"):
            _ = size1 + size2

    def test_multiplication(self):
        """Test multiplication of Size by integer."""
        size = Size(numel=100, bits_per_element=16)

        # Left multiplication
        result = size * 3
        assert result.numel() == 300
        assert result._bits_per_element == 16

        # Right multiplication (rmul)
        result = 5 * size
        assert result.numel() == 500
        assert result._bits_per_element == 16

    def test_floor_division(self):
        """Test floor division of Size by integer."""
        size = Size(numel=1000, bits_per_element=16)
        result = size // 3

        assert result.numel() == 333  # floor division
        assert result._bits_per_element == 16

    def test_repr(self):
        """Test string representation of Size."""
        size = Size(numel=int(1e9), bits_per_element=16)
        repr_str = repr(size)

        assert "1.000 B" in repr_str  # 1B elements
        assert "1.863 GiB" in repr_str  # 2GB in GiB

    def test_edge_cases(self):
        """Test edge cases for Size."""
        # Zero elements
        size = Size(numel=0, bits_per_element=16)
        assert size.bits() == 0
        assert size.bytes() == 0

        # Large numbers
        size = Size(numel=int(1e12), bits_per_element=32)
        assert size.bits() == int(32e12)
        assert size.bytes() == int(4e12)

        # Chain operations
        size = Size(numel=100, bits_per_element=16)
        result = size * 2 + size * 3
        assert result.numel() == 500

        # Division by 1
        result = size // 1
        assert result.numel() == size.numel()


class TestTensorRepr:
    """Test cases for the TensorRepr class."""

    def test_basic_initialization(self):
        """Test basic TensorRepr initialization."""
        tensor = TensorRepr(unpartitioned_shape=(1024, 768), partition_spec={}, bits_per_elt=16)

        assert tensor._unpartitioned_shape == (1024, 768)
        assert tensor._partition_spec == {}
        assert tensor._bits_per_elt == 16

    def test_unpartitioned_shape_and_numel(self):
        """Test shape and numel for unpartitioned tensor."""
        tensor = TensorRepr(unpartitioned_shape=(100, 200, 300), partition_spec={}, bits_per_elt=32)

        assert tensor.shape(partitioned=False) == (100, 200, 300)
        assert tensor.numel(partitioned=False) == 100 * 200 * 300

    def test_partitioned_shape_single_axis(self):
        """Test shape for tensor partitioned along single axis."""
        tensor = TensorRepr(
            unpartitioned_shape=(1024, 768),
            partition_spec={0: 4},  # Partition first dimension by 4
            bits_per_elt=16,
        )

        assert tensor.shape(partitioned=False) == (1024, 768)
        assert tensor.shape(partitioned=True) == (256, 768)

    def test_partitioned_shape_multiple_axes(self):
        """Test shape for tensor partitioned along multiple axes."""
        tensor = TensorRepr(
            unpartitioned_shape=(1024, 768, 512),
            partition_spec={0: 4, 2: 2},  # Partition dim 0 by 4, dim 2 by 2
            bits_per_elt=16,
        )

        assert tensor.shape(partitioned=False) == (1024, 768, 512)
        assert tensor.shape(partitioned=True) == (256, 768, 256)

    def test_partitioned_numel(self):
        """Test numel calculation for partitioned tensor."""
        tensor = TensorRepr(
            unpartitioned_shape=(1000, 2000),
            partition_spec={0: 10, 1: 4},  # Total partitioning degree = 40
            bits_per_elt=32,
        )

        unpartitioned_numel = 1000 * 2000
        assert tensor.numel(partitioned=False) == unpartitioned_numel
        assert tensor.numel(partitioned=True) == unpartitioned_numel // 40

    def test_size_method(self):
        """Test the size method returns correct Size object."""
        tensor = TensorRepr(unpartitioned_shape=(100, 200), partition_spec={1: 2}, bits_per_elt=16)

        # Unpartitioned size
        size = tensor.size(partitioned=False)
        assert isinstance(size, Size)
        assert size.numel() == 20000
        assert size._bits_per_element == 16

        # Partitioned size
        size = tensor.size(partitioned=True)
        assert isinstance(size, Size)
        assert size.numel() == 10000
        assert size._bits_per_element == 16

    def test_enforce_evenly_partitionable_true(self):
        """Test that unevenly partitionable tensors raise error when enforced."""
        with pytest.raises(RuntimeError, match="not divisible"):
            TensorRepr(
                unpartitioned_shape=(100, 200),
                partition_spec={0: 3},  # 100 not divisible by 3
                bits_per_elt=16,
                enforce_evenly_partitionable=True,
            )

    def test_enforce_evenly_partitionable_false(self):
        """Test that unevenly partitionable tensors can be created when not enforced."""
        # Should not raise error during initialization
        tensor = TensorRepr(
            unpartitioned_shape=(100, 200),
            partition_spec={0: 3},  # 100 not divisible by 3
            bits_per_elt=16,
            enforce_evenly_partitionable=False,
        )

        # Note: safe_divide will still raise an error when actually calculating shape
        # This is the current behavior - the flag only affects initialization
        with pytest.raises(ValueError, match="not divisible"):
            _ = tensor.shape(partitioned=True)

    def test_empty_partition_spec(self):
        """Test tensor with empty partition spec behaves as unpartitioned."""
        tensor = TensorRepr(unpartitioned_shape=(512, 1024), partition_spec={}, bits_per_elt=32)

        assert tensor.shape(partitioned=True) == tensor.shape(partitioned=False)
        assert tensor.numel(partitioned=True) == tensor.numel(partitioned=False)

    def test_complex_partitioning(self):
        """Test complex partitioning scenario."""
        # 4D tensor partitioned along 3 dimensions
        tensor = TensorRepr(
            unpartitioned_shape=(64, 128, 256, 512),
            partition_spec={0: 2, 1: 4, 3: 8},  # Total degree = 64
            bits_per_elt=16,
        )

        assert tensor.shape(partitioned=True) == (32, 32, 256, 64)

        unpartitioned_numel = 64 * 128 * 256 * 512
        partitioned_numel = 32 * 32 * 256 * 64

        assert tensor.numel(partitioned=False) == unpartitioned_numel
        assert tensor.numel(partitioned=True) == partitioned_numel

    def test_single_element_tensor(self):
        """Test edge case of single element tensor."""
        tensor = TensorRepr(unpartitioned_shape=(1,), partition_spec={}, bits_per_elt=32)

        assert tensor.shape(partitioned=False) == (1,)
        assert tensor.numel(partitioned=False) == 1
        assert tensor.size(partitioned=False).bytes() == 4

    def test_high_dimensional_tensor(self):
        """Test tensor with many dimensions."""
        shape = (2, 3, 4, 5, 6, 7, 8, 9)
        tensor = TensorRepr(
            unpartitioned_shape=shape, partition_spec={0: 2, 3: 5, 7: 3}, bits_per_elt=8
        )

        expected_partitioned = (1, 3, 4, 1, 6, 7, 8, 3)
        assert tensor.shape(partitioned=True) == expected_partitioned

    def test_partition_spec_with_large_degrees(self):
        """Test partitioning with large partition degrees."""
        tensor = TensorRepr(
            unpartitioned_shape=(10000, 20000), partition_spec={0: 100, 1: 200}, bits_per_elt=16
        )

        assert tensor.shape(partitioned=True) == (100, 100)
        assert tensor.numel(partitioned=True) == 10000

    def test_bits_per_element_variations(self):
        """Test different bits per element values."""
        shape = (100, 100)

        # 8-bit tensor
        tensor8 = TensorRepr(shape, {}, bits_per_elt=8)
        assert tensor8.size(partitioned=False).bytes() == 10000

        # 16-bit tensor
        tensor16 = TensorRepr(shape, {}, bits_per_elt=16)
        assert tensor16.size(partitioned=False).bytes() == 20000

        # 32-bit tensor
        tensor32 = TensorRepr(shape, {}, bits_per_elt=32)
        assert tensor32.size(partitioned=False).bytes() == 40000

        # 64-bit tensor
        tensor64 = TensorRepr(shape, {}, bits_per_elt=64)
        assert tensor64.size(partitioned=False).bytes() == 80000


class TestIntegration:
    """Integration tests for Size and TensorRepr working together."""

    def test_tensor_size_operations(self):
        """Test operations on sizes from TensorRepr."""
        tensor1 = TensorRepr(unpartitioned_shape=(100, 100), partition_spec={}, bits_per_elt=16)
        tensor2 = TensorRepr(unpartitioned_shape=(50, 50), partition_spec={}, bits_per_elt=16)

        size1 = tensor1.size(partitioned=False)
        size2 = tensor2.size(partitioned=False)

        # Add sizes
        total_size = size1 + size2
        assert total_size.numel() == 12500  # 10000 + 2500
        assert total_size.bytes() == 25000  # 16 bits = 2 bytes per element

    def test_partitioned_tensor_memory_calculation(self):
        """Test realistic memory calculation for partitioned tensor."""
        # Simulating a large weight matrix partitioned for tensor parallelism
        tensor = TensorRepr(
            unpartitioned_shape=(4096, 8192),  # 32M parameters
            partition_spec={1: 8},  # TP=8
            bits_per_elt=16,  # fp16
        )

        # Unpartitioned memory
        full_size = tensor.size(partitioned=False)
        assert full_size.numel() == 4096 * 8192
        assert full_size.bytes() == 4096 * 8192 * 2  # 64MB

        # Partitioned memory (per device)
        shard_size = tensor.size(partitioned=True)
        assert shard_size.numel() == 4096 * 1024  # 4M parameters per device
        assert shard_size.bytes() == 4096 * 1024 * 2  # 8MB per device

    def test_multiple_tensor_total_memory(self):
        """Test calculating total memory for multiple tensors."""
        # Simulate transformer layer weights
        qkv_weight = TensorRepr(
            unpartitioned_shape=(768, 2304),  # 768 * 3 for Q, K, V
            partition_spec={1: 4},
            bits_per_elt=16,
        )

        mlp_weight = TensorRepr(
            unpartitioned_shape=(768, 3072), partition_spec={1: 4}, bits_per_elt=16
        )

        # Calculate total partitioned memory
        qkv_size = qkv_weight.size(partitioned=True)
        mlp_size = mlp_weight.size(partitioned=True)
        total_size = qkv_size + mlp_size

        expected_qkv_numel = 768 * 576  # 2304/4
        expected_mlp_numel = 768 * 768  # 3072/4

        assert qkv_size.numel() == expected_qkv_numel
        assert mlp_size.numel() == expected_mlp_numel
        assert total_size.numel() == expected_qkv_numel + expected_mlp_numel
