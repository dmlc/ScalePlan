"""Unit tests for math.py utility functions.

This module tests mathematical utility functions used throughout
the dlcalc package for safe division, ceiling division, and products.

Test Coverage:
- safe_divide: Even division, error cases, edge cases
- ceil_divide: Ceiling division with various inputs
- product: Multiple arguments, edge cases
"""

import pytest

from dlcalc.utils.math import ceil_divide, product, safe_divide


class TestSafeDivide:
    """Test cases for the safe_divide function."""

    def test_even_division(self):
        """Test safe_divide with evenly divisible numbers."""
        assert safe_divide(10, 2) == 5
        assert safe_divide(100, 10) == 10
        assert safe_divide(1000, 50) == 20
        assert safe_divide(0, 5) == 0

    def test_division_by_one(self):
        """Test safe_divide with denominator of 1."""
        assert safe_divide(1, 1) == 1
        assert safe_divide(100, 1) == 100
        assert safe_divide(999, 1) == 999

    def test_same_numerator_denominator(self):
        """Test safe_divide when numerator equals denominator."""
        assert safe_divide(5, 5) == 1
        assert safe_divide(100, 100) == 1
        assert safe_divide(1234, 1234) == 1

    def test_uneven_division_raises_error(self):
        """Test that uneven division raises ValueError."""
        with pytest.raises(ValueError, match="10 not divisible by 3"):
            safe_divide(10, 3)

        with pytest.raises(ValueError, match="100 not divisible by 7"):
            safe_divide(100, 7)

    def test_error_message_includes_valid_denominators(self):
        """Test that error message includes list of valid denominators."""
        with pytest.raises(ValueError) as exc_info:
            safe_divide(12, 5)

        error_msg = str(exc_info.value)
        assert "12 not divisible by 5" in error_msg
        assert "valid denominators: [1, 2, 3, 4, 6]" in error_msg

    def test_large_numbers(self):
        """Test safe_divide with large numbers."""
        assert safe_divide(1000000, 1000) == 1000
        assert safe_divide(1024 * 1024, 1024) == 1024

        # Large number not divisible
        with pytest.raises(ValueError):
            safe_divide(1000001, 1000)

    def test_prime_numbers(self):
        """Test safe_divide with prime numbers."""
        # Prime number only divisible by 1 and itself
        assert safe_divide(17, 1) == 17
        assert safe_divide(17, 17) == 1

        with pytest.raises(ValueError) as exc_info:
            safe_divide(17, 5)

        # Prime numbers have only 1 as valid divisor (besides themselves)
        assert "valid denominators: [1]" in str(exc_info.value)

    def test_negative_numbers(self):
        """Test safe_divide behavior with negative numbers."""
        # Python's modulo works differently with negatives
        # This tests current behavior
        assert safe_divide(-10, 2) == -5
        assert safe_divide(10, -2) == -5
        assert safe_divide(-10, -2) == 5

    def test_powers_of_two(self):
        """Test safe_divide with powers of two (common in ML contexts)."""
        assert safe_divide(1024, 2) == 512
        assert safe_divide(1024, 4) == 256
        assert safe_divide(1024, 8) == 128
        assert safe_divide(1024, 16) == 64
        assert safe_divide(1024, 32) == 32


class TestCeilDivide:
    """Test cases for the ceil_divide function."""

    def test_even_division(self):
        """Test ceil_divide when result is exact."""
        assert ceil_divide(10, 2) == 5
        assert ceil_divide(100, 10) == 10
        assert ceil_divide(1000, 50) == 20

    def test_uneven_division_rounds_up(self):
        """Test ceil_divide rounds up for uneven division."""
        assert ceil_divide(10, 3) == 4  # 3.333... -> 4
        assert ceil_divide(7, 2) == 4  # 3.5 -> 4
        assert ceil_divide(1, 2) == 1  # 0.5 -> 1
        assert ceil_divide(99, 10) == 10  # 9.9 -> 10

    def test_division_by_one(self):
        """Test ceil_divide with denominator of 1."""
        assert ceil_divide(1, 1) == 1
        assert ceil_divide(100, 1) == 100
        assert ceil_divide(999, 1) == 999

    def test_zero_numerator(self):
        """Test ceil_divide with zero numerator."""
        assert ceil_divide(0, 1) == 0
        assert ceil_divide(0, 10) == 0
        assert ceil_divide(0, 100) == 0

    def test_large_numbers(self):
        """Test ceil_divide with large numbers."""
        assert ceil_divide(1000000, 999) == 1002  # ceiling of 1001.001...
        assert ceil_divide(1024 * 1024 + 1, 1024) == 1025

    def test_numerator_less_than_denominator(self):
        """Test ceil_divide when numerator < denominator."""
        assert ceil_divide(1, 10) == 1
        assert ceil_divide(5, 10) == 1
        assert ceil_divide(9, 10) == 1
        assert ceil_divide(10, 10) == 1
        assert ceil_divide(11, 10) == 2

    def test_negative_numbers(self):
        """Test ceil_divide with negative numbers."""
        # Python's math.ceil behavior with negatives
        assert ceil_divide(-10, 3) == -3  # -3.333... -> -3
        assert ceil_divide(10, -3) == -3  # -3.333... -> -3
        assert ceil_divide(-10, -3) == 4  # 3.333... -> 4

    def test_comparison_with_safe_divide(self):
        """Test that ceil_divide matches safe_divide for even divisions."""
        test_cases = [(100, 10), (1024, 32), (1000, 50)]
        for num, denom in test_cases:
            assert ceil_divide(num, denom) == safe_divide(num, denom)

    def test_floating_point_precision(self):
        """Test ceil_divide handles floating point correctly."""
        # These should all round up to the next integer
        assert ceil_divide(1000001, 1000000) == 2
        assert ceil_divide(100000000000001, 100000000000000) == 2


class TestProduct:
    """Test cases for the product function."""

    def test_single_argument(self):
        """Test product with single argument."""
        assert product(5) == 5
        assert product(0) == 0
        assert product(1) == 1
        assert product(-5) == -5

    def test_two_arguments(self):
        """Test product with two arguments."""
        assert product(2, 3) == 6
        assert product(10, 10) == 100
        assert product(0, 100) == 0

    def test_multiple_arguments(self):
        """Test product with multiple arguments."""
        assert product(2, 3, 4) == 24
        assert product(1, 2, 3, 4, 5) == 120
        assert product(10, 10, 10) == 1000

    def test_with_zero(self):
        """Test product when one argument is zero."""
        assert product(0, 1, 2, 3) == 0
        assert product(100, 0, 200) == 0
        assert product(1, 2, 3, 0, 4, 5) == 0

    def test_with_ones(self):
        """Test product with ones."""
        assert product(1, 1, 1, 1) == 1
        assert product(5, 1, 3, 1) == 15
        assert product(1, 10) == 10

    def test_negative_numbers(self):
        """Test product with negative numbers."""
        assert product(-2, 3) == -6
        assert product(-2, -3) == 6
        assert product(-1, -1, -1) == -1
        assert product(-2, -2, -2) == -8
        assert product(-2, 3, -4) == 24

    def test_large_numbers(self):
        """Test product with large numbers."""
        assert product(1000, 1000) == 1000000
        assert product(100, 100, 100) == 1000000

        # Common ML dimensions
        assert product(32, 1024, 768) == 25165824  # batch * seq * hidden

    def test_empty_arguments_assertion(self):
        """Test that product with no arguments raises assertion."""
        with pytest.raises(AssertionError):
            product()

    def test_powers_of_two(self):
        """Test product with powers of two (common in ML)."""
        assert product(2, 2, 2, 2) == 16
        assert product(4, 8, 16) == 512
        assert product(32, 64) == 2048

    def test_factorials_equivalent(self):
        """Test product can compute factorials."""
        # 5! = 5 * 4 * 3 * 2 * 1
        assert product(5, 4, 3, 2, 1) == 120
        # 6!
        assert product(6, 5, 4, 3, 2, 1) == 720

    def test_commutative_property(self):
        """Test that product is commutative."""
        assert product(2, 3, 4) == product(4, 3, 2)
        assert product(5, 10, 2) == product(2, 5, 10)

    def test_associative_property(self):
        """Test that product is associative."""
        # (2 * 3) * 4 = 2 * (3 * 4)
        assert product(product(2, 3), 4) == product(2, product(3, 4))

    def test_unpacking_lists(self):
        """Test product with unpacked lists."""
        numbers = [2, 3, 4, 5]
        assert product(*numbers) == 120

        dimensions = [32, 128, 768]  # batch, seq_len, hidden_dim
        assert product(*dimensions) == 3145728


class TestIntegration:
    """Integration tests for math utility functions."""

    def test_tensor_dimension_calculations(self):
        """Test realistic tensor dimension calculations."""
        # Calculate total elements in a partitioned tensor
        batch_size = 32
        seq_len = 1024
        hidden_dim = 768
        tp_degree = 8

        total_elements = product(batch_size, seq_len, hidden_dim)
        elements_per_partition = safe_divide(total_elements, tp_degree)

        assert total_elements == 25165824
        assert elements_per_partition == 3145728

    def test_memory_page_calculations(self):
        """Test memory page calculations with ceiling division."""
        page_size = 4096  # 4KB pages

        # Small allocation needs 1 page
        assert ceil_divide(100, page_size) == 1

        # Exact page boundary
        assert ceil_divide(4096, page_size) == 1
        assert ceil_divide(8192, page_size) == 2

        # Just over page boundary
        assert ceil_divide(4097, page_size) == 2
        assert ceil_divide(8193, page_size) == 3

    def test_batch_splitting(self):
        """Test batch splitting across devices."""
        global_batch_size = 512
        num_gpus = 8

        # Even split
        per_gpu_batch = safe_divide(global_batch_size, num_gpus)
        assert per_gpu_batch == 64

        # Uneven split should fail with safe_divide
        with pytest.raises(ValueError):
            safe_divide(513, num_gpus)

        # But works with ceil_divide
        assert ceil_divide(513, num_gpus) == 65

    def test_parameter_sharding(self):
        """Test parameter sharding calculations."""
        # Transformer layer dimensions
        vocab_size = 50000
        hidden_dim = 1024
        num_heads = 16

        # Total parameters in embedding
        embedding_params = product(vocab_size, hidden_dim)

        # Shard across tensor parallel dimension
        tp_degree = 4
        params_per_shard = safe_divide(embedding_params, tp_degree)

        assert embedding_params == 51200000
        assert params_per_shard == 12800000

        # Attention head splitting
        head_dim = safe_divide(hidden_dim, num_heads)
        assert head_dim == 64

    def test_combined_operations(self):
        """Test combining multiple math operations."""
        # Calculate memory for batch of sequences
        batch_size = 32
        seq_len = 2048
        hidden_dim = 768
        bytes_per_element = 2  # fp16

        # Total elements
        total_elements = product(batch_size, seq_len, hidden_dim)

        # Total bytes
        total_bytes = product(total_elements, bytes_per_element)

        # Divide into chunks of 1MB (1024 * 1024 bytes)
        chunk_size = product(1024, 1024)
        num_chunks = ceil_divide(total_bytes, chunk_size)

        assert total_elements == 50331648
        assert total_bytes == 100663296
        assert num_chunks == 96  # ~96MB
