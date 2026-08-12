# Utils Module Unit Tests

This directory contains comprehensive unit tests for utility modules in the dlcalc package.

## Test Files

### test_data.py
Tests for `dlcalc.utils.data` module:
- **Size class** (9 tests): Basic operations, arithmetic, edge cases
- **TensorRepr class** (14 tests): Tensor partitioning, shape calculations, size methods
- **Integration tests** (3 tests): Combined usage scenarios

**Coverage**: 100% (52 statements)

### test_math.py
Tests for `dlcalc.utils.math` module:
- **safe_divide** (9 tests): Even division, error handling, edge cases
- **ceil_divide** (10 tests): Ceiling division, rounding behavior
- **product** (13 tests): Multiple arguments, mathematical properties
- **Integration tests** (5 tests): Realistic ML calculations

**Coverage**: 100% (17 statements)

## Running Tests

```bash
# Run all utils tests
uv run pytest tests/unit/utils/ -v

# Run with coverage report
uv run pytest tests/unit/utils/ --cov=dlcalc.utils --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/utils/test_math.py -v

# Run specific test class
uv run pytest tests/unit/utils/test_data.py::TestSize -v

# Run specific test
uv run pytest tests/unit/utils/test_math.py::TestSafeDivide::test_even_division -v
```

## Test Organization

Tests are organized into classes by the component they test:
- `TestClassName`: Unit tests for a specific class or function
- `TestIntegration`: Integration tests showing realistic usage

Each test method has a descriptive name and docstring explaining what it tests.

## Key Testing Patterns

1. **Edge Cases**: Zero values, negative numbers, large numbers
2. **Error Conditions**: Invalid inputs that should raise exceptions
3. **Mathematical Properties**: Commutativity, associativity, etc.
4. **ML-Specific Scenarios**: Tensor dimensions, batch splitting, memory calculations
5. **Integration**: Combined usage of multiple functions

## Total Test Coverage

- **62 total tests** across both modules
- **100% code coverage** (69 statements)
- Comprehensive edge case testing
- Real-world ML scenario validation
