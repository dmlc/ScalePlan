import math


def safe_divide(numerator: int, denominator: int) -> int:
    """return numerator / denominator, or throw an exception if its not divisible."""
    if numerator % denominator != 0:
        valid_denominators = []
        for candidate_denominator in range(1, numerator):
            if numerator % candidate_denominator == 0:
                valid_denominators.append(candidate_denominator)
        raise ValueError(
            f"{numerator} not divisible by {denominator}. valid denominators: {valid_denominators}"
        )

    return numerator // denominator


def ceil_divide(numerator: int, denominator: int) -> int:
    """return ceil(numerator / denominator)"""
    return int(math.ceil(numerator / denominator))


def product(*multiplicands: int) -> int:
    """return the product of a sequence of multiplicands."""
    assert len(multiplicands) > 0
    product = 1
    for multiplicand in multiplicands:
        product *= multiplicand

    return product
