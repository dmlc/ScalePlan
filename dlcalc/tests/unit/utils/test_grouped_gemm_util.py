import pytest
from dlcalc.utils.grouped_gemm_util import (
    grouped_gemm_measured,
    grouped_mlp_up_fwd_time_s,
    grouped_mlp_down_bwd_time_s,
)
from dlcalc.utils.hardware import DType


def test_measured_flag():
    assert grouped_gemm_measured("p6-b200.48xlarge", DType.BF16)
    assert not grouped_gemm_measured("p6-b200.48xlarge", DType.FP8)
    assert not grouped_gemm_measured("p5.48xlarge", DType.BF16)


def test_fwd_positive_and_scales_with_experts():
    # more local experts (num_gemms) at same tokens/expert -> more total work -> more time
    t16 = grouped_mlp_up_fwd_time_s(16, 192, 6144, 3840)
    t4 = grouped_mlp_up_fwd_time_s(4, 192, 6144, 3840)
    assert t16 > t4 > 0


def test_unmeasured_shape_raises():
    with pytest.raises(KeyError):
        grouped_mlp_up_fwd_time_s(16, 192, 9999, 3840)


def test_tokens_interpolation_monotonic():
    lo = grouped_mlp_down_bwd_time_s(16, 128, 4096, 2560)
    hi = grouped_mlp_down_bwd_time_s(16, 512, 4096, 2560)
    assert hi > lo > 0
