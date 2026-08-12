"""Unit tests for the measured CPU-dispatch floor (kernel_launch)."""

from dlcalc.utils.kernel_launch import dispatch_time_per_microbatch_s


def test_scales_with_layers_and_experts():
    # more layers/stage and more local experts -> more launch time
    base = dispatch_time_per_microbatch_s(
        layers_per_stage=2, n_local_experts=16, device_name="p6-b200.48xlarge"
    )
    more_layers = dispatch_time_per_microbatch_s(
        layers_per_stage=4, n_local_experts=16, device_name="p6-b200.48xlarge"
    )
    more_exp = dispatch_time_per_microbatch_s(
        layers_per_stage=2, n_local_experts=128, device_name="p6-b200.48xlarge"
    )
    assert more_layers > base
    assert more_exp > base


def test_ep_permute_term_only_when_ep_gt_1():
    no_ep = dispatch_time_per_microbatch_s(
        layers_per_stage=3,
        n_local_experts=16,
        device_name="p6-b200.48xlarge",
        ep=1,
        moe_frequency=1.0,
    )
    with_ep = dispatch_time_per_microbatch_s(
        layers_per_stage=3,
        n_local_experts=16,
        device_name="p6-b200.48xlarge",
        ep=8,
        moe_frequency=1.0,
    )
    assert with_ep > no_ep


def test_positive_and_host_floor():
    # even a dense 1-layer stage has the fixed per-microbatch host term
    t = dispatch_time_per_microbatch_s(
        layers_per_stage=1, n_local_experts=1, device_name="p6-b200.48xlarge"
    )
    assert t > 0.02  # >= host floor ~24ms


def test_unknown_device_uses_default_gap():
    t = dispatch_time_per_microbatch_s(
        layers_per_stage=2, n_local_experts=16, device_name="mystery-gpu"
    )
    assert t > 0
