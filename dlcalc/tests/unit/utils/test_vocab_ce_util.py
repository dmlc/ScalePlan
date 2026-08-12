"""Unit tests for the measured cross-entropy lookup."""

from dlcalc.utils.vocab_ce_util import cross_entropy_fwd_bwd_time_s, vocab_ce_measured


def test_measured_flag():
    assert vocab_ce_measured("p6-b200.48xlarge")
    assert not vocab_ce_measured("p5.48xlarge")


def test_scales_with_tokens():
    lo = cross_entropy_fwd_bwd_time_s(m_tokens=4096, vocab_padded=257664)
    hi = cross_entropy_fwd_bwd_time_s(m_tokens=16384, vocab_padded=257664)
    assert hi > lo > 0


def test_much_larger_than_naive_4_pass():
    # measured CE is ~5x the naive 4-fp32-HBM-pass estimate at M=8192.
    t = cross_entropy_fwd_bwd_time_s(m_tokens=8192, vocab_padded=257664)
    naive_4pass = 4 * 8192 * 257664 * 4 / 8e12
    assert t > 3 * naive_4pass
