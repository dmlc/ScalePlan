"""Unit tests for the empirical cross-node desync / async-overlap penalty."""

from dlcalc.utils.desync import cross_node_desync_multiplier


def _m(pp, ep, a2a_frac, it=10.0, etp=1, gpn=8):
    return cross_node_desync_multiplier(
        pp=pp,
        ep=ep,
        expert_tp=etp,
        gpus_per_node=gpn,
        ep_a2a_time_s=a2a_frac * it,
        iteration_time_s=it,
    )


def test_single_node_shallow_no_penalty():
    # ep<=8 (1 node), pp<=4 (1 node, shallow): no driver applies -> 1.0
    assert _m(pp=4, ep=8, a2a_frac=0.03) == 1.0
    assert _m(pp=1, ep=1, a2a_frac=0.0) == 1.0


def test_inter_node_ep_a2a_penalty():
    # ep16 spans 2 nodes; penalty = 1.2 * a2a_frac
    m = _m(pp=2, ep=16, a2a_frac=0.28)
    assert abs(m - (1 + 1.2 * 0.28)) < 1e-9


def test_no_a2a_penalty_when_ep_intranode():
    # ep8 fits one node -> A2A term off even with a large a2a_frac
    assert _m(pp=2, ep=8, a2a_frac=0.28) == 1.0


def test_deep_pipeline_penalty():
    # pp16 spans 2 nodes -> +0.5
    m = _m(pp=16, ep=8, a2a_frac=0.02)
    assert abs(m - 1.5) < 1e-9


def test_pp8_intranode_overlap_credit():
    # pp8 within one node: -0.18 overlap credit (real faster than 1F1B accounting)
    m = _m(pp=8, ep=8, a2a_frac=0.03)
    assert abs(m - 0.82) < 1e-9


def test_multiplier_floor():
    # penalty can't drive the multiplier below 0.5
    m = cross_node_desync_multiplier(
        pp=8, ep=8, expert_tp=1, gpus_per_node=8, ep_a2a_time_s=0.0, iteration_time_s=1.0
    )
    assert m >= 0.5
