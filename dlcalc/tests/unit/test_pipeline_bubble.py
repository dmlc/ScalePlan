"""Unit tests for the (interleaved) 1F1B pipeline bubble fraction.

Tests the REAL implementation (dlcalc.utils.pipeline), not a local copy.
The previous version of this file duplicated the formula under test, so a
defect in the formula was invisible to it.

Physics (Narayanan et al. 2021, Megatron-LM, interleaved 1F1B):
    per step, each rank computes n_microbatches microbatch-times of work and
    idles (pp-1)/vpp microbatch-times during pipeline fill+drain. The ramp IS
    the entire bubble -- there is no additional fill/drain term, and no cap:

        bubble / compute = (pp - 1) / (vpp * n_microbatches)

    which legitimately exceeds 1 when n_microbatches < (pp-1)/vpp.

Cross-validated against rank0 profiler traces of the 18b runs
(dlcalc/tests/data/18b_real_mfu_P6_64node.csv): the mbs sweep at pp16
matches (pp-1)/n_mb, and the old fill/drain-weighted formula over-counted
2-3x at moderate n_mb while under-counting 4x at n_mb < pp-1.

TestAgainst1F1BSimulation additionally verifies the vpp=1 formula to machine
precision against a discrete-event simulation of the schedule's dependency
graph (an independent oracle -- no closed form appears in the simulator).
"""

import pytest

from dlcalc.utils.pipeline import (
    compute_pipeline_bubble_fraction,
    compute_stage_imbalance_extra_time_s,
)


def _simulate_1f1b_makespan(
    pp: int, n_mb: int, t_f: float | list[float], t_b: float | list[float]
) -> float:
    """Discrete-event simulation of non-interleaved 1F1B; returns the makespan.

    t_f / t_b may be scalars (uniform stages) or per-stage lists (imbalanced
    stages, e.g. embedding on first / LM head on last).

    Encodes ONLY the schedule's dependency structure -- no closed-form formula:
      * fwd(stage i, mb) requires fwd(stage i-1, mb)
      * bwd(stage i, mb) requires bwd(stage i+1, mb); last stage: its own fwd
      * per-rank op order (Megatron 1F1B): min(n_mb, pp-1-i) warmup fwds,
        steady (fwd, bwd) pairs, then drain bwds.
    """
    t_fs = [t_f] * pp if isinstance(t_f, (int, float)) else list(t_f)
    t_bs = [t_b] * pp if isinstance(t_b, (int, float)) else list(t_b)
    assert len(t_fs) == pp and len(t_bs) == pp

    seq: dict[int, list[tuple[str, int]]] = {}
    for i in range(pp):
        warmup = min(n_mb, pp - 1 - i)
        ops: list[tuple[str, int]] = []
        nf = nb = 0
        for _ in range(warmup):
            ops.append(("F", nf))
            nf += 1
        while nf < n_mb:
            ops.append(("F", nf))
            nf += 1
            ops.append(("B", nb))
            nb += 1
        while nb < n_mb:
            ops.append(("B", nb))
            nb += 1
        seq[i] = ops

    done_f: dict[tuple[int, int], float] = {}
    done_b: dict[tuple[int, int], float] = {}
    free = [0.0] * pp
    ptr = [0] * pp
    scheduled, total = 0, pp * n_mb * 2
    while scheduled < total:
        progressed = False
        for i in range(pp):
            while ptr[i] < len(seq[i]):
                kind, mb = seq[i][ptr[i]]
                if kind == "F":
                    dep = 0.0 if i == 0 else done_f.get((i - 1, mb))
                else:
                    dep = done_f.get((i, mb)) if i == pp - 1 else done_b.get((i + 1, mb))
                if dep is None:
                    break  # blocked on a peer stage; try other ranks first
                end = max(free[i], dep) + (t_fs[i] if kind == "F" else t_bs[i])
                free[i] = end
                (done_f if kind == "F" else done_b)[(i, mb)] = end
                ptr[i] += 1
                scheduled += 1
                progressed = True
        assert progressed, f"schedule deadlock: pp={pp} n_mb={n_mb}"
    return max(free)


class TestAgainst1F1BSimulation:
    """Property check: formula == simulated schedule, for all (pp, n_mb, tb/tf)."""

    @pytest.mark.parametrize("pp", [2, 4, 8, 16])
    @pytest.mark.parametrize("n_mb", [1, 2, 4, 8, 16, 32, 256])
    @pytest.mark.parametrize("t_b_over_t_f", [1.0, 2.0, 2.9])
    def test_formula_matches_simulated_makespan(
        self, pp: int, n_mb: int, t_b_over_t_f: float
    ) -> None:
        t_f, t_b = 1.0, t_b_over_t_f
        makespan = _simulate_1f1b_makespan(pp, n_mb, t_f, t_b)
        compute = n_mb * (t_f + t_b)
        sim_bubble_fraction = (makespan - compute) / compute
        formula = compute_pipeline_bubble_fraction(pp=pp, vpp=1, n_microbatches=n_mb)
        # rel tolerance: the simulator accumulates 2*n_mb float additions of
        # t_b (2.9 is not exactly representable), so exact == is not attainable.
        assert sim_bubble_fraction == pytest.approx(formula, rel=1e-9, abs=1e-9)
        # equivalently: makespan == (n_mb + pp - 1) * (t_f + t_b), independent
        # of the tb/tf ratio.
        assert makespan == pytest.approx((n_mb + pp - 1) * (t_f + t_b), rel=1e-9)


class TestStageImbalanceAgainstSimulation:
    """Property check for compute_stage_imbalance_extra_time_s: the full
    imbalanced-pipeline makespan must decompose exactly into
    balanced compute + (pp-1)-ramp bubble + the imbalance extra."""

    @pytest.mark.parametrize("pp", [2, 4, 8, 16])
    @pytest.mark.parametrize("n_mb", [1, 2, 4, 8, 32, 256])
    @pytest.mark.parametrize(
        ("t_first_extra", "t_last_extra"),
        [(0.0, 0.5), (0.01, 1.0), (0.05, 3.0), (0.02, 0.02)],
    )
    def test_decomposition_matches_simulation(
        self, pp: int, n_mb: int, t_first_extra: float, t_last_extra: float
    ) -> None:
        # uniform transformer stage time (fwd 1, bwd 2), embedding extra on the
        # first stage, LM-head+CE extra on the last (each split 1/3 fwd, 2/3 bwd).
        t_f = [1.0] * pp
        t_b = [2.0] * pp
        t_f[0] += t_first_extra / 3
        t_b[0] += 2 * t_first_extra / 3
        t_f[-1] += t_last_extra / 3
        t_b[-1] += 2 * t_last_extra / 3
        makespan = _simulate_1f1b_makespan(pp, n_mb, t_f, t_b)

        t_stage = 3.0
        balanced = n_mb * t_stage
        bubble = t_stage * (pp - 1) if pp > 1 else 0.0
        extra = compute_stage_imbalance_extra_time_s(
            pp=pp,
            n_microbatches=n_mb,
            t_first_extra_s=t_first_extra,
            t_last_extra_s=t_last_extra,
        )
        assert makespan == pytest.approx(balanced + bubble + extra, rel=1e-9)

    def test_pp1_charges_both_extras_serially(self) -> None:
        assert compute_stage_imbalance_extra_time_s(
            pp=1, n_microbatches=8, t_first_extra_s=0.5, t_last_extra_s=2.0
        ) == 8 * (0.5 + 2.0)

    def test_scales_with_n_microbatches(self) -> None:
        # the dominant (last-stage) extra paces every microbatch: ~n_mb * t_last.
        e256 = compute_stage_imbalance_extra_time_s(
            pp=16, n_microbatches=256, t_first_extra_s=0.0, t_last_extra_s=0.087
        )
        assert e256 == pytest.approx(256 * 0.087, rel=0.01)

    def test_zero_extras_zero_cost(self) -> None:
        assert (
            compute_stage_imbalance_extra_time_s(
                pp=8, n_microbatches=32, t_first_extra_s=0.0, t_last_extra_s=0.0
            )
            == 0.0
        )

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            compute_stage_imbalance_extra_time_s(
                pp=0, n_microbatches=8, t_first_extra_s=0.0, t_last_extra_s=0.0
            )
        with pytest.raises(ValueError):
            compute_stage_imbalance_extra_time_s(
                pp=8, n_microbatches=0, t_first_extra_s=0.0, t_last_extra_s=0.0
            )
        with pytest.raises(ValueError):
            compute_stage_imbalance_extra_time_s(
                pp=8, n_microbatches=8, t_first_extra_s=-1.0, t_last_extra_s=0.0
            )


class TestTextbook1F1BValues:
    """Exact values from the (p-1)/(v*m) formula."""

    def test_pp8_n32(self) -> None:
        # 18b run 6e7b041f (pp8, gbs2048, mbs1, dp64 -> n_mb=32).
        # Trace analyzer independently prints 7/32 = 21.9% for this config.
        assert compute_pipeline_bubble_fraction(pp=8, vpp=1, n_microbatches=32) == 7 / 32

    def test_pp16_n256(self) -> None:
        # 18b run bc59887c (pp16, gbs8192, mbs1, dp32 -> n_mb=256).
        assert compute_pipeline_bubble_fraction(pp=16, vpp=1, n_microbatches=256) == 15 / 256

    def test_pp4_large_n(self) -> None:
        assert compute_pipeline_bubble_fraction(pp=4, vpp=1, n_microbatches=128) == 3 / 128

    def test_vpp_divides_bubble(self) -> None:
        # Interleaving with vpp virtual stages divides the fill/drain ramp by vpp.
        assert compute_pipeline_bubble_fraction(pp=8, vpp=2, n_microbatches=32) == 7 / 64
        assert compute_pipeline_bubble_fraction(pp=8, vpp=4, n_microbatches=32) == 7 / 128


class TestSmallMicrobatchRegime:
    """n_microbatches < pp-1: bubble exceeds compute time. No cap."""

    def test_pp16_n4_exceeds_one(self) -> None:
        # 18b run eaf87ab0 (pp16, gbs512, mbs4, dp32 -> n_mb=4): the rank idles
        # 15 microbatch-times while computing only 4 -> bubble = 375% of compute.
        # The old formula capped this at (pp-1)/pp = 93.75%, under-predicting the
        # measured 12.6s step as 3.4s.
        assert compute_pipeline_bubble_fraction(pp=16, vpp=1, n_microbatches=4) == 15 / 4

    def test_pp16_n8(self) -> None:
        # 18b run a92ab978: bubble = 15/8 = 187.5% of compute.
        assert compute_pipeline_bubble_fraction(pp=16, vpp=1, n_microbatches=8) == 15 / 8

    def test_single_microbatch(self) -> None:
        # One microbatch through a pp-deep pipeline: rank works 1 unit,
        # waits pp-1 units.
        assert compute_pipeline_bubble_fraction(pp=4, vpp=1, n_microbatches=1) == 3.0

    def test_pp_equals_n_coincidence(self) -> None:
        # At n_mb == pp, (pp-1)/n_mb == (pp-1)/pp: the only point where the old
        # else-branch cap happened to be correct (18b run 093b8d67, pp16/n16).
        assert compute_pipeline_bubble_fraction(pp=16, vpp=1, n_microbatches=16) == 15 / 16


class TestNoRegimeBoundary:
    """The schedule has no phase transition at n_mb = 2*(pp-1); the old
    piecewise formula introduced a spurious one."""

    def test_continuous_across_old_boundary_pp8(self) -> None:
        # Old boundary was n_mb > 2*(pp-1) = 14.
        for n in (13, 14, 15):
            assert compute_pipeline_bubble_fraction(pp=8, vpp=1, n_microbatches=n) == 7 / n

    def test_continuous_across_old_boundary_pp4(self) -> None:
        for n in (5, 6, 7):
            assert compute_pipeline_bubble_fraction(pp=4, vpp=1, n_microbatches=n) == 3 / n


class TestEdgeCases:
    def test_pp1_no_bubble(self) -> None:
        assert compute_pipeline_bubble_fraction(pp=1, vpp=1, n_microbatches=10) == 0.0

    def test_pp2_minimal_pipeline(self) -> None:
        assert compute_pipeline_bubble_fraction(pp=2, vpp=1, n_microbatches=10) == 1 / 10

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            compute_pipeline_bubble_fraction(pp=8, vpp=1, n_microbatches=0)
        with pytest.raises(ValueError):
            compute_pipeline_bubble_fraction(pp=0, vpp=1, n_microbatches=8)
        with pytest.raises(ValueError):
            compute_pipeline_bubble_fraction(pp=8, vpp=0, n_microbatches=8)


class TestMonotonicity:
    def test_decreases_with_more_microbatches(self) -> None:
        bubbles = [
            compute_pipeline_bubble_fraction(pp=4, vpp=1, n_microbatches=n)
            for n in (2, 4, 10, 50, 200)
        ]
        assert all(b1 > b2 for b1, b2 in zip(bubbles, bubbles[1:]))

    def test_increases_with_pp(self) -> None:
        bubbles = [
            compute_pipeline_bubble_fraction(pp=pp, vpp=1, n_microbatches=50)
            for pp in (2, 4, 8, 16)
        ]
        assert all(b1 < b2 for b1, b2 in zip(bubbles, bubbles[1:]))

    def test_decreases_with_vpp(self) -> None:
        bubbles = [
            compute_pipeline_bubble_fraction(pp=8, vpp=vpp, n_microbatches=50) for vpp in (1, 2, 4)
        ]
        assert all(b1 > b2 for b1, b2 in zip(bubbles, bubbles[1:]))
