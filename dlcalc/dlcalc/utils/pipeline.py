"""Pipeline-parallelism schedule costs (1F1B and interleaved 1F1B)."""


def compute_stage_imbalance_extra_time_s(
    pp: int,
    n_microbatches: int,
    t_first_extra_s: float,
    t_last_extra_s: float,
) -> float:
    """Extra iteration time from first/last-stage compute the other stages lack.

    Megatron places the embedding on the first pipeline stage and the LM head
    (+ cross-entropy) on the last, ON TOP of those stages' transformer layers.
    In 1F1B the slowest stage paces the steady state, so with per-microbatch
    extras t_first (embedding) and t_last (LM head + CE), fwd+bwd combined,
    the iteration decomposes EXACTLY (verified against a discrete-event
    simulation of the 1F1B dependency graph, see test_pipeline_bubble.py) as:

        T = n_mb * t_stage                  (balanced transformer compute)
          + (pp - 1) * t_stage              (fill/drain bubble)
          + (n_mb - 1) * max(t_first, t_last) + t_first + t_last   <- this term

    provided t_last >= t_first (true in practice: the LM-head GEMM is orders
    of magnitude above the embedding gather/scatter). If t_first > t_last the
    1F1B warmup partially hides the first-stage extra and this expression is
    a slight over-estimate (upper bound) -- documented, not the real regime.

    pp == 1 has no pipelining: every microbatch pays both extras serially.
    """
    if pp < 1 or n_microbatches < 1:
        raise ValueError(f"invalid config: pp={pp} n_microbatches={n_microbatches}")
    if t_first_extra_s < 0 or t_last_extra_s < 0:
        raise ValueError(f"negative stage extras: first={t_first_extra_s} last={t_last_extra_s}")
    if pp == 1:
        return n_microbatches * (t_first_extra_s + t_last_extra_s)
    return (
        (n_microbatches - 1) * max(t_first_extra_s, t_last_extra_s)
        + t_first_extra_s
        + t_last_extra_s
    )


def compute_pipeline_bubble_fraction(pp: int, vpp: int, n_microbatches: int) -> float:
    """Pipeline bubble time as a fraction of per-rank compute time.

    (Interleaved) 1F1B, per Narayanan et al. 2021 (Megatron-LM): with
    per-microbatch stage time t, each rank computes n_microbatches * t per
    step and the last microbatch exits (pp - 1)/vpp stage-times after the
    rank's own work could have finished:

        T_iter   = (n_microbatches + (pp - 1) / vpp) * t
        T_bubble = ((pp - 1) / vpp) * t
        bubble / compute = (pp - 1) / (vpp * n_microbatches)

    The fill/drain ramp IS the entire bubble -- 1F1B has no additional
    per-microbatch inefficiency during warmup/cooldown, and there is no
    regime change at any n_microbatches. The fraction legitimately exceeds
    1 when n_microbatches < (pp - 1) / vpp: the rank then idles longer than
    it computes (e.g. pp=16, n_microbatches=4 -> 3.75x compute time).
    Validated against rank0 profiler traces of the 18b pp16 mbs sweep
    (dlcalc/tests/data/18b_real_mfu_P6_64node.csv).
    """
    if pp < 1 or vpp < 1 or n_microbatches < 1:
        raise ValueError(
            f"invalid pipeline config: pp={pp} vpp={vpp} n_microbatches={n_microbatches}"
        )
    return (pp - 1) / (vpp * n_microbatches)
