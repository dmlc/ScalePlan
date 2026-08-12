"""Empirical cross-node desync / async-overlap penalty.

Motivation (2026-07-17): after all first-principles fixes (measured comm +
grouped-GEMM + CE, CPU-dispatch floor, 1F1B last-stage bubble, the tp=1
phantom-comm bug), the residual golden-set error is real but not
first-principles-modelable, and splits into THREE systematic effects, each
tied to a physical driver:

  (1) INTER-NODE EP ALL-TO-ALL straggler. When the expert-parallel group spans
      >1 node, the dispatch/combine all-to-all is desync-contaminated (the
      slowest rank in the wide group paces it). The over-prediction scales with
      how large a fraction of the step that A2A already is — so the penalty is
      c_a2a * (EP-A2A wire time / iteration time), applied only when ep spans
      nodes. This is what makes 5p3b ep16 (A2A ~28% of the step) run ~30-50%
      slower than the straggler-free model.

  (2) DEEP-PIPELINE desync. A pipeline whose stages span >1 node loses to
      cross-node 1F1B send/recv desync (traces: PP send/recv p50 2.4ms but max
      1957ms). Penalty c_pp * (pp_nodes - 1). Drives 18b pp16 (2 nodes).

  (3) ASYNC-OVERLAP CREDIT for a deep pipeline that fits within ONE node
      (pp>=8, pp_nodes==1): the heavy LM-head/tail on the last stage overlaps
      other stages' compute over NVLink better than the straggler-free 1F1B
      accounting assumes, so the real step is FASTER than predicted. A negative
      penalty c_ovl. Corrects 18b pp8 (real ~24% MFU vs ~16% predicted).

Form: iteration_time *= (1 + penalty), penalty =
    c_a2a * a2a_frac[ep spans nodes] + c_pp*(pp_nodes-1) + c_ovl*[pp>=8, 1 node]
where pp_nodes = ceil(pp / gpus_per_node), ep_nodes = ceil(ep*etp/gpus_per_node).
"""

from __future__ import annotations

import math

# Empirical coefficients (see module docstring). NOT first-principles.
_DESYNC_A2A_FRAC = 1.2  # per unit of (inter-node EP-A2A time / iteration time)
_DESYNC_PER_PP_NODE = 0.5  # per extra node the pipeline spans
_OVERLAP_CREDIT_PP8_INTRANODE = -0.18  # deep pipeline within one node: LM/tail overlaps


def cross_node_desync_multiplier(
    pp: int,
    ep: int,
    expert_tp: int,
    gpus_per_node: int,
    ep_a2a_time_s: float,
    iteration_time_s: float,
) -> float:
    """Multiplier applied to the straggler-free iteration time (see module docs).

    Args:
        pp, ep, expert_tp: parallelism degrees.
        gpus_per_node: devices per node (node = collective-locality boundary).
        ep_a2a_time_s: modeled per-step EP all-to-all wire time (the audit total).
        iteration_time_s: the straggler-free iteration time BEFORE this penalty.

    Returns 1.0 when no driver applies (single-node EP and PP, shallow pipeline).
    """
    ep_nodes = max(1, math.ceil(ep * expert_tp / gpus_per_node))
    pp_nodes = max(1, math.ceil(pp / gpus_per_node))

    penalty = 0.0
    if ep_nodes > 1 and iteration_time_s > 0:
        penalty += _DESYNC_A2A_FRAC * (ep_a2a_time_s / iteration_time_s)
    penalty += _DESYNC_PER_PP_NODE * (pp_nodes - 1)
    if pp >= gpus_per_node and pp_nodes == 1:
        penalty += _OVERLAP_CREDIT_PP8_INTRANODE

    # The overlap credit can make the multiplier < 1 (real is faster than the
    # straggler-free 1F1B accounting); clamp to a sane floor.
    return max(0.5, 1.0 + penalty)
