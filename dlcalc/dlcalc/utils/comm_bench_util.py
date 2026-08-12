"""Measured NCCL collective bandwidth lookups (B200 4-node pod, 2026-07-17).

Sources (benchmarks/, all run inside the training image on a 4-node p6-b200
pod, torch 2.7 / NCCL 2.27.5 / EFA):
  * results/a2a_b200_4node.parquet      — all_to_all_single per-GPU egress BW
        vs (group_size in {2,4,8,16,32}, buffer 1MB..1.2GB). Groups <=8 are
        intra-node NVSwitch; 16/32 span 2/4 nodes.
  * results/dp_collectives_b200_4node.parquet — all_reduce / reduce_scatter /
        all_gather ring bus BW vs (group_size in {8,16,32}, payload 4MB..2GB),
        CONTIGUOUS rank groups (dense-DP topology: NVLink inside the node, the
        node-boundary crossing striped over all 8 EFA rails by NCCL channels).
  * results/dp_collectives_b200_stride8.parquet — same collectives on
        STRIDE-8 groups (expert-DP topology with ep=8: one rank per node,
        every hop crosses EFA on the GPU's own NIC rail; plateau ~47 GB/s).

Why lookups instead of the analytic link-peak * protocol-efficiency model:
the measured curves differ from the analytic constants by up to 8x in BOTH
directions (dense-DP ring busbw ~350 GB/s vs the 45 GB/s the peak/8 model
charged; 8-rank A2A ~550 GB/s vs 900 charged; A2A at 16MB ~170 GB/s), and the
size-dependence (NCCL channel/protocol ramp) is not derivable from specs.
Physics basis: measured on the exact fabric/NCCL build the benchmark jobs use
(GUIDELINES §1, §2).

Interpolation policy (GUIDELINES §2):
  * payload: log-linear interpolation within the measured size range; clamped
    to the endpoint value outside it (below the smallest size the latency term
    of the caller dominates; above the largest the curve is a plateau).
  * group size: exact match if measured; otherwise the nearest measured group
    size ABOVE is used for rings (ring per-link busbw is n-invariant once the
    pipeline is full — measured n=16 vs n=32 differ <2%), and A2A groups are
    only ever 2/4/8 (intra) or 16/32 (inter) in our configs — all measured.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import pandas as pd  # type: ignore[import-untyped]

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks", "results")

_A2A_PARQUET = os.path.join(_RESULTS_DIR, "a2a_b200_4node.parquet")
_DP_CONTIG_PARQUET = os.path.join(_RESULTS_DIR, "dp_collectives_b200_4node.parquet")
_DP_STRIDED_PARQUET = os.path.join(_RESULTS_DIR, "dp_collectives_b200_stride8.parquet")

# Devices the measured tables cover. Other SKUs fall back to the analytic model.
_MEASURED_DEVICE_PREFIX = "p6-b200"


def measured_tables_cover(machine_name: str) -> bool:
    """True if the measured-B200 collective tables apply to this machine."""
    return machine_name.startswith(_MEASURED_DEVICE_PREFIX) and os.path.exists(_A2A_PARQUET)


@lru_cache(maxsize=1)
def _load_a2a() -> pd.DataFrame:  # type: ignore[no-any-unimported]
    df = pd.read_parquet(_A2A_PARQUET)
    return df[df["status"] == "OK"]


@lru_cache(maxsize=2)
def _load_dp(strided: bool) -> pd.DataFrame:  # type: ignore[no-any-unimported]
    path = _DP_STRIDED_PARQUET if strided else _DP_CONTIG_PARQUET
    df = pd.read_parquet(path)
    return df[df["status"] == "OK"]


def _interp_log_size(curve: list[tuple[float, float]], x_bytes: float) -> float:
    """Log-linear interpolation of (bytes, bw) points; clamps outside range."""
    pts = sorted(curve)
    if x_bytes <= pts[0][0]:
        return pts[0][1]
    if x_bytes >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x_bytes <= x1:
            f = (math.log(x_bytes) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)
    raise AssertionError("unreachable")


def measured_a2a_egress_bw_bytes_per_s(group_size: int, buffer_bytes: int) -> float:
    """Measured per-GPU egress bandwidth for all_to_all_single.

    `buffer_bytes` is the FULL local dispatch buffer (the model's `size`);
    egress per rank = buffer * (n-1)/n. Group sizes are matched exactly
    (2/4/8/16/32 measured); unmeasured sizes raise — every EP degree in the
    validation set is measured.
    """
    df = _load_a2a()
    sub = df[df["group_size"] == group_size]
    if sub.empty:
        raise KeyError(
            f"A2A group_size={group_size} not in measured table "
            f"({sorted(df['group_size'].unique())}); re-run benchmarks/a2a_benchmark.py"
        )
    curve = list(zip(sub["buffer_bytes"].astype(float), sub["egress_gbps_p50"] * 1e9))
    return _interp_log_size(curve, float(buffer_bytes))


def measured_ring_busbw_bytes_per_s(
    collective: str,  # "all_reduce" | "reduce_scatter" | "all_gather"
    group_size: int,
    payload_bytes: int,
    strided: bool,
) -> float:
    """Measured ring bus bandwidth for a DP collective.

    time = latency_term + bus_factor(collective, n) * payload / busbw,
    where bus_factor is (n-1)/n for RS/AG and 2(n-1)/n for AR — the caller
    applies it; this returns busbw only.

    Group-size handling: exact match when measured; else the largest measured
    group (ring per-link busbw is n-invariant once full: n=16 vs n=32 measured
    within 2%). Strided tables cover n in {2,4}; contiguous {8,16,32}.
    """
    df = _load_dp(strided)
    sub = df[df["collective"] == collective]
    if sub.empty:
        raise KeyError(f"collective {collective!r} not in measured DP table")
    sizes = sorted(sub["group_size"].unique())
    if group_size in sizes:
        n_key = group_size
    else:
        larger = [s for s in sizes if s >= group_size]
        n_key = larger[0] if larger else sizes[-1]
    sub = sub[sub["group_size"] == n_key]
    curve = list(zip(sub["payload_bytes"].astype(float), sub["busbw_gbps_p50"] * 1e9))
    return _interp_log_size(curve, float(payload_bytes))
