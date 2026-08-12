"""Comm-cost audit harness (Effect B) — predicted vs measured per-collective ms.

For each config, compare 3dtrn's predicted per-STEP collective wire time
(DP reduce-scatter+all-gather / all-reduce; PP send-recv; EP all-to-all) to the
measured per-collective GPU-time bucket, and REPORT per-collective divergence.

Ground truth: the measured per-collective buckets in
``profile_parse/trace_summaries/all_component_errors_enriched.csv`` (columns
``meas_allreduce_ms``, ``meas_allgather_ms``, ``meas_pp_ms``, ``meas_ep_ms``, plus
``meas_compute_only_ms`` / ``meas_nccl_only_ms``), which come from analyze_traces.py
GPU-time buckets. Architecture is reconstructed from the frozen example configs, so
this harness is restricted to the models for which a full-arch config exists:
``5p3b`` (example_configs/5p3b_p6_actual.yaml) and ``700m`` (700m-config.yaml). The
CSV's ``tp/pp/cp/dp_dense`` are read per row; ``ep`` from the model tag; ``dp`` is
formed as ``dp_dense * ep`` so ``tp*pp*cp*dp == world`` (matching the ep-in-dp mesh).

IMPORTANT — why most of this is reported, not asserted as a tight bound
=====================================================================
The measured buckets fold in straggler/desync WAIT (a collective kernel doesn't
return until the slowest rank arrives) and PP-recv folds in the pipeline bubble, and
the sum spans a variable number of steps. So they are NOT clean wire-time targets and
we do NOT tune bandwidth/protocol constants to them (that stochastic gap is Task 04).
This harness REPORTS the per-collective ratio and asserts only the
measurement-INDEPENDENT physics:
  * DP-comm participant split: any MoE config with a real DP replica group (dp>1)
    must predict non-zero DP comm (the pre-fix model zeroed it when expert_dp==1).
  * PP send-recv is zero iff pp==1 (per-stage-crossing, not per-layer).
(The unit-level guarantees live in test_dp_all_reduce.py + test_pp_send_recv.py.)
"""

from __future__ import annotations

import csv
import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CFG_DIR = REPO_ROOT / "example_configs"
ENRICHED_CSV = (
    REPO_ROOT / "profile_parse" / "trace_summaries" / "all_component_errors_enriched.csv"
)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Models with a full-arch example config (needed to run 3dtrn). The CSV `model` tag
# is "<experiment>:<size>"; we match on the size suffix.
ARCH_BY_SIZE = {
    "5p3b": "5p3b_p6_actual.yaml",
    "700m": "700m-config.yaml",
}
# 5p3b runs trained bf16 GEMMs (the YAML's fp8 would halve the MFU denominator).
FIVE_P3B_PRECISION = "bf16"


def _cf(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def _load_configs() -> list[dict]:
    """Build a 3dtrn config + measured comm buckets for every enriched-CSV row whose
    model has a full-arch example config (5p3b / 700m)."""
    if not ENRICHED_CSV.exists():
        raise FileNotFoundError(f"Enriched CSV {ENRICHED_CSV} missing.")
    recs = []
    with open(ENRICHED_CSV) as f:
        for row in csv.DictReader(f):
            size = row["model"].split(":")[-1]
            if size not in ARCH_BY_SIZE:
                continue
            tp, pp, cp = int(row["tp"]), int(row["pp"]), int(row["cp"])
            ep = int(row["ep"])
            dp_dense = int(row["dp_dense"])
            dp_total = dp_dense * ep  # ep lives in the dp region: tp*pp*cp*dp_total==world
            base = yaml.safe_load((EXAMPLE_CFG_DIR / ARCH_BY_SIZE[size]).read_text())
            if size == "5p3b":
                base["model"]["precision"] = FIVE_P3B_PRECISION
            # gbs must be divisible by dp_total; use the smallest power-of-two GBS that
            # works and gives >=1 microbatch (the comm audit compares PER-STEP wire time,
            # which is gbs-independent for DP and scales cleanly for PP/EP).
            gbs = dp_total
            while gbs < dp_total or gbs % dp_total:
                gbs *= 2
            base["data"]["gbs"] = gbs
            base["parallelism"] = {
                "tp": tp, "pp": pp, "ep": ep, "cp": cp, "dp": dp_total,
                "vpp": 1, "sp": True, "zero_level": 1, "n_param_buckets": 8,
            }
            base.pop("search", None)
            recs.append(
                {
                    "tag": f"{row['model']}_tp{tp}_pp{pp}_ep{ep}_cp{cp}_dp{dp_total}",
                    "size": size,
                    "pp": pp,
                    "dp": dp_total,
                    "world": tp * pp * cp * dp_total,
                    "config": base,
                    "meas_dp": (_cf(row, "meas_allreduce_ms") or 0.0)
                    + (_cf(row, "meas_allgather_ms") or 0.0),
                    "meas_pp": _cf(row, "meas_pp_ms") or 0.0,
                    "meas_ep": _cf(row, "meas_ep_ms") or 0.0,
                    "compute_only": _cf(row, "meas_compute_only_ms") or 0.0,
                    "nccl_only": _cf(row, "meas_nccl_only_ms") or 0.0,
                }
            )
    if not recs:
        raise ValueError("No 5p3b/700m rows found in the enriched CSV.")
    return recs


def _run_report(cfg: dict) -> str:
    import dlcalc.training_3d as training_3d

    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    cfg_path = fh.name
    buf = io.StringIO()
    saved_argv = sys.argv
    sys.argv = ["3dtrn", cfg_path]
    try:
        yaml.safe_dump(cfg, fh, sort_keys=False)
        fh.close()
        with redirect_stdout(buf):
            training_3d.main()
    finally:
        sys.argv = saved_argv
        fh.close()
        os.remove(cfg_path)
    return _ANSI.sub("", buf.getvalue())


def _grab(rx: str, report: str) -> float | None:
    m = re.search(rx, report)
    return float(m.group(1)) if m else None


def _predicted_per_collective_ms(report: str) -> dict[str, float | None]:
    dp = _grab(r"Step DP Comm:\s*([0-9.]+)\s*ms", report)
    if dp is None:
        dp = _grab(r"Combined DP Comm:\s*([0-9.]+)\s*ms", report)
    if dp is None:
        dp = _grab(r"All-Reduce Total:?\s*([0-9.]+)\s*ms", report)
    return {
        "dp": dp,
        "pp": _grab(r"Step PP Send/Recv:\s*([0-9.]+)\s*ms", report),
        "ep": _grab(r"Step EP All-to-All:\s*([0-9.]+)\s*ms", report),
    }


@pytest.mark.slow
def test_comm_cost_audit_report(capsys):
    """Report predicted-vs-measured per-collective divergence; gate only the
    measurement-independent physics invariants."""
    recs = _load_configs()
    rows = []
    for e in recs:
        report = _run_report(e["config"])
        pred = _predicted_per_collective_ms(report)
        rows.append({**e, "pred": pred})

    def ratio(p, m):
        return m / p if p else float("inf")

    with capsys.disabled():
        print("\n" + "=" * 92)
        print("COMM-COST AUDIT — predicted (straggler-FREE wire) vs measured (wait-INCLUSIVE)")
        print("per-collective ms. Target is NOISY: REPORT, not a gate. 5p3b/700m only.")
        print("=" * 92)
        print(f"  {'config':46}{'pp':>3}{'pDP':>8}{'mDP':>8}{'r':>6}"
              f"{'pPP':>7}{'mPP':>8}{'pEP':>7}{'mEP':>8}{'r':>6}")
        for r in sorted(rows, key=lambda x: (x["pp"], x["world"], x["tag"])):
            p = r["pred"]
            print(f"  {r['tag'][:46]:46}{r['pp']:>3}"
                  f"{(p['dp'] or 0):>8.1f}{r['meas_dp']:>8.1f}{ratio(p['dp'], r['meas_dp']):>6.1f}"
                  f"{(p['pp'] or 0):>7.1f}{r['meas_pp']:>8.1f}"
                  f"{(p['ep'] or 0):>7.1f}{r['meas_ep']:>8.1f}{ratio(p['ep'], r['meas_ep']):>6.1f}")
        print("=" * 92)

    # Measurement-INDEPENDENT physics gates.
    for r in rows:
        moe = r["config"]["model"].get("moe")
        # (1) MoE config with a real DP replica group predicts non-zero DP comm.
        if moe and r["dp"] > 1:
            assert r["pred"]["dp"] and r["pred"]["dp"] > 0.0, (
                f"{r['tag']}: MoE dp={r['dp']} predicts {r['pred']['dp']} ms DP comm — "
                f"dense params must reduce over dp (expert_dp<dp fix regression)."
            )
        # (2) PP send/recv is zero iff pp==1 (per-stage-crossing term).
        if r["pp"] == 1:
            assert (r["pred"]["pp"] or 0.0) == 0.0, (
                f"{r['tag']}: pp=1 must predict 0 PP send/recv, got {r['pred']['pp']}."
            )
        else:
            assert r["pred"]["pp"] and r["pred"]["pp"] > 0.0, (
                f"{r['tag']}: pp={r['pp']} must predict non-zero PP send/recv."
            )
