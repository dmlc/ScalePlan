"""Ground-truth predicted-MFU harness: 3dtrn vs measured MFU from the P6 runs.

Ground truth is the set of real MFU-validation runs in
``dlcalc/tests/data/*_real_mfu_P6_*node.csv`` (columns: config_name, tp, pp, ep, cp,
dp, real_mfu, ..., mlflow_run_id, run_name, status). Each row is one MLflow run's
measured MFU (%) for a (tp,pp,ep,cp,dp) parallelism config; ``dp`` is the TOTAL data-
parallel degree so that ``tp*pp*cp*dp == world`` (the run_name embeds the smaller
dense dp = dp/ep). Multiple rows may share a config (repeat runs / spikes) — we keep
them all and score each row.

Architecture per model comes from the frozen example configs (per the 2026-07-14
decision): ``example_configs/{5p3b_p6_actual,700m-config,18b_p6_actual}.yaml``,
with the parallelism (tp/pp/ep/cp/dp) and the data shape (gbs, mbs — per-row CSV
columns added 2026-07-15) overridden per CSV row, plus one fixed override:
  * 5p3b precision = bf16 (the runs trained bf16 GEMMs; the YAML's fp8 is overridden
    so the MFU denominator uses the 2250 TFLOPS bf16 B200 peak, not 4500)
Rows with ``real_mfu <= 0`` (measured value unknown) are skipped.

The model runs IN-PROCESS (import training_3d, capture stdout) and is deterministic.

Reporting, not a tight gate. ``real_mfu`` is a per-run measured MFU; the analytical
model is straggler-free and predicts a compute upper bound, so predicted > measured is
expected and the residual is unmodeled idle/framework/straggler overhead (see the
expert-GEMM / TopK / dispatch-gap investigation and Task 04). We print the per-regime
MAPE breakdown and assert only wide sanity windows that catch a mis-wired harness, not
model accuracy.
"""

from __future__ import annotations

import copy
import csv
import io
import os
import re
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXAMPLE_CFG_DIR = Path(__file__).resolve().parents[3] / "example_configs"

# CSV ground-truth files -> the example-config arch they use.
GROUND_TRUTH = [
    ("5p3b_real_mfu_P6_32node.csv", "5p3b_p6_actual.yaml"),
    ("5p3b_real_mfu_P6_64node.csv", "5p3b_p6_actual.yaml"),
    ("700m_real_mfu_P6_4node.csv", "700m-config.yaml"),
    ("18b_real_mfu_P6_64node.csv", "18b_p6_actual.yaml"),
]

# Fixed override for what the validation runs actually used (2026-07-14 decision).
FIXED_5P3B_PRECISION = "bf16"

# Sanity windows — guard the harness wiring (right configs / column / error
# formula), NOT model accuracy. Final state (2026-07-17) after the measured-B200
# physics (comm curves, grouped-GEMM, CE, CPU-dispatch floor), the correct-1F1B
# fixes (last-stage bubble amplification, tp=1 phantom-comm removal), and the
# 3-driver empirical cross-node desync term (utils/desync — the model's ONE
# fitted coefficient set: inter-node EP-A2A exposure, deep-pipeline desync,
# intra-node deep-pipeline overlap credit): per-model MAPE 700m 9.6%, 5p3b 9.0%,
# 18b 13.2% (all under the <15% goal), overall ~10.2%, mean-signed ~0%. Only the
# desync term is fitted; everything else is first-principles or measured.
_ACCEPTANCE_MAPE_RANGE = (5.0, 18.0)
# Mean-signed is ~0 now: the desync term balances residual over-prediction
# (low-parallelism configs) against under-prediction (deep/wide, mfu_max spikes).
_ACCEPTANCE_SIGNED_RANGE = (-12.0, 12.0)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _load_ground_truth() -> list[dict]:
    """One record per CSV row (FINISHED only), with the built 3dtrn config."""
    records = []
    for csv_name, arch_yaml in GROUND_TRUTH:
        csv_path = DATA_DIR / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Ground-truth CSV {csv_path} missing.")
        base = yaml.safe_load((EXAMPLE_CFG_DIR / arch_yaml).read_text())
        is_5p3b = "5p3b" in arch_yaml
        model_name = arch_yaml.split("_")[0].split("-")[0]  # 5p3b / 700m / 18b
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                # Only score FINISHED runs — RUNNING rows aren't steady-state yet.
                if row["status"].strip().upper() != "FINISHED":
                    continue
                # Skip rows whose measured MFU is unknown (recorded as <= 0).
                if float(row["real_mfu"]) <= 0:
                    continue
                tp, pp, ep, cp, dp = (
                    int(row["tp"]),
                    int(row["pp"]),
                    int(row["ep"]),
                    int(row["cp"]),
                    int(row["dp"]),
                )
                cfg = copy.deepcopy(base)
                if is_5p3b:
                    cfg["model"]["precision"] = FIXED_5P3B_PRECISION
                # Per-row data shape (columns added 2026-07-15): gbs always
                # present; mbs may be blank in older files -> keep the arch
                # YAML's microbatch_sz.
                cfg["data"]["gbs"] = int(row["gbs"])
                if row.get("mbs", "").strip():
                    cfg["data"]["microbatch_sz"] = int(row["mbs"])
                cfg["parallelism"] = {
                    "tp": tp,
                    "pp": pp,
                    "ep": ep,
                    "cp": cp,
                    "dp": dp,
                    "vpp": 1,
                    "sp": True,
                    "zero_level": 1,
                    "n_param_buckets": 8,
                }
                # Drop keys the model doesn't consume as parallelism.
                cfg.pop("search", None)
                world = tp * pp * cp * dp
                records.append(
                    {
                        "source": csv_name,
                        "model": model_name,
                        "run_id": row["mlflow_run_id"],
                        "tp": tp,
                        "pp": pp,
                        "ep": ep,
                        "cp": cp,
                        "dp": dp,
                        "world": world,
                        "measured_mfu": float(row["real_mfu"]),
                        "config": cfg,
                    }
                )
    if not records:
        raise ValueError("No FINISHED ground-truth rows loaded.")
    return records


def run_3dtrn_mfu_in_process(cfg: dict) -> float:
    """Run the 3dtrn analytical model in-process and return predicted MFU (%)."""
    import dlcalc.training_3d as training_3d

    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    cfg_path = fh.name
    buf = io.StringIO()
    saved_argv = sys.argv
    sys.argv = ["3dtrn", cfg_path]
    try:
        yaml.safe_dump(cfg, fh, sort_keys=False)
        fh.close()
        from contextlib import redirect_stdout

        with redirect_stdout(buf):
            training_3d.main()
    finally:
        sys.argv = saved_argv
        fh.close()
        os.remove(cfg_path)

    clean = _ANSI.sub("", buf.getvalue())
    match = re.search(r"Theoretical MFU:\s*([0-9]+\.[0-9]+)%", clean)
    if match is None:
        raise AssertionError(
            "Could not find 'Theoretical MFU' in 3dtrn output. Tail:\n" + clean[-800:]
        )
    return float(match.group(1))


def _mape(errs: list[float]) -> float:
    return sum(abs(e) for e in errs) / len(errs)


def _mean_signed(errs: list[float]) -> float:
    return sum(errs) / len(errs)


def _signed_err_pct(pred: float, measured: float) -> float:
    """Positive => model over-predicts MFU (predicts faster than reality)."""
    return 100.0 * (pred - measured) / measured


@pytest.fixture(scope="module")
def predictions() -> list[dict]:
    """Run the model once per ground-truth row; cache for all tests in the module."""
    records = _load_ground_truth()
    out = []
    for r in records:
        pred = run_3dtrn_mfu_in_process(r["config"])
        out.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "source",
                        "model",
                        "run_id",
                        "tp",
                        "pp",
                        "ep",
                        "cp",
                        "dp",
                        "world",
                        "measured_mfu",
                    )
                },
                "pred_mfu": pred,
                "err_pct": _signed_err_pct(pred, r["measured_mfu"]),
            }
        )
    return out


def _print_summary(records: list[dict]) -> None:
    all_errs = [r["err_pct"] for r in records]
    print("\n" + "=" * 78)
    print("GROUND-TRUTH PREDICTED-MFU ERROR  (positive % = model over-predicts MFU)")
    print("real_mfu = measured per-run MFU (P6). Model is a straggler-free upper bound.")
    print("=" * 78)
    print(f"  rows:         {len(records)}")
    print(f"  overall MAPE: {_mape(all_errs):6.2f}%")
    print(f"  mean-signed:  {_mean_signed(all_errs):+6.2f}%")

    print("\n  by model:")
    for model in sorted({r["model"] for r in records}):
        errs = [r["err_pct"] for r in records if r["model"] == model]
        print(
            f"    {model:5}  n={len(errs):2d}  "
            f"MAPE={_mape(errs):6.2f}%  mean-signed={_mean_signed(errs):+6.2f}%"
        )

    print("\n  by expert-parallel degree (EP):")
    for ep in sorted({r["ep"] for r in records}):
        errs = [r["err_pct"] for r in records if r["ep"] == ep]
        print(
            f"    EP={ep:<3d} n={len(errs):2d}  "
            f"MAPE={_mape(errs):6.2f}%  mean-signed={_mean_signed(errs):+6.2f}%"
        )

    print("\n  per-run (sorted by |error|):")
    print(f"    {'model':5} {'tp/pp/ep/cp/dp':18} {'pred':>7} {'meas':>7} {'err%':>8}")
    for r in sorted(records, key=lambda x: abs(x["err_pct"]), reverse=True):
        par = f"{r['tp']}/{r['pp']}/{r['ep']}/{r['cp']}/{r['dp']}"
        print(
            f"    {r['model']:5} {par:18} "
            f"{r['pred_mfu']:7.2f} {r['measured_mfu']:7.2f} {r['err_pct']:+8.1f}"
        )
    print("=" * 78)


@pytest.mark.slow
class TestMFUGroundTruth:
    """Predicted-MFU vs measured P6 MFU across the real validation runs."""

    def test_summary_and_acceptance(self, predictions, capsys):
        """Print the regime breakdown and sanity-check the headline stays in the
        current acceptance window. The bounds are WIDE — they assert the harness is
        correctly wired (right configs, right measured column, right error formula),
        not model accuracy."""
        with capsys.disabled():
            _print_summary(predictions)

        assert predictions, "no predictions produced"
        all_errs = [r["err_pct"] for r in predictions]
        mape = _mape(all_errs)
        signed = _mean_signed(all_errs)

        lo, hi = _ACCEPTANCE_MAPE_RANGE
        assert lo <= mape <= hi, (
            f"overall MAPE {mape:.2f}% outside acceptance window [{lo}, {hi}]. "
            f"If this is an intended cost-model change, update this window with the change."
        )
        lo, hi = _ACCEPTANCE_SIGNED_RANGE
        assert lo <= signed <= hi, (
            f"mean-signed error {signed:+.2f}% outside acceptance window [{lo}, {hi}]."
        )

    def test_per_model_mape_targets(self, predictions):
        """All three model sizes must stay under the <15% MAPE goal (achieved
        2026-07-17): measured-B200 comm/grouped-GEMM/CE curves + CPU-dispatch
        floor + correct-1F1B fixes (last-stage bubble amplification, tp=1
        phantom-comm removal) + the 3-driver empirical cross-node desync term
        (utils/desync). 700m ~9.6%, 5p3b ~9.0%, 18b ~13.2%. Regression guard."""
        by_model: dict[str, list[float]] = {}
        for r in predictions:
            by_model.setdefault(r["model"], []).append(r["err_pct"])
        mape = {m: _mape(errs) for m, errs in by_model.items()}

        for model in ("700m", "5p3b", "18b"):
            assert mape[model] < 15.0, (
                f"{model} MAPE {mape[model]:.1f}% regressed above the 15% goal."
            )

    def test_world_size_consistent(self, predictions):
        """Wiring guard: tp*pp*cp*dp must equal a plausible whole-node world (a
        multiple of 8 GPUs, no larger than the source file's node count). Catches
        a dp-convention slip (dp must be TOTAL dp, not dense dp). Note a file may
        legitimately contain rows SMALLER than its nominal node count (e.g. the
        5p3b 64-node file carries gbs2048 world-256 re-runs of 32-node shapes as
        distinct MLflow runs), so we bound, not pin."""
        max_world = {
            "5p3b_real_mfu_P6_32node.csv": 256,
            "5p3b_real_mfu_P6_64node.csv": 512,
            "700m_real_mfu_P6_4node.csv": 32,
            "18b_real_mfu_P6_64node.csv": 512,
        }
        for r in predictions:
            assert r["world"] % 8 == 0 and 8 <= r["world"] <= max_world[r["source"]], (
                f"{r['source']} row {r['tp']}/{r['pp']}/{r['ep']}/{r['cp']}/{r['dp']}: "
                f"world {r['world']} not a whole-node count <= {max_world[r['source']]} "
                f"(dp must be the TOTAL dp so tp*pp*cp*dp == world)."
            )

    def test_all_configs_predict_positive_mfu(self, predictions):
        """Every config must yield a finite positive predicted MFU (model ran)."""
        for r in predictions:
            assert r["pred_mfu"] > 0.0, (
                f"{r['model']} {r['tp']}/{r['pp']}/{r['ep']}/{r['cp']}/{r['dp']} "
                f"predicted non-positive MFU {r['pred_mfu']}"
            )
