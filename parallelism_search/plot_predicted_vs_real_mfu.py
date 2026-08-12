#!/usr/bin/env python3
"""Predicted (3dtrn analytical model) vs real (measured P6) MFU, in absolute MFU %.

Same figures/style as the normalized set in ``figures/``, but the y axis carries the
raw MFU percentage instead of a normalized value.

Ground truth: ``dlcalc/tests/data/*_real_mfu_P6_*node.csv`` (one row per MLflow run).
Predictions: the 3dtrn model run in-process, using the frozen ``example_configs``
architectures with the per-row parallelism / data shape overridden -- the same wiring
as ``dlcalc/tests/integration/test_mfu_golden.py`` (5p3b precision forced to bf16,
since those runs trained bf16 GEMMs).

Usage (needs numpy/pandas/pyyaml/tqdm/matplotlib and dlcalc importable):
    python parallelism_search/plot_predicted_vs_real_mfu.py [--outdir figures]
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dlcalc" / "tests" / "data"
EXAMPLE_CFG_DIR = REPO_ROOT / "example_configs"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# One entry per figure: which ground-truth rows it covers and how it's titled.
# Mirrors the existing normalized figures 1:1 (same rows, same order, same names).
FIGURES = [
    {
        "outfile": "700m_p6-b200_32dev_gbs256",
        "csv": "700m_real_mfu_P6_4node.csv",
        "arch_yaml": "700m-config.yaml",
        "gbs": 256,
        "devices": 32,
        "size_label": r"0.7B $N_\mathrm{act}$,  16B $N_\mathrm{tot}$",
    },
    {
        "outfile": "5p3b_p6-b200_256dev_gbs2048",
        "csv": "5p3b_real_mfu_P6_32node.csv",
        "arch_yaml": "5p3b_p6_actual.yaml",
        "gbs": 2048,
        "devices": 256,
        "size_label": r"5.4B $N_\mathrm{act}$,  127B $N_\mathrm{tot}$",
    },
    {
        "outfile": "5p3b_p6-b200_512dev_gbs2048",
        "csv": "5p3b_real_mfu_P6_64node.csv",
        "arch_yaml": "5p3b_p6_actual.yaml",
        "gbs": 2048,
        "devices": 512,
        "size_label": r"5.4B $N_\mathrm{act}$,  127B $N_\mathrm{tot}$",
    },
    {
        "outfile": "18b_p6-b200_512dev_gbs512",
        "csv": "18b_real_mfu_P6_64node.csv",
        "arch_yaml": "18b_p6_actual.yaml",
        "gbs": 512,
        "devices": 512,
        "size_label": r"18B $N_\mathrm{act}$,  434B $N_\mathrm{tot}$",
    },
]


def load_rows(csv_name: str, arch_yaml: str, gbs: int) -> list[dict]:
    """Rows of one ground-truth CSV at one global batch size, with a built 3dtrn config."""
    base = yaml.safe_load((EXAMPLE_CFG_DIR / arch_yaml).read_text())
    is_5p3b = "5p3b" in arch_yaml
    rows = []
    with open(DATA_DIR / csv_name) as f:
        for row in csv.DictReader(f):
            if int(row["gbs"]) != gbs or float(row["real_mfu"]) <= 0:
                continue
            tp, pp, ep, cp, dp = (int(row[k]) for k in ("tp", "pp", "ep", "cp", "dp"))
            mbs = int(row["mbs"]) if row.get("mbs", "").strip() else base["data"]["microbatch_sz"]
            cfg = copy.deepcopy(base)
            if is_5p3b:
                # The validation runs trained bf16 GEMMs (the YAML's fp8 would put a
                # 4500 TFLOPS peak in the MFU denominator instead of 2250).
                cfg["model"]["precision"] = "bf16"
            cfg["data"]["gbs"] = gbs
            cfg["data"]["microbatch_sz"] = mbs
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
            cfg.pop("search", None)
            rows.append(
                {
                    "label": f"{pp}/{ep}/{dp}/{mbs}",
                    "real_mfu": float(row["real_mfu"]),
                    "config": cfg,
                }
            )
    if not rows:
        raise ValueError(f"no rows in {csv_name} at gbs={gbs}")
    return rows


def predict_mfu(cfg: dict) -> float:
    """Run the 3dtrn analytical model in-process; return predicted MFU (%)."""
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

    match = re.search(r"Theoretical MFU:\s*([0-9]+\.[0-9]+)%", _ANSI.sub("", buf.getvalue()))
    if match is None:
        raise AssertionError("no 'Theoretical MFU' in 3dtrn output:\n" + buf.getvalue()[-800:])
    return float(match.group(1))


def draw(spec: dict, rows: list[dict], outdir: Path, suffix: str, normalize: bool) -> Path:
    """Grouped predicted/real bar chart, in absolute MFU % or normalized.

    ``normalize`` divides BOTH series by one per-figure denominator: the largest
    value across all bars in that figure, max(max predicted, max real). The tallest
    bar lands at 1.0 and the shape of the sweep is preserved.
    """
    labels = [r["label"] for r in rows]
    predicted = [r["pred_mfu"] for r in rows]
    real = [r["real_mfu"] for r in rows]

    denom = max(max(predicted), max(real)) if normalize else 1.0
    predicted = [v / denom for v in predicted]
    real = [v / denom for v in real]

    # Same look as the normalized set: greyscale bars, heavy fonts, y-only grid.
    plt.rcParams.update({"font.size": 18, "axes.linewidth": 2.0})
    fig, ax = plt.subplots(figsize=(10.0, 6.2))

    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, predicted, w, label="Predicted MFU", color="lightgray", edgecolor="black")
    ax.bar(x + w / 2, real, w, label="Real MFU", color="black", edgecolor="black")

    ax.set_title(
        f"MFU Comparison: Predicted vs Real  ({spec['size_label']})\n"
        f"({spec['devices']} devices, Global Batch Size={spec['gbs']})",
        fontsize=21,
    )
    ax.set_xlabel("Parallelism Configuration (pp/ep/dp/mbs)", fontsize=25)
    ax.set_ylabel("Normalized MFU" if normalize else "MFU (%)", fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylim(0, max(max(predicted), max(real)) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=22)

    fig.tight_layout()
    outpath = outdir / f"{spec['outfile']}{suffix}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(REPO_ROOT / "figures"))
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="divide both series by the per-figure max(predicted, real) instead of "
        "plotting absolute MFU %%",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="appended to the figure name; defaults to _normalized / _unnormalized",
    )
    args = parser.parse_args()
    suffix = args.suffix if args.suffix is not None else (
        "_normalized" if args.normalize else "_unnormalized"
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for spec in FIGURES:
        rows = load_rows(spec["csv"], spec["arch_yaml"], spec["gbs"])
        for r in rows:
            r["pred_mfu"] = predict_mfu(r["config"])
        # Same ordering as the normalized figures: descending measured MFU.
        rows.sort(key=lambda r: r["real_mfu"], reverse=True)
        outpath = draw(spec, rows, outdir, suffix, args.normalize)

        denom = max(max(r["pred_mfu"] for r in rows), max(r["real_mfu"] for r in rows))
        print(f"\n{spec['outfile']}  ->  {outpath}")
        if args.normalize:
            print(f"  normalization denominator = max(pred, real) = {denom:.2f}% MFU")
        print(
            f"  {'pp/ep/dp/mbs':>16} {'pred %':>8} {'real %':>8} {'err %':>8}"
            + (f" {'pred_n':>7} {'real_n':>7}" if args.normalize else "")
        )
        for r in rows:
            err = 100.0 * (r["pred_mfu"] - r["real_mfu"]) / r["real_mfu"]
            line = f"  {r['label']:>16} {r['pred_mfu']:8.2f} {r['real_mfu']:8.2f} {err:+8.1f}"
            if args.normalize:
                line += f" {r['pred_mfu'] / denom:7.3f} {r['real_mfu'] / denom:7.3f}"
            print(line)
        mape = sum(
            abs(100.0 * (r["pred_mfu"] - r["real_mfu"]) / r["real_mfu"]) for r in rows
        ) / len(rows)
        print(f"  MAPE: {mape:.2f}%  (n={len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
