"""Regression test: the parallelism-search MoE cost model must match 3dtrn.

``parallelism_search/training_calculator.py`` used to RE-IMPLEMENT the 3dtrn cost
model for the grid search, and its copy drifted: it carried its own expert-GEMM
forward closure with a 1/EP over-credit (Effect A). Because the search's whole job
is to CHOOSE the expert-parallel degree, a 1/EP bias in its expert compute skews
the rankings toward high EP for the wrong reason.

The duplicate is gone -- ``training_calculator`` is now a thin adapter over
``dlcalc.training_3d.calculate_training_metrics`` -- so this parity holds by
construction. The test stays as the guard that it KEEPS holding: it fails again
the moment someone re-forks the physics into the search path.
"""

from __future__ import annotations

import copy
import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

# The search package imports as ``parallelism_search`` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# A clean MoE config; world = tp*pp*cp*dp = 32 held fixed while EP varies.
_BASE_CONFIG: dict = {
    "model": {
        "n_layers": 16,
        "hidden_sz": 2048,
        "inter_sz": 5120,
        "n_q_heads": 16,
        "n_kv_heads": 8,
        "head_dim": 128,
        "vocab_sz": 257152,
        "precision": "bf16",
        "sdpa_precision": "bf16",
        "glu": True,
        "rotary_embeds": True,
        "dropout": False,
        "tie_embeddings": False,
        "moe": {
            "n_experts": 128,
            "experts_per_token": 3,
            "capacity_factor": 1,
            "expert_inter_sz": 1280,
            "moe_frequency": 1.0,
            "expert_tp_degree": 1,
            # Exercise the shared-expert path too, so this parity check guards that
            # BOTH cost models model Effect C the same way (not just the routed GEMM).
            "shared_expert_inter_sz": 1280,
        },
    },
    "parallelism": {
        "tp": 1,
        "ep": 8,  # overridden per case
        "pp": 1,
        "cp": 1,
        "dp": 32,
        "vpp": 1,
        "sp": True,
        "zero_level": 1,
        "n_param_buckets": 8,
    },
    "performance": {"activation_checkpointing_type": "none"},
    "optimizer": {"optimizer_type": "muon"},
    "data": {"gbs": 256, "seqlen": 8192, "microbatch_sz": 1},
    "hardware": {"node_type": "p6-b200.48xlarge"},
}

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _t3dtrn_mfu(cfg: dict) -> float:
    """Predicted MFU (%) from the canonical 3dtrn model, run in-process."""
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
    clean = _ANSI.sub("", buf.getvalue())
    return float(re.search(r"Theoretical MFU:\s*([0-9]+\.[0-9]+)%", clean).group(1))


@pytest.mark.slow
def test_search_mfu_matches_3dtrn_across_ep() -> None:
    """Search-predicted MFU must equal 3dtrn-predicted MFU on the same MoE config.

    Both implement the same model; the search only duplicates it for the grid.
    If the search's expert-GEMM forward diverges from 3dtrn's (the pre-fix 1/EP
    copy), the two disagree and the disagreement grows with EP. Post-fix they
    agree to well under 0.1pp across EP -- the invariant that guards against the
    duplicate closure silently drifting again.
    """
    pytest.importorskip("tqdm", reason="parallelism_search imports tqdm at package init")
    from parallelism_search.training_calculator import calculate_training_metrics

    for ep in (1, 2, 4, 8):
        cfg = copy.deepcopy(_BASE_CONFIG)
        cfg["parallelism"]["ep"] = ep
        search_mfu, _iter_s, _mem = calculate_training_metrics(cfg)
        canonical_mfu = _t3dtrn_mfu(cfg)
        assert abs(search_mfu - canonical_mfu) < 0.05, (
            f"parallelism-search MFU {search_mfu:.3f}% disagrees with 3dtrn "
            f"{canonical_mfu:.3f}% at EP={ep}. The search's expert-GEMM cost model "
            f"has drifted from dlcalc.training_3d (see Effect A / model A1)."
        )
