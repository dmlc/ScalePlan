"""PP send/recv is a per-STAGE-crossing p2p, not a per-layer op (Task 03).

A microbatch crosses a pipeline-stage boundary once per stage (one activation send
forward, one activation-grad recv back), independent of how many layers sit inside
the stage. The pre-fix model charged the activation send inside every layer's block
dict and then multiplied the block by layers_per_pp_stage, over-counting PP p2p by
that factor (e.g. 6x on an 18b pp8 stage of 6 layers).

These tests pin the corrected physics against the 3dtrn output:
  * predicted per-step PP send/recv is INDEPENDENT of layers_per_pp_stage
    (holding pp, hidden, seq, mbs, dp fixed while doubling n_layers), and
  * it equals the closed form 2 * n_microbatches_per_dp * single_send_time, and
  * it is exactly ZERO when pp == 1 (no stage boundary).
"""

from __future__ import annotations

import copy
import io
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Minimal dense config; PP crossing physics doesn't need MoE. world=tp*pp*cp*dp.
_BASE: dict = {
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
    },
    "parallelism": {
        "tp": 1,
        "ep": 1,
        "pp": 4,
        "cp": 1,
        "dp": 8,
        "vpp": 1,
        "sp": True,
        "zero_level": 1,
        "n_param_buckets": 8,
    },
    "performance": {"activation_checkpointing_type": "none"},
    "optimizer": {"optimizer_type": "adamw"},
    "data": {"gbs": 256, "seqlen": 8192, "microbatch_sz": 1},
    "hardware": {"node_type": "p6-b200.48xlarge"},
}


def _report(cfg: dict) -> str:
    import dlcalc.training_3d as training_3d

    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    path = fh.name
    buf = io.StringIO()
    saved_argv = sys.argv
    sys.argv = ["3dtrn", path]
    try:
        yaml.safe_dump(cfg, fh, sort_keys=False)
        fh.close()
        with redirect_stdout(buf):
            training_3d.main()
    finally:
        sys.argv = saved_argv
        fh.close()
        Path(path).unlink(missing_ok=True)
    return _ANSI.sub("", buf.getvalue())


def _step_pp_ms(report: str) -> float:
    m = re.search(r"Step PP Send/Recv:\s*([0-9.]+)\s*ms", report)
    assert m, "3dtrn did not print 'Step PP Send/Recv' in the audit summary"
    return float(m.group(1))


def _single_send_ms(report: str) -> float:
    m = re.search(r"PP Send/Recv Time:\s*([0-9.]+)\s*ms", report)
    assert m, "3dtrn did not print single-crossing 'PP Send/Recv Time'"
    return float(m.group(1))


@pytest.mark.slow
def test_pp_send_recv_independent_of_layers_per_stage() -> None:
    """Doubling n_layers (same pp) must NOT change per-step PP send/recv.

    Per-step PP p2p = 2 * n_microbatches_per_dp * single_send — it depends on the
    number of microbatches and the boundary-activation size, never on how many
    layers live in a stage. The pre-fix per-layer model would DOUBLE this when
    n_layers doubles; the corrected per-stage model holds it flat.
    """
    cfg16 = copy.deepcopy(_BASE)
    cfg16["model"]["n_layers"] = 16  # 4 layers/stage at pp=4
    cfg32 = copy.deepcopy(_BASE)
    cfg32["model"]["n_layers"] = 32  # 8 layers/stage at pp=4

    pp16 = _step_pp_ms(_report(cfg16))
    pp32 = _step_pp_ms(_report(cfg32))

    assert pp16 > 0.0
    assert abs(pp16 - pp32) < 1e-6, (
        f"per-step PP send/recv changed with layers_per_pp_stage "
        f"({pp16:.4f} ms @16L vs {pp32:.4f} ms @32L) — PP p2p is being charged "
        f"per layer again instead of per stage-crossing."
    )


@pytest.mark.slow
def test_pp_send_recv_matches_closed_form() -> None:
    """Per-step PP send/recv equals 2 * n_microbatches_per_dp * single_send."""
    cfg = copy.deepcopy(_BASE)
    report = _report(cfg)
    n_microbatches_per_dp = cfg["data"]["gbs"] // (
        cfg["parallelism"]["dp"] * cfg["data"]["microbatch_sz"]
    )
    expected = 2 * n_microbatches_per_dp * _single_send_ms(report)
    # Both terms are parsed from 3-decimal printed ms, so the single-send value
    # carries up to ~5e-4 ms of rounding that the 2*n_microbatches multiply
    # amplifies; tolerate that display-precision slack rather than a real gap.
    tol = 0.5e-3 * 2 * n_microbatches_per_dp + 1e-3
    assert abs(_step_pp_ms(report) - expected) < tol, (
        f"per-step PP send/recv {_step_pp_ms(report):.4f} ms != closed form "
        f"{expected:.4f} ms (2 * {n_microbatches_per_dp} microbatches * single send)."
    )


@pytest.mark.slow
def test_pp_send_recv_zero_when_pp1() -> None:
    """pp=1 has no pipeline-stage boundary, so per-step PP send/recv is exactly 0."""
    cfg = copy.deepcopy(_BASE)
    cfg["parallelism"]["pp"] = 1
    cfg["parallelism"]["dp"] = 32  # keep world = tp*pp*cp*dp constant (32)
    assert _step_pp_ms(_report(cfg)) == 0.0


@pytest.mark.slow
def test_pp_send_recv_scales_linearly_with_vpp() -> None:
    """Interleaved 1F1B (vpp>1) multiplies PP p2p by vpp.

    Each rank owns vpp virtual pipeline chunks, so a microbatch enters/leaves the
    rank vpp times per step — vpp× the crossings of non-interleaved 1F1B. Doubling
    vpp (n_layers fixed so vpp divides layers) must therefore DOUBLE per-step PP
    send/recv. (The flip side is the 1/vpp steady-state bubble reduction, modeled
    separately.) This guards that PP p2p is not silently treated as vpp-invariant.
    """
    cfg1 = copy.deepcopy(_BASE)
    cfg1["model"]["n_layers"] = 16  # pp=4 -> 4 layers/stage; vpp must divide layers/stage
    cfg1["parallelism"]["vpp"] = 1
    cfg2 = copy.deepcopy(cfg1)
    cfg2["parallelism"]["vpp"] = 2

    pp_vpp1 = _step_pp_ms(_report(cfg1))
    pp_vpp2 = _step_pp_ms(_report(cfg2))

    assert pp_vpp1 > 0.0
    assert abs(pp_vpp2 - 2 * pp_vpp1) < 1e-3, (
        f"per-step PP send/recv must scale with vpp: vpp=1 gave {pp_vpp1:.4f} ms, "
        f"vpp=2 gave {pp_vpp2:.4f} ms (expected ~{2 * pp_vpp1:.4f} ms). Interleaving "
        f"multiplies p2p crossings by vpp."
    )
