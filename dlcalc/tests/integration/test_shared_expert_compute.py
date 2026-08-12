"""Regression test for the shared-expert compute term (Effect C).

The shared expert is a DENSE GLU MLP run on ALL tokens every MoE layer (not routed,
not divided by n_experts / EP). Two properties must hold:

1. It appears in the MoE block breakdown when ``shared_expert_inter_sz > 0`` and is
   absent otherwise.
2. Its time is EP-invariant at fixed global tokens (it is dense, not expert-parallel)
   — i.e. it scales with M = seq*mbs, NOT ÷EP. This distinguishes it from the routed
   experts and is the core of the Effect-C fix.

Runs the full 3dtrn model in-process (no subprocess).
"""

from __future__ import annotations

import copy
import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout

import pytest
import yaml

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
            "shared_expert_inter_sz": 1280,  # == expert_inter_sz (as MLflow logs it)
        },
    },
    "parallelism": {
        "tp": 1,
        "ep": 8,  # overridden per case
        "pp": 1,
        "cp": 1,
        "dp": 32,  # world = tp*pp*cp*dp = 32 fixed across EP
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


def _run(cfg: dict) -> str:
    """Run 3dtrn in-process and return its (ANSI-stripped) stdout report."""
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


def _shared_up_proj_ms(report: str) -> float | None:
    m = re.search(r"Shared Expert Up Proj\s+([0-9]+\.[0-9]+)\s*ms", report)
    return float(m.group(1)) if m else None


@pytest.mark.slow
def test_shared_expert_present_only_when_configured() -> None:
    with_shared = _run(_BASE_CONFIG)
    assert _shared_up_proj_ms(with_shared) is not None, "shared expert missing when configured"

    cfg_off = copy.deepcopy(_BASE_CONFIG)
    cfg_off["model"]["moe"]["shared_expert_inter_sz"] = 0
    without_shared = _run(cfg_off)
    assert _shared_up_proj_ms(without_shared) is None, (
        "shared expert must be absent when shared_expert_inter_sz == 0"
    )


@pytest.mark.slow
def test_shared_expert_compute_flat_in_ep() -> None:
    """Shared-expert compute scales with M = seq*mbs, NOT ÷EP.

    At a fixed world size and global token count, the dense shared-expert GEMM
    time must be identical across EP (unlike the routed experts, it is not
    expert-parallel-sharded).
    """
    times = {}
    for ep in (1, 2, 4, 8):
        cfg = copy.deepcopy(_BASE_CONFIG)
        cfg["parallelism"]["ep"] = ep
        t = _shared_up_proj_ms(_run(cfg))
        assert t is not None and t > 0, f"no shared-expert time at EP={ep}"
        times[ep] = t

    assert len(set(times.values())) == 1, (
        f"shared-expert compute must be flat in EP (dense, all tokens): {times}"
    )
