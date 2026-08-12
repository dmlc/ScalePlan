"""Web server exposing dlcalc's training calculator and parallelism search.

Run:
    uv run python webapp/server.py
"""

from __future__ import annotations

import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dlcalc.utils.hardware import DType, MachineSpec  # noqa: E402
from parallelism_search.searcher import ParallelismSearcher  # noqa: E402
from parallelism_search.training_calculator import calculate_training_metrics  # noqa: E402

EXAMPLE_DIR = REPO_ROOT / "example_configs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

NODE_TYPES = [
    "p4d.24xlarge",
    "p4de.24xlarge",
    "p5.48xlarge",
    "p6-b200.48xlarge",
    "trn1n.32xlarge",
]

PRECISIONS = ["fp4", "fp8", "fp8_e4m3", "fp16", "bf16", "fp32"]
ACT_CKPT_TYPES = ["none", "selective", "full"]
OPTIMIZERS = ["adam", "muon"]

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


def divisors(n: int, max_v: int | None = None) -> list[int]:
    out = []
    for i in range(1, n + 1):
        if n % i == 0:
            if max_v is None or i <= max_v:
                out.append(i)
    return out


def hardware_info(node_type: str) -> dict:
    spec = MachineSpec.from_str(node_type)
    return {
        "node_type": node_type,
        "n_devices_per_node": spec.n_devices,
        "device_memory_gb": spec.device_spec.mem_capacity_bytes / (1024**3),
        "supported_dtypes": [d.value for d in spec.device_spec.dtype_to_peak_flops.keys()],
        "peak_flops": {
            d.value: spec.device_spec.dtype_to_peak_flops[d]
            for d in spec.device_spec.dtype_to_peak_flops.keys()
        },
        "intra_node_bw_gbps": spec.intra_node_connect.unidirectional_bw_bytes_per_sec / 1e9,
        "inter_node_bw_gbps": spec.inter_node_connect.unidirectional_bw_bytes_per_sec / 1e9,
    }


def build_full_config(payload: dict) -> dict:
    """Build the complete cfg dict expected by calculate_training_metrics."""
    model = deepcopy(payload["model"])
    data = deepcopy(payload["data"])
    hardware = deepcopy(payload["hardware"])
    performance = {"activation_checkpointing_type": payload["performance"]["activation_checkpointing_type"]}
    optimizer = payload.get("optimizer", {"optimizer_type": "adam"})

    parallelism = deepcopy(payload["parallelism"])
    parallelism.setdefault("vpp", 1)
    parallelism.setdefault("sp", True)
    parallelism.setdefault("zero_level", 1)
    parallelism.setdefault("n_param_buckets", 5)

    if "moe" in model:
        model["moe"]["expert_tp_degree"] = parallelism.get(
            "etp", model["moe"].get("expert_tp_degree", 1)
        )

    cfg = {
        "model": model,
        "data": data,
        "hardware": hardware,
        "parallelism": parallelism,
        "performance": performance,
        "optimizer": optimizer,
    }
    return cfg


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/meta")
def api_meta():
    """Return static reference data: configs, hardware, precisions, etc."""
    config_files = sorted([p.name for p in EXAMPLE_DIR.glob("*.yaml")])
    hardware = {nt: hardware_info(nt) for nt in NODE_TYPES}
    return jsonify(
        {
            "configs": config_files,
            "hardware": hardware,
            "node_types": NODE_TYPES,
            "precisions": PRECISIONS,
            "act_ckpt_types": ACT_CKPT_TYPES,
            "optimizers": OPTIMIZERS,
        }
    )


@app.route("/api/config/<name>")
def api_config(name: str):
    """Load a named example config file."""
    safe_name = os.path.basename(name)
    path = EXAMPLE_DIR / safe_name
    if not path.is_file():
        return jsonify({"error": f"Config not found: {safe_name}"}), 404
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return jsonify(cfg)


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Evaluate a single (model, data, hardware, parallelism) config."""
    payload = request.get_json(force=True)
    try:
        cfg = build_full_config(payload)
        mfu, iter_time_s, mem_per_device_gb = calculate_training_metrics(cfg)
        spec = MachineSpec.from_str(cfg["hardware"]["node_type"])
        device_mem_gb = spec.device_spec.mem_capacity_bytes / (1024**3)
        gbs = cfg["data"]["gbs"]
        seqlen = cfg["data"]["seqlen"]
        throughput_tokens_per_sec = gbs * seqlen / iter_time_s if iter_time_s > 0 else 0.0
        return jsonify(
            {
                "ok": True,
                "mfu": mfu,
                "iteration_time_s": iter_time_s,
                "memory_per_device_gb": mem_per_device_gb,
                "device_memory_gb": device_mem_gb,
                "memory_fraction": mem_per_device_gb / device_mem_gb if device_mem_gb else 0.0,
                "throughput_tokens_per_sec": throughput_tokens_per_sec,
                "tokens_per_day": throughput_tokens_per_sec * 86400,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc(limit=3),
            }
        )


@app.route("/api/parallelism-options", methods=["POST"])
def api_parallelism_options():
    """Given num_devices and n_layers, return valid divisors for tp,pp,cp slots."""
    payload = request.get_json(force=True)
    num_devices = int(payload["num_devices"])
    n_layers = int(payload.get("n_layers", 1))
    n_experts = int(payload.get("n_experts", 1))

    tp_options = divisors(num_devices)
    pp_options = [p for p in divisors(num_devices) if n_layers % p == 0]
    cp_options = divisors(num_devices)
    ep_options = divisors(n_experts) if n_experts > 1 else [1]

    return jsonify(
        {
            "tp": tp_options,
            "pp": pp_options,
            "cp": cp_options,
            "ep": ep_options,
        }
    )


@app.route("/api/search", methods=["POST"])
def api_search():
    """Run a parallelism grid search and return top-K configs.

    Body:
      cfg: full base config (model/data/hardware/performance/optimizer)
      num_devices: int
      top_k: int
      max_ep, max_tp, max_cp, max_etp, max_mbs: int
      memory_limit_fraction: float
      activation_checkpointing: "none"|"selective"|"full"|"all"
    """
    payload = request.get_json(force=True)
    base_cfg = payload["cfg"]
    num_devices = int(payload.get("num_devices", base_cfg.get("search", {}).get("num_devices", 64)))

    tmp = REPO_ROOT / "webapp" / "_search_tmp.yaml"
    cfg_to_dump = deepcopy(base_cfg)
    cfg_to_dump.setdefault("search", {})["num_devices"] = num_devices
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg_to_dump, f)

    try:
        searcher = ParallelismSearcher(
            str(tmp),
            num_devices=num_devices,
            mlflow_experiment_id=None,
            max_ep=int(payload.get("max_ep", 32)),
            max_tp=payload.get("max_tp"),
            max_cp=payload.get("max_cp"),
            max_etp=payload.get("max_etp"),
            memory_limit_fraction=float(payload.get("memory_limit_fraction", 0.9)),
            activation_checkpointing=payload.get("activation_checkpointing", "all"),
            max_mbs=int(payload.get("max_mbs", 1)),
        )
        top_k = int(payload.get("top_k", 10))
        results = searcher.search(top_k=top_k)

        spec = MachineSpec.from_str(base_cfg["hardware"]["node_type"])
        device_mem_gb = spec.device_spec.mem_capacity_bytes / (1024**3)

        gbs = base_cfg["data"]["gbs"]
        seqlen = base_cfg["data"]["seqlen"]

        rows = []
        for r in results:
            tp_per_sec = gbs * seqlen / r.iteration_time_s if r.iteration_time_s > 0 else 0.0
            rows.append(
                {
                    "config": r.config,
                    "mfu": r.mfu,
                    "memory_per_device_gb": r.memory_per_device_gb,
                    "iteration_time_s": r.iteration_time_s,
                    "throughput_tokens_per_sec": tp_per_sec,
                    "memory_fraction": r.memory_per_device_gb / device_mem_gb,
                }
            )

        valid_count = sum(1 for r in searcher.results if r.valid)
        return jsonify(
            {
                "ok": True,
                "results": rows,
                "total_evaluated": len(searcher.results),
                "valid_count": valid_count,
                "device_memory_gb": device_mem_gb,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc(limit=5),
            }
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5057"))
    print(f"\n  Performance Modeling Studio — http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
