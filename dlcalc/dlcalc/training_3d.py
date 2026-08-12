"""CLI tool for estimating performance characteristics of 3D parallel training.

This module is the SINGLE source of truth for the cost model. Consumers that want
the numbers rather than the report (the parallelism search, the webapp) call
:func:`calculate_training_metrics`, which runs the same code path as the CLI with
the report suppressed. Do not re-implement the physics elsewhere -- a second copy
silently drifts (see the Effect A/B/C fixes and the parity test that caught them).
"""

import io
import json
import math
from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dlcalc.utils.backward import (
    compute_glu_bwd_time_s,
    compute_grouped_linear_bwd_time_s,
    compute_layernorm_bwd_time_s,
    compute_linear_bwd_time_s,
    compute_permutation_bwd_time_s,
    compute_residual_bwd_time_s,
    compute_rope_bwd_time_s,
    compute_sdpa_bwd_time_s,
    compute_topk_bwd_time_s,
)
from dlcalc.utils.comms import (
    get_all_to_all_comm_time_s,
    get_cross_dc_dp_all_gather_comm_time_s,
    get_cross_dc_dp_all_reduce_comm_time_s,
    get_cross_dc_dp_reduce_scatter_comm_time_s,
    get_dp_all_gather_comm_time_s,
    get_dp_all_reduce_comm_time_s,
    get_dp_reduce_scatter_comm_time_s,
    get_expert_tp_all_gather_comm_time_s,
    get_expert_tp_reduce_scatter_comm_time_s,
    get_tp_all_gather_comm_time_s,
    get_tp_reduce_scatter_comm_time_s,
)
from dlcalc.utils.compute import compute_gemm_flops
from dlcalc.utils.configurations import ActivationCheckpointingType, CrossDCConfig
from dlcalc.utils.data import Size, TensorRepr
from dlcalc.utils.desync import cross_node_desync_multiplier
from dlcalc.utils.gemm_util import get_gemm_utilization_or_default
from dlcalc.utils.grouped_gemm_util import (
    grouped_gemm_measured,
    grouped_mlp_down_bwd_time_s,
    grouped_mlp_down_fwd_time_s,
    grouped_mlp_up_bwd_time_s,
    grouped_mlp_up_fwd_time_s,
)
from dlcalc.utils.hardware import DType, MachineSpec
from dlcalc.utils.kernel_launch import (
    calculate_kernel_launch_overhead,
    dispatch_time_per_microbatch_s,
    estimate_kernel_count_per_transformer_block,
)
from dlcalc.utils.math import safe_divide
from dlcalc.utils.model_3d import MoeCfg, ParallelConfig, ThreeDParallelModel
from dlcalc.utils.moe_router_util import calculate_permutation_time, calculate_topk_time
from dlcalc.utils.norm_rope_util import get_norm_time, get_rope_time
from dlcalc.utils.overlap import calculate_exposed_dp_time
from dlcalc.utils.pipeline import (
    compute_pipeline_bubble_fraction,
    compute_stage_imbalance_extra_time_s,
)
from dlcalc.utils.printing import (
    _BOLD,
    _END,
    _GRAY,
    format_number,
    get_color_by_percentage,
    get_color_by_time_ms,
    get_color_for_component_percentage,
    print_h1_header,
    print_h2_header,
    print_info,
    print_kv,
    print_metric,
    print_section_separator,
    print_success,
)
from dlcalc.utils.vocab_ce_util import cross_entropy_fwd_bwd_time_s, vocab_ce_measured


@dataclass(frozen=True)
class TrainingMetrics:
    """Headline results of one cost-model evaluation."""

    mfu_pct: float
    iteration_time_s: float
    memory_per_device_gb: float


def calculate_training_metrics(cfg: dict[str, Any], *, verbose: bool = False) -> TrainingMetrics:
    """Evaluate the cost model on a parsed config dict.

    Same code path as the ``3dtrn`` CLI; ``verbose=False`` just discards the report.
    This is the entry point for programmatic consumers (parallelism search, webapp).
    """
    if verbose:
        return _calculate_and_report(cfg)
    with redirect_stdout(io.StringIO()):
        return _calculate_and_report(cfg)


def _calculate_and_report(cfg: dict[str, Any]) -> TrainingMetrics:
    print_h1_header("CONFIGURATION")
    print(json.dumps(cfg, indent=2))

    # Load SDPA performance data
    sdpa_parquet_path = (
        Path(__file__).parent.parent.parent / "benchmarks" / "results" / "sdpa.parquet"
    )
    try:
        sdpa_df = pd.read_parquet(sdpa_parquet_path)
        print_info(f"Loaded SDPA timing data from {sdpa_parquet_path}")
    except FileNotFoundError:
        print_info(f"Warning: SDPA timing data not found at {sdpa_parquet_path}, will use calculated values")
        sdpa_df = None

    sequence_len = cfg["data"]["seqlen"]
    microbatch_sz = cfg["data"]["microbatch_sz"]
    hidden_sz = cfg["model"]["hidden_sz"]

    # Setup expert parallelism configuration if MoE is enabled
    expert_mesh = None
    if "moe" in cfg["model"]:
        ep = cfg["parallelism"]["ep"]
        expert_tp = cfg["model"]["moe"]["expert_tp_degree"]
        # Calculate expert_dp from the constraint
        tp = cfg["parallelism"]["tp"]
        cp = cfg["parallelism"].get("cp", 1)
        dp = cfg["parallelism"]["dp"]
        expert_dp = safe_divide(dp * cp * tp, ep * expert_tp)
        expert_mesh = ParallelConfig.ExpertParallelCfg(ep=ep, tp=expert_tp, dp=expert_dp)
        print("expert mesh:", expert_mesh)

    # Parse optional cross-DC configuration
    cross_dc_config = None
    if "cross_dc" in cfg and cfg["cross_dc"] is not None:
        # The cross-DC DP-comm helpers still reduce the WHOLE bucket over expert_dp
        # (the pre-split "assume all params are MoE" approximation). The non-cross-DC
        # path now splits expert vs dense DP reductions; the cross-DC path does not.
        # For a dense model these agree (no expert params), but for MoE+cross-DC they
        # would report an internally inconsistent degradation (and the cross-DC
        # latency term goes negative when expert_dp==1). Fail loudly rather than
        # silently mis-report until the cross-DC helpers get the same expert/dense
        # split (bias-for-failure, GUIDELINES §5).
        if expert_mesh is not None:
            raise NotImplementedError(
                "cross-DC DP-comm modeling assumes a dense model; it has not been "
                "updated for the MoE expert/dense DP-reduction split (Effect B). "
                "Remove cross_dc or extend the cross-DC comm helpers first."
            )
        cross_dc_config = CrossDCConfig(
            n_dcs=cfg["cross_dc"]["n_dcs"],
            interconnect_bandwidth_gbps=cfg["cross_dc"]["interconnect_bandwidth_gbps"],
            interconnect_latency_s=cfg["cross_dc"]["interconnect_latency_s"],
        )

    # Parse optimizer configuration (default to adam if not specified)
    optimizer_type = "adam"
    if "optimizer" in cfg and "optimizer_type" in cfg["optimizer"]:
        optimizer_type = cfg["optimizer"]["optimizer_type"]

    model_repr = ThreeDParallelModel(
        parallelism_cfg=ParallelConfig(
            tp=cfg["parallelism"]["tp"],
            cp=cfg["parallelism"].get("cp", 1),
            pp=cfg["parallelism"]["pp"],
            dp=cfg["parallelism"]["dp"],
            expert_mesh=expert_mesh,
            vpp=cfg["parallelism"]["vpp"],
            sp_enabled=cfg["parallelism"]["sp"],
            zero_level=ParallelConfig.ZeroLevel(cfg["parallelism"]["zero_level"]),
        ),
        sequence_len=sequence_len,
        microbatch_sz=microbatch_sz,
        hidden_sz=hidden_sz,
        n_layers=cfg["model"]["n_layers"],
        n_q_heads=cfg["model"]["n_q_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        head_dim=cfg["model"]["head_dim"],
        inter_sz=cfg["model"]["inter_sz"],
        glu=cfg["model"]["glu"],
        moe_cfg=MoeCfg(
            n_experts=cfg["model"]["moe"]["n_experts"],
            expert_inter_sz=cfg["model"]["moe"]["expert_inter_sz"],
            experts_per_token=cfg["model"]["moe"]["experts_per_token"],
            capacity_factor=cfg["model"]["moe"]["capacity_factor"],
            moe_frequency=cfg["model"]["moe"]["moe_frequency"],
            expert_tp_degree=cfg["model"]["moe"]["expert_tp_degree"],
            # Optional dense shared-expert FFN (0 = none). Set from MLflow
            # moe_shared_expert_intermediate_size (== moe_ffn_hidden_size).
            shared_expert_inter_sz=cfg["model"]["moe"].get("shared_expert_inter_sz", 0),
        )
        if "moe" in cfg["model"]
        else None,
        rotary_embed=cfg["model"]["rotary_embeds"],
        dropout=cfg["model"]["dropout"],
        vocab_sz=cfg["model"]["vocab_sz"],
        tie_embeddings=cfg["model"]["tie_embeddings"],
        act_ckpting_type=ActivationCheckpointingType.from_str(
            cfg["performance"]["activation_checkpointing_type"]
        ),
        n_param_buckets=cfg["parallelism"]["n_param_buckets"],
        optimizer_type=optimizer_type,
    )

    machine_spec = MachineSpec.from_str(cfg["hardware"]["node_type"])
    cluster_size = model_repr.parallelism_cfg.world_size()

    print_section_separator()
    print_info("Hardware Configuration")
    print_kv("Node Type", cfg["hardware"]["node_type"], key_width=30)
    print_kv("Total Devices", str(cluster_size), key_width=30)
    print_kv("Total Nodes", str(safe_divide(cluster_size, machine_spec.n_devices)), key_width=30)
    print_kv(
        "Device Memory",
        f"{machine_spec.device_spec.mem_capacity_bytes / (1024**3):.0f} GiB",
        key_width=30,
    )
    # Determine model precision, defaulting to FP16 if not specified
    model_dtype = DType.FP16
    if "precision" in cfg["model"]:
        model_dtype = DType(cfg["model"]["precision"])

    print_kv(
        "Peak FLOPS/device",
        f"{machine_spec.device_spec.peak_flops(model_dtype) / 1e12:.0f} TFLOPS",
        key_width=30,
    )

    if cross_dc_config is not None:
        print_section_separator()
        print_info("Cross-DC Configuration")
        print_kv("Number of DCs", str(cross_dc_config.n_dcs), key_width=30)
        print_kv(
            "Interconnect Bandwidth",
            f"{cross_dc_config.interconnect_bandwidth_gbps:.0f} Gbps",
            key_width=30,
        )
        print_kv(
            "Interconnect Latency",
            f"{cross_dc_config.interconnect_latency_s * 1000:.2f} ms",
            key_width=30,
        )
        print_kv(
            "Max Ring Latency",
            f"{cross_dc_config.interconnect_latency_s * 1000:.2f} ms",
            key_width=30,
        )
        nodes_per_dc = safe_divide(cluster_size, machine_spec.n_devices) // cross_dc_config.n_dcs
        print_kv("Nodes per DC", str(nodes_per_dc), key_width=30)

    ###################################################################################
    # DATA
    ###################################################################################
    print_section_separator()
    print_info("Data Configuration")
    gbs = cfg["data"]["gbs"]
    mbs = cfg["data"]["microbatch_sz"]

    bs_per_mp_rank = safe_divide(gbs, model_repr.parallelism_cfg.dp)
    n_microbatches_per_mp_rank = safe_divide(bs_per_mp_rank, mbs)

    print_kv("Global Batch Size", f"{gbs} samples", key_width=30)
    print_kv("Total Tokens/Batch", f"{format_number(gbs * sequence_len)} tokens", key_width=30)
    print_kv("Batch Size per DP Rank", str(bs_per_mp_rank), key_width=30)
    print_kv("Microbatches per Rank", str(n_microbatches_per_mp_rank), key_width=30)
    print_kv("Sequence Length", f"{sequence_len} tokens", key_width=30)

    ###################################################################################
    # MODEL SUMMARY
    ###################################################################################
    print_section_separator()
    print_info("Model Architecture")
    total_params = model_repr.get_n_total_params(partitioned=False)
    active_params = model_repr.get_n_active_params(partitioned=False)

    print_metric("Total Parameters", format_number(total_params), highlight=True)
    print_metric("Active Parameters", format_number(active_params))
    print_kv("Hidden Size", str(hidden_sz), key_width=30)
    print_kv("Number of Layers", str(cfg["model"]["n_layers"]), key_width=30)
    print_kv(
        "Attention Heads",
        f"{cfg['model']['n_q_heads']} (Q) / {cfg['model']['n_kv_heads']} (KV)",
        key_width=30,
    )

    ###################################################################################
    # MEMORY ANALYSIS
    ###################################################################################
    print_h1_header("MEMORY")
    print_section_separator()
    print_info("Model States")
    print(model_repr.states)

    print_section_separator()
    print_info("Activations")
    act_size_per_layer_per_inflight_microbatch = (
        model_repr.activation_size_per_microbatch_per_layer()
    )
    max_inflight_microbatches = model_repr.parallelism_cfg.pp  # 1F1B
    layers_per_pp_stage = model_repr.layers_per_pp_stage()
    vpp_multiplier = model_repr.vpp_penalty()

    print_kv(
        "Activation/Layer/Microbatch", str(act_size_per_layer_per_inflight_microbatch), key_width=30
    )
    print_kv("Max Inflight Microbatches", str(max_inflight_microbatches), key_width=30)
    print_kv("Layers per PP Stage", str(layers_per_pp_stage), key_width=30)
    print_kv("VPP Memory Multiplier", f"{vpp_multiplier:.2f}x", key_width=30)
    act_memory = (
        act_size_per_layer_per_inflight_microbatch
        * min(n_microbatches_per_mp_rank, max_inflight_microbatches)
        * math.ceil(vpp_multiplier * layers_per_pp_stage)
    )
    print_kv("Total Activation Memory", f"{act_memory.bytes() / (1024**3):.3f} GiB", key_width=30)

    print()
    print_info("Activation Breakdown per Layer/Microbatch")
    activation_breakdown = model_repr.activation_breakdown_per_microbatch_per_layer()

    # Calculate max sizes for better alignment
    max_name_len = max(len(name) for name in activation_breakdown.keys())
    total_numel = sum(activation_breakdown.values())
    total_size_mib = (total_numel * model_repr.bits_per_parameter // 8) / (1024**2)

    for name, numel in activation_breakdown.items():
        size_mib = (numel * model_repr.bits_per_parameter // 8) / (1024**2)
        percentage = (numel / total_numel) * 100
        bar_width = int(percentage / 2)  # Scale to fit in terminal
        bar = "█" * bar_width if bar_width > 0 else ""
        color = get_color_for_component_percentage(percentage)
        print(
            f"    {name:<{max_name_len}} │ {numel:>12,} │ {size_mib:>8.1f} MiB │ {color} {percentage:>5.1f}% {bar}{_END}"
        )
    print(f"  {'─' * (max_name_len + 45)}")
    print(
        f"{_BOLD}    {'TOTAL':<{max_name_len}} │ {total_numel:>12,} │ {total_size_mib:>8.1f} MiB {_END}"
    )

    # MoE workspace memory (if applicable)
    moe_workspace = model_repr.get_moe_workspace_memory()
    if moe_workspace.bytes() > 0:
        print_section_separator()
        print_info("MoE Workspace")
        print_kv("MoE Workspace Memory", f"{moe_workspace.bytes() / (1024**3):.3f} GiB", key_width=30)
        print("  (all-to-all buffers, GEMM workspace, permutation buffers, etc.)")

    print_section_separator()
    print_info("Summary")
    total_memory_gib = (
        model_repr.states.total_bytes(partitioned=True)
        + act_memory.bytes()
        + moe_workspace.bytes()
    ) / (1024**3)

    # TODO: temporary fp8 fix
    # Apply 12% memory increase for fp8 precision
    if model_dtype == DType.FP8 or model_dtype == DType.FP8_E4M3:
        total_memory_gib = total_memory_gib * 1.12

    print_success(f"Total Memory Required: {total_memory_gib:.3f} GiB per device")

    ###################################################################################
    # PERF ANALYSIS
    ###################################################################################
    print_h1_header("COMPUTE: GEMM OPERATIONS")
    print_info("Note: Numbers calculated assuming 100% FLOPS and bandwidth utilization")
    n_tokens_cp = (
        safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp)
        * model_repr.microbatch_sz
    )
    projections = OrderedDict(
        {
            "QKV Projection": (model_repr.qkv_weight, n_tokens_cp),
            "Attention Combine Projection": (model_repr.attn_out_weight, n_tokens_cp),
            "MLP Up Projection": (model_repr.mlp_up_weight, n_tokens_cp),
            "MLP Down Projection": (model_repr.mlp_down_weight, n_tokens_cp),
        }
    )

    if model_repr.mlp_up_exp_weight is not None:
        expert_dim, *other_dims = model_repr.mlp_up_exp_weight.shape(partitioned=False)
        single_expert_shape = tuple(other_dims)
        single_expert_partition_spec = {
            k - 1: v for k, v in model_repr.mlp_up_exp_weight._partition_spec.items() if k != 0
        }

        projections["MLP Up (Expert)"] = (
            TensorRepr(
                unpartitioned_shape=single_expert_shape,
                partition_spec=single_expert_partition_spec,
                bits_per_elt=model_repr.bits_per_parameter,
            ),
            model_repr.expert_capacity(),
        )
    if model_repr.mlp_down_exp_weight is not None:
        expert_dim, *other_dims = model_repr.mlp_down_exp_weight.shape(partitioned=False)
        single_expert_shape = tuple(other_dims)
        single_expert_partition_spec = {
            k - 1: v for k, v in model_repr.mlp_down_exp_weight._partition_spec.items() if k != 0
        }

        projections["MLP Down (Expert)"] = (
            TensorRepr(
                unpartitioned_shape=single_expert_shape,
                partition_spec=single_expert_partition_spec,
                bits_per_elt=model_repr.bits_per_parameter,
            ),
            model_repr.expert_capacity(),
        )

    for proj_name, (weight_repr, n_tokens) in projections.items():
        flops = compute_gemm_flops(
            n_tokens=n_tokens,
            weight_shape=weight_repr.shape(partitioned=True),
        )
        compute_time_ms = flops / machine_spec.device_spec.peak_flops(model_dtype) * 1000

        # Color based on compute intensity
        color = get_color_by_time_ms(compute_time_ms)

        print(f"\n  {_BOLD}{proj_name}{_END}")
        print(
            f"    Shape: {weight_repr.shape(partitioned=False)} → Partitioned: {weight_repr.shape(partitioned=True)}"
        )

        # Compute metrics with bar
        print(f"    Compute: {color}{compute_time_ms:.3f} ms{_END}")
        print(f"             {_GRAY}({format_number(float(flops))} FLOPs){_END}")

        # Memory bandwidth metrics in a compact format
        bytes_per_element = safe_divide(model_repr.bits_per_parameter, 8)
        gemm_input_dim, gemm_output_dim = weight_repr.shape(partitioned=True)
        weight_bytes = bytes_per_element * weight_repr.numel(partitioned=True)
        input_bytes = n_tokens * gemm_input_dim * bytes_per_element
        output_bytes = n_tokens * gemm_output_dim * bytes_per_element
        input_time_ms = input_bytes / machine_spec.device_spec.mem_bandwidth_bytes_per_sec * 1000
        weight_time_ms = weight_bytes / machine_spec.device_spec.mem_bandwidth_bytes_per_sec * 1000
        output_time_ms = output_bytes / machine_spec.device_spec.mem_bandwidth_bytes_per_sec * 1000
        print(f"    Memory:  Input: {input_bytes / 1e9:.2f} GB ({input_time_ms:.3f} ms)")
        print(f"             Weight: {weight_bytes / 1e9:.2f} GB ({weight_time_ms:.3f} ms)")
        print(f"             Output: {output_bytes / 1e9:.2f} GB ({output_time_ms:.3f} ms)")

    print()

    print_h1_header("COMMUNICATION: TENSOR PARALLELISM")
    if not model_repr.parallelism_cfg.sp_enabled:
        raise NotImplementedError("not implemented for non-SP case")

    activation_size = Size(
        numel=safe_divide(sequence_len, model_repr.parallelism_cfg.cp) * microbatch_sz * hidden_sz,
        bits_per_element=model_repr.bits_per_parameter,
    )
    tp_ag_time = get_tp_all_gather_comm_time_s(
        size=activation_size, parallel_config=model_repr.parallelism_cfg, machine_spec=machine_spec
    )
    tp_rs_time = get_tp_reduce_scatter_comm_time_s(
        size=activation_size, parallel_config=model_repr.parallelism_cfg, machine_spec=machine_spec
    )

    print_kv("TP All-Gather", f"{tp_ag_time * 1000:.3f} ms", key_width=30)
    print_kv("TP Reduce-Scatter", f"{tp_rs_time * 1000:.3f} ms", key_width=30)
    print_kv("Activation Size", str(activation_size), key_width=30)

    print_h1_header("COMMUNICATION: PIPELINE PARALLELISM")
    activation_send_time_s = (
        activation_size.bytes() / machine_spec.inter_node_connect.unidirectional_bw_bytes_per_sec
    )
    print_kv("PP Send/Recv Time", f"{activation_send_time_s * 1000:.3f} ms", key_width=30)
    print_kv("Activation Size", str(activation_size), key_width=30)

    print_h1_header("PERFORMANCE: PIPELINE BUBBLE")

    vpp = cfg["parallelism"]["vpp"]
    bs_per_mp_rank = safe_divide(gbs, model_repr.parallelism_cfg.dp)
    n_microbatches_per_mp_rank = safe_divide(bs_per_mp_rank, mbs)

    pp = model_repr.parallelism_cfg.pp

    # (Interleaved) 1F1B: bubble/compute = (pp-1)/(vpp*n_mb), uncapped -- the
    # fill/drain ramp IS the entire bubble. See utils/pipeline.py for the
    # derivation and the 18b trace validation. Can exceed 1 for n_mb < pp-1.
    pipeline_bubble_fraction = compute_pipeline_bubble_fraction(
        pp=pp, vpp=vpp, n_microbatches=n_microbatches_per_mp_rank
    )

    print_kv("VPP Pipeline Bubble Multiplier", f"{(1 / vpp):.2f}x", key_width=30)
    print_kv("Pipeline Bubble Fraction", f"{pipeline_bubble_fraction:.2%}", key_width=30)

    print_h1_header("COMMUNICATION: DATA PARALLELISM")

    # Initialize cross-DC communication times (will be updated if cross-DC is enabled)
    cross_dc_grad_bucket_rs_time_s = None
    cross_dc_param_bucket_ag_time_s = None
    cross_dc_grad_bucket_ar_time_s = None

    zero_level = model_repr.parallelism_cfg.zero_level
    uses_all_reduce = zero_level == ParallelConfig.ZeroLevel.NONE

    print_section_separator()
    print_info("Microbatch Compute Times (100% FLOPS utilization)")

    devices_in_pp_stage_flops = (
        model_repr.parallelism_cfg.cp
        * model_repr.parallelism_cfg.tp
        * machine_spec.device_spec.peak_flops(model_dtype)
    )

    # divide by single pipeline stage TFLOPs, since its just for single
    # microbatch there's only one active pipeline stage at a time
    single_microbatch_fwd_time = (
        model_repr.get_single_microbatch_fwd_flops() / devices_in_pp_stage_flops
    )
    single_microbatch_bwd_time = (
        model_repr.get_single_microbatch_bwd_flops() / devices_in_pp_stage_flops
    )

    print_kv("Forward Pass", f"{single_microbatch_fwd_time * 1000:.3f} ms", key_width=30)
    print_kv("Backward Pass", f"{single_microbatch_bwd_time * 1000:.3f} ms", key_width=30)

    # Gradient bucketing configuration
    print_section_separator()
    print_info("Gradient Bucketing")
    print_kv(
        "Zero Level",
        f"{zero_level.name} -> {'AllReduce' if uses_all_reduce else 'Reduce-Scatter + All-Gather'}",
        key_width=30,
    )

    # grads are reduced in full-precision
    # params are all-gathered in half-precision
    n_buckets = model_repr.n_param_buckets
    # When zero_level=NONE, params are unsharded (dp=1 passed into opt states),
    # so params_shard is full size. We still want the per-DP-replica bucket.
    mp_params_size = model_repr.states.params_shard.size(partitioned=True)
    param_bucket_numel = mp_params_size.numel() // model_repr.n_param_buckets
    # TODO. precisions here assume we are doing AMP
    param_bucket_size = Size(
        numel=param_bucket_numel,
        bits_per_element=model_repr.bits_per_parameter,
    )
    grad_bucket_size = Size(
        numel=param_bucket_numel,
        bits_per_element=model_repr.bits_per_grad,
    )

    print_kv("Params per MP rank", str(mp_params_size), key_width=30)
    print_kv("Bucket Size", f"{format_number(param_bucket_numel)} params", key_width=30)
    print_kv("Number of Buckets", str(n_buckets), key_width=30)

    # DP gradient reduction has TWO groups with different replica sizes: routed-expert
    # params reduce over expert_mesh.dp, every other param over the full dp. Modeling
    # the whole bucket as one expert_dp reduction under-counts (and, when expert_dp==1,
    # zeroes) the dense-param reduction — the "assume all params are MoE" approximation
    # that this split replaces. We partition each bucket into expert/dense shares by
    # their param fraction and reduce each over its own group. (No expert params ->
    # everything reduces over dp, recovering the dense-model behavior.)
    n_expert_params = model_repr.get_n_expert_params_per_stage(partitioned=True)
    n_dense_params = model_repr.get_n_dense_params_per_stage(partitioned=True)
    n_total_params_for_dp = max(1, n_expert_params + n_dense_params)
    expert_param_fraction = n_expert_params / n_total_params_for_dp
    dense_param_fraction = n_dense_params / n_total_params_for_dp

    def _dp_bucket_time_s(bits_per_element: int, comm_fn: Callable[..., float]) -> float:
        """Per-bucket DP comm time = expert-share reduced over expert_dp
        + dense-share reduced over dp. comm_fn is one of the get_dp_* helpers."""
        expert_bucket_numel = int(param_bucket_numel * expert_param_fraction)
        dense_bucket_numel = int(param_bucket_numel * dense_param_fraction)
        total = 0.0
        # Guard on the TRUNCATED numel (not the fraction): a zero-byte message would
        # still incur the pure ring-latency term, over-counting.
        if model_repr.parallelism_cfg.expert_mesh is not None and expert_bucket_numel > 0:
            total += comm_fn(
                size=Size(numel=expert_bucket_numel, bits_per_element=bits_per_element),
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
                is_expert_comm=True,
            )
        if dense_bucket_numel > 0:
            total += comm_fn(
                size=Size(numel=dense_bucket_numel, bits_per_element=bits_per_element),
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
                is_expert_comm=False,
            )
        return total

    # Initialize comm-time variables used downstream. Only one of the two paths
    # (AllReduce vs. RS+AG) contributes to exposed time; the other stays at 0.
    grad_bucket_reduce_scatter_time_s = 0.0
    param_bucket_all_gather_time_s = 0.0
    grad_bucket_all_reduce_time_s = 0.0

    if uses_all_reduce:
        # zero_level=NONE: gradients are all-reduced across DP each step,
        # optimizer state is replicated across DP (not sharded). Split expert vs
        # dense params across their respective DP groups (see _dp_bucket_time_s).
        grad_bucket_all_reduce_time_s = _dp_bucket_time_s(
            model_repr.bits_per_grad, get_dp_all_reduce_comm_time_s
        )

        print_section_separator()
        print_info("Communication Breakdown (per bucket)")

        print(f"\n  {_BOLD}All-Reduce (Gradients){_END}")
        print_kv(
            "  Dense params reduced over",
            f"dp={model_repr.parallelism_cfg.dp}",
            key_width=30,
        )
        if model_repr.parallelism_cfg.expert_mesh is not None:
            print_kv(
                "  Expert params reduced over",
                f"expert_dp={model_repr.parallelism_cfg.expert_mesh.dp}",
                key_width=30,
            )
        print_metric(
            "  Total", f"{grad_bucket_all_reduce_time_s * 1000:.3f}", "ms", highlight=True
        )

        print_section_separator()
        print_info("Total Communication Time (all buckets)")
        total_ar_time = grad_bucket_all_reduce_time_s * n_buckets * 1000
        print_metric("All-Reduce Total", f"{total_ar_time:.2f}", "ms", highlight=True)
        # Per-step DP comm total (ms) for the collective-cost audit summary.
        _dp_comm_total_step_ms = total_ar_time

        if cross_dc_config is not None:
            print_section_separator()
            print_info("Cross-DC Impact on DP Communication")
            cross_dc_grad_bucket_ar_time_s = get_cross_dc_dp_all_reduce_comm_time_s(
                size=grad_bucket_size,
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
                cross_dc_config=cross_dc_config,
            )
            ar_degradation_ms = (
                cross_dc_grad_bucket_ar_time_s - grad_bucket_all_reduce_time_s
            ) * 1000
            ar_degradation_pct = (
                ar_degradation_ms / (grad_bucket_all_reduce_time_s * 1000) * 100
                if grad_bucket_all_reduce_time_s > 0
                else 0.0
            )
            print_kv(
                "  All-Reduce Delta",
                f"{ar_degradation_ms:.3f} ms ({ar_degradation_pct:.1f}% slower)",
                key_width=30,
            )
            total_cross_dc_ar_time = cross_dc_grad_bucket_ar_time_s * n_buckets * 1000
            print_metric(
                "  Total Cross-DC DP",
                f"{total_cross_dc_ar_time:.2f}",
                f"ms ({ar_degradation_pct:.1f}% slower)",
                highlight=True,
            )
            # Cross-DC is the real DP path here; report it (not the intra-DC time).
            _dp_comm_total_step_ms = total_cross_dc_ar_time
    else:
        # zero_level=PARTITION_OPTIMIZER (ZeRO-1): grads are reduce-scattered,
        # params are all-gathered. Split expert vs dense params across their
        # respective DP groups (see _dp_bucket_time_s).
        grad_bucket_reduce_scatter_time_s = _dp_bucket_time_s(
            model_repr.bits_per_grad, get_dp_reduce_scatter_comm_time_s
        )
        # Communication breakdown per bucket
        print_section_separator()
        print_info("Communication Breakdown (per bucket)")

        print(f"\n  {_BOLD}Reduce-Scatter (Gradients){_END}")
        print_kv(
            "  Dense params reduced over",
            f"dp={model_repr.parallelism_cfg.dp}",
            key_width=30,
        )
        if model_repr.parallelism_cfg.expert_mesh is not None:
            print_kv(
                "  Expert params reduced over",
                f"expert_dp={model_repr.parallelism_cfg.expert_mesh.dp}",
                key_width=30,
            )
        print_metric(
            "  Total", f"{grad_bucket_reduce_scatter_time_s * 1000:.3f}", "ms", highlight=True
        )

        param_bucket_all_gather_time_s = _dp_bucket_time_s(
            model_repr.bits_per_parameter, get_dp_all_gather_comm_time_s
        )

        print(f"\n  {_BOLD}All-Gather (Parameters){_END}")
        print_kv(
            "  Dense params gathered over",
            f"dp={model_repr.parallelism_cfg.dp}",
            key_width=30,
        )
        if model_repr.parallelism_cfg.expert_mesh is not None:
            print_kv(
                "  Expert params gathered over",
                f"expert_dp={model_repr.parallelism_cfg.expert_mesh.dp}",
                key_width=30,
            )
        print_metric(
            "  Total", f"{param_bucket_all_gather_time_s * 1000:.3f}", "ms", highlight=True
        )

        # Total communication times
        print_section_separator()
        print_info("Total Communication Time (all buckets)")

        total_rs_time = grad_bucket_reduce_scatter_time_s * n_buckets * 1000
        total_ag_time = param_bucket_all_gather_time_s * n_buckets * 1000

        print_kv("Reduce-Scatter Total", f"{total_rs_time:.2f} ms", key_width=30)
        print_kv("All-Gather Total", f"{total_ag_time:.2f} ms", key_width=30)
        print_metric(
            "Combined DP Comm", f"{total_rs_time + total_ag_time:.2f}", "ms", highlight=True
        )
        # Per-step DP comm total (ms) for the collective-cost audit summary.
        _dp_comm_total_step_ms = total_rs_time + total_ag_time

        # Cross-DC impact analysis
        if cross_dc_config is not None:
            print_section_separator()
            print_info("Cross-DC Impact on DP Communication")

            cross_dc_grad_bucket_rs_time_s = get_cross_dc_dp_reduce_scatter_comm_time_s(
                size=grad_bucket_size,
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
                cross_dc_config=cross_dc_config,
            )

            cross_dc_param_bucket_ag_time_s = get_cross_dc_dp_all_gather_comm_time_s(
                size=param_bucket_size,
                parallel_config=model_repr.parallelism_cfg,
                machine_spec=machine_spec,
                cross_dc_config=cross_dc_config,
            )

            rs_degradation_ms = (
                cross_dc_grad_bucket_rs_time_s - grad_bucket_reduce_scatter_time_s
            ) * 1000
            ag_degradation_ms = (
                cross_dc_param_bucket_ag_time_s - param_bucket_all_gather_time_s
            ) * 1000

            rs_degradation_pct = (
                rs_degradation_ms / (grad_bucket_reduce_scatter_time_s * 1000)
            ) * 100
            ag_degradation_pct = (ag_degradation_ms / (param_bucket_all_gather_time_s * 1000)) * 100

            print(f"\n  {_BOLD}Per-Bucket Cross-DC Degradation{_END}")
            print_kv(
                "  Reduce-Scatter Delta",
                f"{rs_degradation_ms:.3f} ms ({rs_degradation_pct:.1f}% slower)",
                key_width=30,
            )
            print_kv(
                "  All-Gather Delta",
                f"{ag_degradation_ms:.3f} ms ({ag_degradation_pct:.1f}% slower)",
                key_width=30,
            )

            total_cross_dc_rs_time = cross_dc_grad_bucket_rs_time_s * n_buckets * 1000
            total_cross_dc_ag_time = cross_dc_param_bucket_ag_time_s * n_buckets * 1000
            total_cross_dc_dp_time = total_cross_dc_rs_time + total_cross_dc_ag_time

            print(f"\n  {_BOLD}Total Cross-DC DP Communication{_END}")
            print_kv("  Cross-DC RS Total", f"{total_cross_dc_rs_time:.2f} ms", key_width=30)
            print_kv("  Cross-DC AG Total", f"{total_cross_dc_ag_time:.2f} ms", key_width=30)

            total_degradation_ms = total_cross_dc_dp_time - (total_rs_time + total_ag_time)
            total_degradation_pct = (total_degradation_ms / (total_rs_time + total_ag_time)) * 100

            print_metric(
                "  Total Cross-DC DP",
                f"{total_cross_dc_dp_time:.2f}",
                f"ms ({total_degradation_pct:.1f}% slower)",
                highlight=True,
            )
            # Cross-DC is the real DP path here; report it (not the intra-DC time).
            _dp_comm_total_step_ms = total_cross_dc_dp_time

    ##################################################################################
    # Iteration Time
    ##################################################################################
    print_h1_header("PERFORMANCE: ITERATION TIME ANALYSIS")
    print_info(
        "NOTE: This is intended to give theoretical time estimates. \n  "
        "Any gaps between this and observations will be a combination of: \n  "
        "\n  "
        "a) errors in the modeling. please cut an issue if you find one. \n  "
        "b) implementation issues that you should consider fixing. \n  "
        "   common issues include CPU boundedness, jitter, stragglers, \n  "
        "   dataloading, etc. \n  "
    )
    n_tokens = model_repr.microbatch_sz * model_repr.sequence_len

    def compute_gemm_time_s(weight_repr: TensorRepr) -> float:
        m = safe_divide(n_tokens, model_repr.parallelism_cfg.cp)
        weight_shape = weight_repr.shape(partitioned=True)
        k, n = weight_shape  # weight shape is (input_dim, output_dim)

        gemm_util = get_gemm_utilization_or_default(
            m=m, n=n, k=k, machine_spec=machine_spec, dtype=model_dtype
        )
        
        flops = compute_gemm_flops(n_tokens=m, weight_shape=weight_shape)
        total_runtime_flops = machine_spec.device_spec.peak_flops(model_dtype) * gemm_util
        if total_runtime_flops == 0:
            total_runtime_flops = 0.05
        return flops / total_runtime_flops

    ag_time_s = get_tp_all_gather_comm_time_s(
        size=activation_size,
        parallel_config=model_repr.parallelism_cfg,
        machine_spec=machine_spec,
    )
    rs_time_s = get_tp_reduce_scatter_comm_time_s(
        size=activation_size,
        parallel_config=model_repr.parallelism_cfg,
        machine_spec=machine_spec,
    )

    hbm_load_store_time_s = (
        2 * activation_size.bytes() / machine_spec.device_spec.mem_bandwidth_bytes_per_sec
    )

    # Get LayerNorm and RoPE timings from empirical data
    # These replace the rough HBM approximation with kernel-specific benchmarks
    norm_time_s = get_norm_time(
        batch=model_repr.microbatch_sz,
        seqlen=model_repr.sequence_len,
        hidden_dim=model_repr.hidden_sz,
        dtype=model_dtype,
        machine_spec=machine_spec,
    )
    rope_time_s = get_rope_time(
        batch=model_repr.microbatch_sz,
        seqlen=model_repr.sequence_len,
        hidden_dim=model_repr.hidden_sz,
        dtype=model_dtype,
        machine_spec=machine_spec,
    )

    # SDPA time.
    # Prefer the newer fwd+bwd measurement (sdpa_fwd_bwd_timings.parquet, indexed
    # by TP-sharded head counts to match what actually runs on a single GPU).
    # Fall back to the legacy un-sharded sdpa.parquet lookup, then to theoretical.
    from dlcalc.utils.sdpa_util import get_sdpa_fwd_time_s
    sdpa_n_q_heads_local = safe_divide(model_repr.n_q_heads, model_repr.parallelism_cfg.tp)
    sdpa_n_kv_heads_local = (
        safe_divide(model_repr.n_kv_heads, model_repr.parallelism_cfg.tp)
        if model_repr.n_kv_heads >= model_repr.parallelism_cfg.tp
        else model_repr.n_kv_heads
    )
    sdpa_time = get_sdpa_fwd_time_s(
        seq_len=sequence_len,
        micro_bs=microbatch_sz,
        n_q_heads=sdpa_n_q_heads_local,
        n_kv_heads=sdpa_n_kv_heads_local,
        head_dim=model_repr.head_dim,
        machine_spec=machine_spec,
        dtype=model_dtype,
    )
    if sdpa_time is None:
        # Legacy fallback: un-sharded lookup in sdpa.parquet (kept for parity
        # with configurations that haven't been re-measured with TP-sharded data).
        try:
            sdpa_time_ms = sdpa_df.loc[
                (sequence_len, microbatch_sz, model_repr.n_q_heads, model_repr.n_kv_heads, model_repr.head_dim),
                'time_ms_te_med'
            ]
            sdpa_time = sdpa_time_ms / 1000.0
        except (KeyError, AttributeError):
            print_info(
                f"Warning: No SDPA timing found for seq_len={sequence_len}, micro_bs={microbatch_sz}, "
                f"n_q_heads={model_repr.n_q_heads}, n_kv_heads={model_repr.n_kv_heads}, head_dim={model_repr.head_dim}. "
                f"Falling back to calculated value."
            )
            sdpa_m = safe_divide(sequence_len, model_repr.parallelism_cfg.cp)
            sdpa_n = sequence_len
            sdpa_k = model_repr.head_dim
            sdpa_gemm_util = get_gemm_utilization_or_default(
                m=sdpa_m, n=sdpa_n, k=sdpa_k, machine_spec=machine_spec, dtype=model_dtype
            )
            sdpa_flops = safe_divide(model_repr.n_q_heads, model_repr.parallelism_cfg.tp) * sum([
                2 * sdpa_m * sdpa_k * sdpa_n,
                2 * sequence_len * sequence_len * model_repr.head_dim,
            ])
            sdpa_time = sdpa_flops / (machine_spec.device_spec.peak_flops(model_dtype) * sdpa_gemm_util)

    transformer_block_time_components_dense: dict[str, float] = OrderedDict(
        {
            # Attention
            "Pre Attn Norm": norm_time_s,  # LayerNorm timing from lookup table
            "RoPE": rope_time_s,  # RoPE timing from lookup table
            "Pre Attn AG": ag_time_s,
            "QKV Proj": compute_gemm_time_s(model_repr.qkv_weight),
            "SDPA": sdpa_time,
            "Attn Out Proj": compute_gemm_time_s(model_repr.attn_out_weight),
            "Post Attn RS": rs_time_s,
            "Post Attn Residual": hbm_load_store_time_s,
            # MLP
            "Pre MLP Norm": norm_time_s,  # LayerNorm timing from lookup table
            "Pre MLP AG": ag_time_s,
            "MLP Up Proj": compute_gemm_time_s(model_repr.mlp_up_weight),
            "MLP Down Proj": compute_gemm_time_s(model_repr.mlp_down_weight),
            "Post MLP RS": rs_time_s,
            "Post MLP Residual": hbm_load_store_time_s,
        }
    )
    # NOTE: PP activation send/recv is NOT a per-layer block component. A microbatch
    # crosses a pipeline-stage boundary once per STAGE (one send at the stage's last
    # layer, one grad-recv on the way back), not once per layer. It is therefore
    # charged once per microbatch as a separate per-step term (see pp_send_recv_time_s
    # below), not multiplied by layers_per_pp_stage here. (Charging it per layer
    # over-counted PP p2p by a factor of layers_per_pp_stage — e.g. 6x on 18b pp8.)

    transformer_block_time_components_moe: dict[str, float] = {}
    if model_repr.moe_cfg is not None:
        assert model_repr.parallelism_cfg.expert_mesh is not None

        expert_capacity = model_repr.expert_capacity()
        n_local_experts = safe_divide(
            model_repr.moe_cfg.n_experts,
            model_repr.parallelism_cfg.expert_mesh.ep,
        )

        # Per-DEVICE per-expert token load (the grouped-GEMM tile M). Dropless
        # mean load = seq*mbs*top_k / n_experts, which is EP- AND DP-invariant
        # (each device's local experts see only that device's own token shard).
        # NOTE expert_capacity() = this * expert_dp (it sums the per-expert load
        # over the expert-DP replicas); the grouped kernel on ONE device runs at
        # the per-device tile, so we key the measured table on this value.
        tokens_per_expert_local = safe_divide(
            safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp)
            * model_repr.microbatch_sz
            * model_repr.moe_cfg.experts_per_token,
            model_repr.moe_cfg.n_experts,
        )
        expert_ffn = model_repr.moe_cfg.expert_inter_sz
        use_measured_grouped = grouped_gemm_measured(machine_spec.name, model_dtype)

        def compute_expert_gemm_time_s(n_tokens_per_expert: int, weight_repr: TensorRepr) -> float:
            # The local experts run as ONE grouped/batched GEMM (TE GroupedLinear;
            # `nvjet_tst_*` on B200), whose FLOPs are the aggregate over all local
            # experts (M_eff = n_local_experts * tokens/expert -- EP-invariant), but
            # whose EFFICIENCY is set by the PER-GROUP tile M (= tokens/expert), NOT
            # the aggregate M_eff.
            #
            # MEASURED on B200 (benchmarks/gemm_grouped_benchmark.py ->
            # results/gemm_grouped_b200.parquet, TE 2.4.90, 2026-07-14): at the
            # dropless cap tokens/expert=192 the grouped kernel runs at ~4.5% util,
            # matching a single GEMM at M=192 (~5%), NOT the ~29% a single GEMM at
            # M_eff=3072 achieves. Looking util up at M_eff over-credited expert
            # compute by 5-11x in the dropless regime -- the dominant MFU
            # over-prediction (see expert-gemm-grouped-util-overcredit).
            #
            # Fix (physics, measurement-validated; no fudge factor): keep FLOPs at
            # the aggregate M_eff, look util up at the PER-GROUP M. This is flat in
            # EP (tokens/expert is EP-invariant) as the ~140ms measured expert-GEMM
            # requires. It under-counts util (over-predicts time) at large
            # tokens/expert (>=1024), where the grouped kernel amortizes toward the
            # aggregate; the measured grouped parquet can back a higher-fidelity
            # tokens/expert->util lookup as a follow-up. See
            # backward.compute_grouped_linear_bwd_time_s for the symmetric bwd fix.
            n_local_experts, *gemm_dims = weight_repr.shape(partitioned=True)
            k, n = gemm_dims  # gemm_dims is (input_dim, output_dim)
            m_eff = n_local_experts * n_tokens_per_expert

            gemm_util = get_gemm_utilization_or_default(
                m=n_tokens_per_expert, n=n, k=k, machine_spec=machine_spec, dtype=model_dtype
            )

            flops = compute_gemm_flops(
                n_tokens=m_eff,
                weight_shape=tuple(gemm_dims),
            )
            return flops / (machine_spec.device_spec.peak_flops(model_dtype) * gemm_util)

        a2a_time_s = get_all_to_all_comm_time_s(
            size=Size(
                n_local_experts
                * safe_divide(expert_capacity, model_repr.parallelism_cfg.expert_mesh.tp)
                * hidden_sz,
                bits_per_element=model_repr.bits_per_parameter,
            ),
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )

        expert_activation_size = Size(
            numel=n_local_experts * expert_capacity * model_repr.hidden_sz,
            bits_per_element=model_repr.bits_per_parameter,
        )

        assert model_repr.router_weight is not None

        # Calculate MoE router overhead (TopK selection and token permutation)
        # These operations are in addition to the router GEMM already calculated above
        router_topk_time_s = calculate_topk_time(
            batch=model_repr.microbatch_sz,
            seqlen=safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp),
            n_experts=model_repr.moe_cfg.n_experts,
            k=model_repr.moe_cfg.experts_per_token,
            machine_spec=machine_spec,
        )

        router_permutation_time_s = calculate_permutation_time(
            batch=model_repr.microbatch_sz,
            seqlen=safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp),
            hidden_dim=model_repr.hidden_sz,
            machine_spec=machine_spec,
            dtype_bytes=safe_divide(model_repr.bits_per_parameter, 8),
        )

        transformer_block_time_components_moe: dict[str, float] = OrderedDict(  # type: ignore[no-redef]
            {
                # Attention
                "Pre Attn Norm": norm_time_s,  # LayerNorm timing from lookup table
                "RoPE": rope_time_s,  # RoPE timing from lookup table
                "Pre Attn AG": ag_time_s,
                "QKV Proj": compute_gemm_time_s(model_repr.qkv_weight),
                "SDPA": sdpa_time,
                "Attn Out Proj": compute_gemm_time_s(model_repr.attn_out_weight),
                "Post Attn RS": rs_time_s,
                "Post Attn Residual": hbm_load_store_time_s,
                # MLP / MoE Router
                "Pre MLP Norm": norm_time_s,  # LayerNorm timing from lookup table
                "Router GEMM": compute_gemm_time_s(model_repr.router_weight),
                "Router TopK": router_topk_time_s,
                "Router Permutation": router_permutation_time_s,
                "Pre MLP A2A": a2a_time_s,
                "Pre MLP AG": get_expert_tp_all_gather_comm_time_s(
                    size=expert_activation_size,
                    parallel_config=model_repr.parallelism_cfg,
                    machine_spec=machine_spec,
                ),
                "MLP Up Proj": (
                    grouped_mlp_up_fwd_time_s(
                        n_local_experts=n_local_experts,
                        tokens_per_expert=tokens_per_expert_local,
                        hidden=model_repr.hidden_sz,
                        ffn=expert_ffn,
                    )
                    if use_measured_grouped
                    else compute_expert_gemm_time_s(
                        n_tokens_per_expert=expert_capacity,
                        weight_repr=model_repr.mlp_up_exp_weight,  # type: ignore[arg-type]
                    )
                ),
                "Glu Act": 3  # read 2, write 1
                * (
                    n_local_experts
                    * expert_capacity
                    * model_repr.moe_cfg.expert_inter_sz
                    * safe_divide(model_repr.bits_per_parameter, 8)
                )
                / machine_spec.device_spec.mem_bandwidth_bytes_per_sec,
                "MLP Down Proj": (
                    grouped_mlp_down_fwd_time_s(
                        n_local_experts=n_local_experts,
                        tokens_per_expert=tokens_per_expert_local,
                        hidden=model_repr.hidden_sz,
                        ffn=expert_ffn,
                    )
                    if use_measured_grouped
                    else compute_expert_gemm_time_s(
                        n_tokens_per_expert=expert_capacity,
                        weight_repr=model_repr.mlp_down_exp_weight,  # type: ignore[arg-type]
                    )
                ),
                "Post MLP RS": get_expert_tp_reduce_scatter_comm_time_s(
                    size=expert_activation_size,
                    parallel_config=model_repr.parallelism_cfg,
                    machine_spec=machine_spec,
                ),
                "Post MLP A2A": a2a_time_s,
                # Shared expert: a dense GLU MLP run on ALL tokens (not routed, not
                # ÷ n_experts). Timed with the dense compute_gemm_time_s (m = seq*mbs/cp)
                # and a GLU activation over the shared intermediate. Real FLOPs the
                # hardware does that the model previously omitted (Effect C).
                **(
                    {
                        "Shared Expert Up Proj": compute_gemm_time_s(
                            model_repr.shared_mlp_up_weight  # type: ignore[arg-type]
                        ),
                        "Shared Expert Glu Act": 3  # read 2, write 1
                        * (
                            n_tokens_cp
                            * safe_divide(
                                model_repr.moe_cfg.shared_expert_inter_sz,
                                model_repr.parallelism_cfg.tp,
                            )
                            * safe_divide(model_repr.bits_per_parameter, 8)
                        )
                        / machine_spec.device_spec.mem_bandwidth_bytes_per_sec,
                        "Shared Expert Down Proj": compute_gemm_time_s(
                            model_repr.shared_mlp_down_weight  # type: ignore[arg-type]
                        ),
                    }
                    if model_repr.shared_mlp_up_weight is not None
                    else {}
                ),
                "Post MLP Residual": hbm_load_store_time_s,
            }
        )
        # PP activation send is a per-stage-crossing term, not per-layer — see the
        # dense-block note above; it is charged once per microbatch below.

    print()

    ###############################################################################
    # BACKWARD PASS COMPONENTS
    #
    # Explicit autograd-style model: each forward op gets a corresponding backward
    # op (no multiplier heuristics). The backward graph is the reverse of the
    # forward graph, with TP all-gather/reduce-scatter swapping direction
    # (activation AG becomes grad RS, and vice versa) since that's how SP
    # autograd splits/replicates gradients.
    ###############################################################################
    m_tokens = safe_divide(n_tokens, model_repr.parallelism_cfg.cp)
    activation_numel = activation_size.numel()

    def _linear_bwd(weight_repr: TensorRepr) -> float:
        return compute_linear_bwd_time_s(
            n_tokens=m_tokens,
            weight_shape=weight_repr.shape(partitioned=True),
            machine_spec=machine_spec,
            dtype=model_dtype,
        )

    # SDPA backward: measured wall-clock from benchmarks/sdpa_benchmark.py.
    # GQA-aware: n_kv_heads is usually shared across TP groups but we still
    # pass it so the lookup can differentiate GQA ratios.
    sdpa_bwd_time = compute_sdpa_bwd_time_s(
        seqlen_per_cp=safe_divide(sequence_len, model_repr.parallelism_cfg.cp),
        seqlen_full=sequence_len,
        head_dim=model_repr.head_dim,
        n_q_heads_local=safe_divide(model_repr.n_q_heads, model_repr.parallelism_cfg.tp),
        n_kv_heads_local=max(1, safe_divide(model_repr.n_kv_heads, model_repr.parallelism_cfg.tp)
                             if model_repr.n_kv_heads >= model_repr.parallelism_cfg.tp
                             else model_repr.n_kv_heads),
        micro_bs=model_repr.microbatch_sz,
        machine_spec=machine_spec,
        dtype=model_dtype,
    )

    norm_bwd_time_s = compute_layernorm_bwd_time_s(
        numel=activation_numel, machine_spec=machine_spec, dtype=model_dtype
    )
    rope_bwd_time_s = compute_rope_bwd_time_s(
        numel=activation_numel, machine_spec=machine_spec, dtype=model_dtype
    )
    residual_bwd_time_s = compute_residual_bwd_time_s(
        activation_size=activation_size, machine_spec=machine_spec
    )

    transformer_block_bwd_time_components_dense: dict[str, float] = OrderedDict(
        {
            # Reverse order: backward flows from MLP → Attention.
            # PP activation-grad recv is a per-stage-crossing term (charged once per
            # microbatch below), NOT per layer — see the fwd dense-block note.
            # MLP backward
            "Post MLP Residual (bwd)": residual_bwd_time_s,
            "Post MLP AG (bwd)": ag_time_s,  # fwd RS reverses to AG for grads
            "MLP Down Proj (bwd)": _linear_bwd(model_repr.mlp_down_weight),
            "MLP Up Proj (bwd)": _linear_bwd(model_repr.mlp_up_weight),
            "Pre MLP RS (bwd)": rs_time_s,  # fwd AG reverses to RS for grads
            "Pre MLP Norm (bwd)": norm_bwd_time_s,
            # Attention backward
            "Post Attn Residual (bwd)": residual_bwd_time_s,
            "Post Attn AG (bwd)": ag_time_s,
            "Attn Out Proj (bwd)": _linear_bwd(model_repr.attn_out_weight),
            "SDPA (bwd)": sdpa_bwd_time,
            "QKV Proj (bwd)": _linear_bwd(model_repr.qkv_weight),
            "RoPE (bwd)": rope_bwd_time_s,
            "Pre Attn RS (bwd)": rs_time_s,
            "Pre Attn Norm (bwd)": norm_bwd_time_s,
        }
    )

    # For FULL activation checkpointing, the fwd is re-executed before the bwd
    # (this is what autograd actually does when the forward is wrapped in a
    # no_grad checkpoint). The recompute forward re-runs every block component;
    # PP activation send is no longer a block component (it's a per-step term), so
    # nothing to exclude here.
    if model_repr.act_ckpting_type == ActivationCheckpointingType.FULL:
        recompute_fwd_dense: dict[str, float] = OrderedDict(
            (f"[recompute] {name}", t)
            for name, t in transformer_block_time_components_dense.items()
        )
        transformer_block_bwd_time_components_dense = OrderedDict(
            list(recompute_fwd_dense.items())
            + list(transformer_block_bwd_time_components_dense.items())
        )

    transformer_block_bwd_time_components_moe: dict[str, float] = {}
    if model_repr.moe_cfg is not None:
        assert model_repr.parallelism_cfg.expert_mesh is not None
        assert model_repr.router_weight is not None
        assert model_repr.mlp_up_exp_weight is not None
        assert model_repr.mlp_down_exp_weight is not None

        # Expert backward GEMMs are grouped: each local expert processes its
        # per-device token load (tokens_per_expert_local) through its weight.
        _, *up_dims = model_repr.mlp_up_exp_weight.shape(partitioned=True)
        _, *down_dims = model_repr.mlp_down_exp_weight.shape(partitioned=True)

        if use_measured_grouped:
            mlp_up_exp_bwd_time = grouped_mlp_up_bwd_time_s(
                n_local_experts=n_local_experts,
                tokens_per_expert=tokens_per_expert_local,
                hidden=model_repr.hidden_sz,
                ffn=expert_ffn,
            )
            mlp_down_exp_bwd_time = grouped_mlp_down_bwd_time_s(
                n_local_experts=n_local_experts,
                tokens_per_expert=tokens_per_expert_local,
                hidden=model_repr.hidden_sz,
                ffn=expert_ffn,
            )
        else:
            mlp_up_exp_bwd_time = compute_grouped_linear_bwd_time_s(
                n_tokens_per_group=expert_capacity,
                n_groups=n_local_experts,
                weight_shape=tuple(up_dims),
                machine_spec=machine_spec,
                dtype=model_dtype,
            )
            mlp_down_exp_bwd_time = compute_grouped_linear_bwd_time_s(
                n_tokens_per_group=expert_capacity,
                n_groups=n_local_experts,
                weight_shape=tuple(down_dims),
                machine_spec=machine_spec,
                dtype=model_dtype,
            )
        glu_bwd_time = compute_glu_bwd_time_s(
            numel=n_local_experts * expert_capacity * model_repr.moe_cfg.expert_inter_sz,
            machine_spec=machine_spec,
            dtype=model_dtype,
        )
        topk_bwd_time = compute_topk_bwd_time_s(
            batch=model_repr.microbatch_sz,
            seqlen=safe_divide(model_repr.sequence_len, model_repr.parallelism_cfg.cp),
            n_experts=model_repr.moe_cfg.n_experts,
            machine_spec=machine_spec,
        )
        permute_bwd_time = compute_permutation_bwd_time_s(
            n_tokens=safe_divide(
                model_repr.microbatch_sz * model_repr.sequence_len,
                model_repr.parallelism_cfg.cp,
            ),
            hidden_dim=model_repr.hidden_sz,
            machine_spec=machine_spec,
            dtype=model_dtype,
        )
        expert_tp_ag_time = get_expert_tp_all_gather_comm_time_s(
            size=expert_activation_size,
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )
        expert_tp_rs_time = get_expert_tp_reduce_scatter_comm_time_s(
            size=expert_activation_size,
            parallel_config=model_repr.parallelism_cfg,
            machine_spec=machine_spec,
        )

        # Shared-expert dense MLP backward (all tokens): two linear backwards + GLU
        # backward, mirroring the dense-layer treatment. Only present when the
        # shared expert is configured.
        shared_expert_bwd_components: dict[str, float] = {}
        if model_repr.shared_mlp_up_weight is not None:
            assert model_repr.shared_mlp_down_weight is not None
            shared_expert_bwd_components = {
                "Shared Expert Down Proj (bwd)": _linear_bwd(model_repr.shared_mlp_down_weight),
                "Shared Expert Glu Act (bwd)": compute_glu_bwd_time_s(
                    numel=safe_divide(
                        m_tokens * model_repr.moe_cfg.shared_expert_inter_sz,
                        model_repr.parallelism_cfg.tp,
                    ),
                    machine_spec=machine_spec,
                    dtype=model_dtype,
                ),
                "Shared Expert Up Proj (bwd)": _linear_bwd(model_repr.shared_mlp_up_weight),
            }

        transformer_block_bwd_time_components_moe = OrderedDict(
            {
                # PP activation-grad recv is per-stage-crossing (charged once per
                # microbatch below), NOT per layer — see the fwd dense-block note.
                "Post MLP Residual (bwd)": residual_bwd_time_s,
                # Shared-expert backward (dense, all tokens) — runs alongside the
                # routed-expert backward; reverse order of the fwd shared MLP.
                **shared_expert_bwd_components,
                "Post MLP A2A (bwd)": a2a_time_s,
                "Post MLP AG (bwd)": expert_tp_ag_time,  # swap of fwd expert RS
                "MLP Down Proj (bwd)": mlp_down_exp_bwd_time,
                "Glu Act (bwd)": glu_bwd_time,
                "MLP Up Proj (bwd)": mlp_up_exp_bwd_time,
                "Pre MLP RS (bwd)": expert_tp_rs_time,  # swap of fwd expert AG
                "Pre MLP A2A (bwd)": a2a_time_s,
                "Router Permutation (bwd)": permute_bwd_time,
                "Router TopK (bwd)": topk_bwd_time,
                "Router GEMM (bwd)": _linear_bwd(model_repr.router_weight),
                "Pre MLP Norm (bwd)": norm_bwd_time_s,
                "Post Attn Residual (bwd)": residual_bwd_time_s,
                "Post Attn AG (bwd)": ag_time_s,
                "Attn Out Proj (bwd)": _linear_bwd(model_repr.attn_out_weight),
                "SDPA (bwd)": sdpa_bwd_time,
                "QKV Proj (bwd)": _linear_bwd(model_repr.qkv_weight),
                "RoPE (bwd)": rope_bwd_time_s,
                "Pre Attn RS (bwd)": rs_time_s,
                "Pre Attn Norm (bwd)": norm_bwd_time_s,
            }
        )

        if model_repr.act_ckpting_type == ActivationCheckpointingType.FULL:
            recompute_fwd_moe: dict[str, float] = OrderedDict(
                (f"[recompute] {name}", t)
                for name, t in transformer_block_time_components_moe.items()
            )
            transformer_block_bwd_time_components_moe = OrderedDict(
                list(recompute_fwd_moe.items())
                + list(transformer_block_bwd_time_components_moe.items())
            )

    transformer_block_time_summaries: dict[str, tuple[float, dict[str, float]]] = {}
    transformer_block_bwd_time_summaries: dict[str, tuple[float, dict[str, float]]] = {}
    moe_layer_ratio = model_repr.moe_cfg.moe_frequency if model_repr.moe_cfg else 0
    if moe_layer_ratio < 1:
        transformer_block_time_summaries["Dense"] = (
            1 - moe_layer_ratio,
            transformer_block_time_components_dense,
        )
        transformer_block_bwd_time_summaries["Dense"] = (
            1 - moe_layer_ratio,
            transformer_block_bwd_time_components_dense,
        )
    if moe_layer_ratio > 0:
        transformer_block_time_summaries["MoE"] = (
            moe_layer_ratio,
            transformer_block_time_components_moe,
        )
        transformer_block_bwd_time_summaries["MoE"] = (
            moe_layer_ratio,
            transformer_block_bwd_time_components_moe,
        )

    def _print_component_breakdown(
        header: str,
        components: dict[str, float],
    ) -> float:
        print_h2_header(header)
        total_s = sum(components.values())
        if total_s <= 0:
            print_metric("Total Block Time", "0.00", "ms", highlight=True)
            print()
            return 0.0

        for component_name, component_time_s in components.items():
            time_ms = component_time_s * 1000
            percentage = (component_time_s / total_s) * 100
            color = get_color_for_component_percentage(percentage)
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(
                f"  {component_name.ljust(30)} {color}{time_ms:7.2f} ms{_END}  {percentage:5.1f}%  {_GRAY}{bar}{_END}"
            )

        print()
        print_metric("Total Block Time", f"{total_s * 1000:.2f}", "ms", highlight=True)
        print()
        return total_s

    for block_type, (
        weight,
        transformer_block_time_components,
    ) in transformer_block_time_summaries.items():
        _print_component_breakdown(
            f"TRANSFORMER BLOCK COMPONENTS [FORWARD] ({block_type}, weight={weight})",
            transformer_block_time_components,
        )

    for block_type, (
        weight,
        transformer_block_bwd_time_components,
    ) in transformer_block_bwd_time_summaries.items():
        _print_component_breakdown(
            f"TRANSFORMER BLOCK COMPONENTS [BACKWARD] ({block_type}, weight={weight})",
            transformer_block_bwd_time_components,
        )

    # Explicit autograd-style backward: sum over per-component bwd times instead
    # of applying a forward multiplier. For FULL activation checkpointing, the
    # recompute forward is already folded into the backward component dicts
    # above (entries prefixed with "[recompute]").
    transformer_block_fwd_time = sum(
        weight * sum(components.values())
        for weight, components in transformer_block_time_summaries.values()
    )
    transformer_block_bwd_time = sum(
        weight * sum(components.values())
        for weight, components in transformer_block_bwd_time_summaries.values()
    )
    # GPU-compute time for one microbatch on this rank (all its layers, fwd+bwd).
    compute_per_microbatch_s = layers_per_pp_stage * (
        transformer_block_fwd_time + transformer_block_bwd_time
    )

    # CPU-dispatch floor per microbatch (measured; see kernel_launch). Small
    # models are dispatch-bound (tiny kernels + mbs1 starve the GPU); the
    # per-microbatch wall is max(compute, dispatch), self-limiting so large
    # models (18b) stay compute-bound and the term vanishes. n_local_experts=1
    # for dense models.
    n_local_experts_for_dispatch = (
        safe_divide(model_repr.moe_cfg.n_experts, model_repr.parallelism_cfg.expert_mesh.ep)
        if model_repr.moe_cfg is not None and model_repr.parallelism_cfg.expert_mesh is not None
        else 1
    )
    dispatch_per_microbatch_s = dispatch_time_per_microbatch_s(
        layers_per_stage=layers_per_pp_stage,
        n_local_experts=n_local_experts_for_dispatch,
        device_name=machine_spec.name,
        ep=(
            model_repr.parallelism_cfg.expert_mesh.ep
            if model_repr.parallelism_cfg.expert_mesh is not None
            else 1
        ),
        moe_frequency=(
            model_repr.moe_cfg.moe_frequency if model_repr.moe_cfg is not None else 0.0
        ),
    )
    wall_per_microbatch_s = max(compute_per_microbatch_s, dispatch_per_microbatch_s)

    transformer_block_time = n_microbatches_per_mp_rank * wall_per_microbatch_s
    # Bubble is (pp-1)/(vpp*n_mb) of the per-rank wall (whichever bound applies).
    pipeline_bubble_time = transformer_block_time * pipeline_bubble_fraction

    # PP send/recv is a per-STAGE-CROSSING p2p, not a per-layer op. It is charged
    # per crossing, NOT × layers_per_pp_stage. With pp==1 there is no boundary.
    #   * non-interleaved 1F1B (vpp==1): a microbatch enters/leaves each rank once,
    #     so 2 crossings/microbatch (1 fwd activation send + 1 bwd grad recv).
    #   * interleaved 1F1B (vpp>1): each rank owns `vpp` virtual chunks, so a
    #     microbatch enters/leaves each rank `vpp` times -> 2*vpp crossings, each
    #     transferring one microbatch's boundary activation (same size). PP p2p
    #     therefore scales with vpp. (The 1/vpp steady-state bubble reduction above
    #     is the flip side: interleaving shrinks the bubble but multiplies p2p.)
    # (The measured meas_pp_send_recv bucket folds in pipeline-bubble/desync wait
    # and so runs 100–1000× larger; that stochastic gap is Task 04's non-goal — we
    # model only the wire-time p2p here. See test_comm_cost_audit.)
    pp_send_recv_time_s = (
        n_microbatches_per_mp_rank * vpp * 2 * activation_send_time_s
        if model_repr.parallelism_cfg.pp > 1
        else 0.0
    )

    # Per-step EP all-to-all wire time (for the comm-cost audit). Each MoE block
    # does 2 all-to-alls in the forward (dispatch + combine) and 2 in the backward,
    # over every MoE-layer execution this rank runs in a step:
    #   n_microbatches_per_mp_rank * (moe layers on this stage) * 4 * a2a_time_s.
    # a2a_time_s exists only on the MoE path. NOTE: with FULL activation
    # checkpointing the recompute-forward re-issues the 2 fwd a2a's (6 total, not 4);
    # this diagnostic uses 4 (no golden config is FULL+MoE) — a documented
    # simplification of the print-only audit line, not of iteration time.
    n_moe_layers_per_stage = layers_per_pp_stage * (
        model_repr.moe_cfg.moe_frequency if model_repr.moe_cfg is not None else 0.0
    )
    ep_all_to_all_time_s = (
        n_microbatches_per_mp_rank * n_moe_layers_per_stage * 4 * a2a_time_s
        if model_repr.moe_cfg is not None
        else 0.0
    )

    # ------------------------------------------------------------------
    # Embedding (first stage) + LM head & cross-entropy (last stage).
    #
    # Megatron places these ON TOP of the first/last stages' transformer
    # layers; no other stage has them. In 1F1B the slowest stage paces the
    # steady state, so the last-stage surplus is paid once per microbatch
    # ((n_mb-1)*max(first,last) + first + last — exact decomposition, verified
    # against the 1F1B dependency-graph simulation in test_pipeline_bubble).
    # This was the dominant unmodeled term in the 18b pp16 high-n_mb traces
    # (bc59887c: ~24s PP-recv wait ≈ 87 ms/microbatch of last-stage pacing).
    #
    # Last stage per microbatch:
    #   * LM-head GEMM fwd: (n_tokens x hidden) @ (hidden x vocab_padded/tp)
    #   * its autograd bwd (dX + dW GEMMs)
    #   * NO recompute under FULL checkpointing — Megatron checkpoints
    #     transformer layers only; the post_process LM-head/CE block is not
    #     checkpointed.
    #   * CE + softmax elementwise: a few passes over the (n_tokens x
    #     vocab_padded/tp) fp32 logits, HBM-bandwidth-bound. 4 passes total:
    #     softmax read+write in fwd, grad-of-logits read+write in bwd.
    # First stage per microbatch:
    #   * embedding fwd gather + bwd scatter-add: HBM traffic of
    #     ~2 x n_tokens x hidden x bytes (read row + write activation; bwd
    #     accumulates grads) — negligible FLOPs, bandwidth-bound.
    lm_head_fwd_time_s = compute_gemm_time_s(model_repr.embed_weight)
    lm_head_bwd_time_s = _linear_bwd(model_repr.embed_weight)
    vocab_local = safe_divide(model_repr.vocab_sz_padded, model_repr.parallelism_cfg.tp)
    if vocab_ce_measured(machine_spec.name):
        # Measured fused vocab-parallel CE fwd+bwd (softmax + logsumexp +
        # grad-of-logits, all fp32) — ~19.6 effective HBM passes, ~5x the naive
        # 4-pass estimate. See vocab_ce_util / vocab_gemm_ce benchmark.
        ce_softmax_time_s = cross_entropy_fwd_bwd_time_s(
            m_tokens=m_tokens, vocab_padded=vocab_local
        )
    else:
        # fp32 logits (Megatron upcasts for the softmax/CE), 4 HBM passes.
        ce_softmax_time_s = (
            4 * m_tokens * vocab_local * 4 / machine_spec.device_spec.mem_bandwidth_bytes_per_sec
        )
    t_last_extra_s = lm_head_fwd_time_s + lm_head_bwd_time_s + ce_softmax_time_s
    # embedding: fwd gather (read m_tokens rows + write activation) and bwd
    # scatter-add (read+write grad rows) ≈ 4 activation-sized HBM passes.
    embedding_time_s = (
        4 * m_tokens * model_repr.hidden_sz * (model_repr.bits_per_parameter / 8)
    ) / machine_spec.device_spec.mem_bandwidth_bytes_per_sec
    t_first_extra_s = embedding_time_s

    vocab_stage_extra_time_s = compute_stage_imbalance_extra_time_s(
        pp=model_repr.parallelism_cfg.pp,
        n_microbatches=n_microbatches_per_mp_rank,
        t_first_extra_s=t_first_extra_s,
        t_last_extra_s=t_last_extra_s,
    )
    # The heavy last stage (LM head) also lengthens the pipeline fill/drain ramp:
    # 1F1B is paced by the SLOWEST stage, so the (pp-1)/(vpp*n_mb) bubble applies
    # to the last-stage wall (compute + LM head), not just the balanced compute.
    # This extra ramp term is what makes a deep pipeline (pp16) with a fat LM-head
    # last stage stall far more than the balanced bubble predicts, while a shallow
    # pipeline (pp8, large n_mb) sees it vanish — verified against a per-stage 1F1B
    # discrete-event simulation and the 18b traces. Uses t_last (the dominant
    # extra); embedding on the first stage is negligible beside the LM head.
    pipeline_bubble_time += t_last_extra_s * n_microbatches_per_mp_rank * pipeline_bubble_fraction

    # Model DP communication overlap efficiency
    # Physics: DP communication can overlap with compute, but the first pipeline stage
    # has no backward compute to overlap with after completing its microbatch.
    # For other stages, overlap depends on whether compute windows are sufficient
    # to hide communication time per bucket.
    # NOTE: This is an approximation when using EP, where there will be separate
    # reduction groups for DP_exp and DP_nonexp.

    # Microbatch compute time for overlap calculation.
    # Use the explicit per-component fwd/bwd totals rather than the
    # "FLOPs/peak" shortcut so overlap modeling sees the same time basis as
    # the iteration-time accounting below.
    microbatch_compute_time = layers_per_pp_stage * (
        transformer_block_fwd_time + transformer_block_bwd_time
    )
    pp_degree = model_repr.parallelism_cfg.pp

    # Default exposed times to 0; the branch below fills in whichever path applies.
    exposed_dp_ag_time = 0.0
    exposed_dp_rs_time = 0.0
    exposed_dp_ar_time = 0.0

    if uses_all_reduce:
        # zero_level=NONE: a single all-reduce replaces reduce-scatter + all-gather.
        # Only the backward-overlap window is available (grads are produced during
        # backward and all-reduced as buckets fill up). Use the same overlap model
        # as reduce-scatter.
        ar_time_per_bucket = (
            cross_dc_grad_bucket_ar_time_s
            if cross_dc_config is not None and cross_dc_grad_bucket_ar_time_s is not None
            else grad_bucket_all_reduce_time_s
        )
        exposed_dp_ar_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=ar_time_per_bucket,
            n_buckets=n_buckets,
            microbatch_compute_time=microbatch_compute_time,
            pp_degree=pp_degree,
        )
    elif cross_dc_config is not None:
        assert cross_dc_param_bucket_ag_time_s is not None
        assert cross_dc_grad_bucket_rs_time_s is not None
        # Calculate exposed time using overlap efficiency model
        exposed_dp_ag_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=cross_dc_param_bucket_ag_time_s,
            n_buckets=n_buckets,
            microbatch_compute_time=microbatch_compute_time,
            pp_degree=pp_degree,
        )
        exposed_dp_rs_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=cross_dc_grad_bucket_rs_time_s,
            n_buckets=n_buckets,
            microbatch_compute_time=microbatch_compute_time,
            pp_degree=pp_degree,
        )
    else:
        # Calculate exposed time using overlap efficiency model
        exposed_dp_ag_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=param_bucket_all_gather_time_s,
            n_buckets=n_buckets,
            microbatch_compute_time=microbatch_compute_time,
            pp_degree=pp_degree,
        )
        exposed_dp_rs_time = calculate_exposed_dp_time(
            dp_comm_time_per_bucket=grad_bucket_reduce_scatter_time_s,
            n_buckets=n_buckets,
            microbatch_compute_time=microbatch_compute_time,
            pp_degree=pp_degree,
        )

    # Optimizer step time = memory bandwidth time + computation time
    # Memory bandwidth: time to read/write optimizer states to/from HBM
    opt_step_memory_time_s = (
        2
        * model_repr.states.total_bytes(partitioned=True)
        / machine_spec.device_spec.mem_bandwidth_bytes_per_sec
    )

    # Computation time: FLOPs for optimizer operations
    opt_step_flops = model_repr.get_optimizer_step_flops()
    opt_step_compute_time_s = opt_step_flops / machine_spec.device_spec.peak_flops(model_dtype)

    opt_step_time_s = opt_step_memory_time_s + opt_step_compute_time_s

    # Calculate kernel launch overhead
    # Model the cumulative CPU-GPU synchronization overhead across all kernel launches
    has_tp = model_repr.parallelism_cfg.tp > 1
    has_moe = model_repr.moe_cfg is not None
    n_experts_active = (
        model_repr.moe_cfg.experts_per_token if model_repr.moe_cfg is not None else 0
    )

    kernels_per_block = estimate_kernel_count_per_transformer_block(
        has_tp=has_tp,
        has_moe=has_moe,
        n_experts_active=n_experts_active,
    )
    # NOTE: kernel-launch overhead is now folded into the per-microbatch
    # max(compute, dispatch) wall (dispatch_time_per_microbatch_s), which is the
    # physically correct self-limiting form. This standalone additive term is
    # kept at 0 to avoid double-counting; the helper is retained for reference.
    _ = calculate_kernel_launch_overhead(
        n_layers=model_repr.n_layers,
        kernels_per_block=kernels_per_block,
        device_name=machine_spec.name,
    )
    kernel_launch_overhead_s = 0.0

    if uses_all_reduce:
        iteration_time_components: dict[str, float] = OrderedDict(
            {
                "Transformer Block": transformer_block_time,
                "Embedding + LM Head/CE": vocab_stage_extra_time_s,
                "DP All-Reduce (Exposed)": exposed_dp_ar_time,
                "PP Send/Recv": pp_send_recv_time_s,
                "Pipeline Bubble": pipeline_bubble_time,
                "Optimizer Step": opt_step_time_s,
                "Kernel Launch Overhead": kernel_launch_overhead_s,
            }
        )
    else:
        iteration_time_components = OrderedDict(
            {
                "Transformer Block": transformer_block_time,
                "Embedding + LM Head/CE": vocab_stage_extra_time_s,
                "DP All-Gather (Exposed)": exposed_dp_ag_time,
                "DP Reduce-Scatter (Exposed)": exposed_dp_rs_time,
                "PP Send/Recv": pp_send_recv_time_s,
                "Pipeline Bubble": pipeline_bubble_time,
                "Optimizer Step": opt_step_time_s,
                "Kernel Launch Overhead": kernel_launch_overhead_s,
            }
        )

    print_h2_header("ITERATION TIME COMPONENTS")
    iteration_time_s = sum(iteration_time_components.values())

    # Empirical cross-node desync/straggler penalty (the model's ONE fitted term;
    # see utils/desync). Multi-node EP/PP groups run slower than the
    # straggler-free analytical step by an amount that grows with the inter-node
    # boundaries each dimension crosses; single-node groups incur nothing.
    _expert_tp = (
        model_repr.parallelism_cfg.expert_mesh.tp
        if model_repr.parallelism_cfg.expert_mesh is not None
        else 1
    )
    _ep_for_desync = (
        model_repr.parallelism_cfg.expert_mesh.ep
        if model_repr.parallelism_cfg.expert_mesh is not None
        else 1
    )
    desync_multiplier = cross_node_desync_multiplier(
        pp=model_repr.parallelism_cfg.pp,
        ep=_ep_for_desync,
        expert_tp=_expert_tp,
        gpus_per_node=machine_spec.n_devices,
        ep_a2a_time_s=ep_all_to_all_time_s,
        iteration_time_s=iteration_time_s,
    )
    iteration_time_s *= desync_multiplier

    # Sort components by time (descending) for better readability
    sorted_iteration_components = sorted(
        iteration_time_components.items(), key=lambda x: x[1], reverse=True
    )

    for component_name, component_time_s in sorted_iteration_components:
        time_ms = component_time_s * 1000
        percentage = (component_time_s / iteration_time_s) * 100

        # Use color coding based on percentage
        color = get_color_by_percentage(percentage)

        # Create a simple bar chart
        bar_length = int(percentage / 2)  # Scale to max 50 chars
        bar = "█" * bar_length

        print(
            f"  {component_name.ljust(20)} {color}{time_ms:8.2f} ms{_END}  {percentage:5.1f}%  {_GRAY}{bar}{_END}"
        )

    print()

    # Per-step collective wire-time totals — the audit target (test_comm_cost_audit).
    # These are the modeled STRAGGLER-FREE wire times per collective family. The
    # measured buckets (meas_allreduce/allgather/pp/ep) fold in desync + bubble wait
    # and run larger; we report the comparison and gate only the wire-time invariants
    # (never tune bandwidth to the noisy buckets — GUIDELINES §1, Task 04 non-goal).
    dp_comm_step_ms = _dp_comm_total_step_ms
    print_h2_header("PER-STEP COLLECTIVE WIRE TIME (audit)")
    print_kv("Step DP Comm", f"{dp_comm_step_ms:.3f} ms", key_width=30)
    print_kv("Step PP Send/Recv", f"{pp_send_recv_time_s * 1000:.3f} ms", key_width=30)
    print_kv("Step EP All-to-All", f"{ep_all_to_all_time_s * 1000:.3f} ms", key_width=30)
    print()

    print_h2_header("FINAL RESULTS")
    print()
    print_metric("Iteration Time", f"{iteration_time_s:.2f}", "seconds", highlight=True)
    tokens_per_day = ((gbs * sequence_len) / iteration_time_s) * 60 * 60 * 24
    print_metric("Tokens per Day (w/ 100% Goodput)", f"{tokens_per_day / 1e9:.2f}", unit="B")

    ideal_iteration_time = (
        gbs
        * sequence_len
        * 6
        * model_repr.get_n_active_params(partitioned=False)
        / (cluster_size * machine_spec.device_spec.peak_flops(model_dtype))
    )
    print_metric("Ideal Iteration Time", f"{ideal_iteration_time:.5f}", "seconds")

    mfu_percentage = (ideal_iteration_time / iteration_time_s) * 100
    print()
    print_success(f"Theoretical MFU: {mfu_percentage:.2f}%")
    print()

    return TrainingMetrics(
        mfu_pct=float(mfu_percentage),
        iteration_time_s=float(iteration_time_s),
        memory_per_device_gb=float(total_memory_gib),
    )


def main() -> None:
    # First docstring line only — the rest documents the module for callers, and
    # ArgumentParser's first positional is `prog` (it shows up in `3dtrn --help`).
    parser = ArgumentParser(__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    calculate_training_metrics(cfg, verbose=True)


if __name__ == "__main__":
    main()
