# MFU Calculation Technical Guide

This guide explains how this library calculates analytical MFU (Model FLOPs Utilization) predictions for distributed training with 3D parallelism.

## Table of Contents

1. [Overview](#overview)
2. [Core MFU Formula](#core-mfu-formula)
3. [Component Breakdown](#component-breakdown)
4. [Overhead Factors](#overhead-factors)
5. [Validation Methodology](#validation-methodology)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The MFU calculation models the entire training iteration, accounting for:
- **Compute time**: GEMM operations, attention, norms, activations
- **Communication time**: TP, DP, PP, EP collectives
- **Overlap effects**: DP communication overlapping with compute
- **Pipeline bubbles**: Fill/drain phases and steady-state inefficiency
- **Launch overhead**: CUDA kernel launch latency

**Key principle:** All models are physics-based or algorithm-based. No arbitrary multipliers.

---

## Core MFU Formula

```
MFU = (Model FLOPs per iteration) / (Peak Device FLOPs × Iteration Time)

where:
  Model FLOPs = Forward + Backward compute (attention + MLP GEMMs)
  Peak Device FLOPs = Hardware peak (e.g., 989 TFLOPS BF16 for H100)
  Iteration Time = Compute + Communication + Bubbles + Overhead
```

---

## Component Breakdown

### 1. Transformer Block Compute Time

Each transformer block consists of:

#### Attention Block
```
Pre Attn Norm:     LayerNorm timing from lookup table (norm_rope_util.py)
RoPE:              Rotary Position Embedding from lookup table
Pre Attn AG:       TP All-Gather for Q, K, V projections
QKV Proj:          GEMM [batch*seq, hidden] @ [hidden, 3*hidden]
SDPA:              Scaled Dot-Product Attention (FlashAttention timing)
Attn Out Proj:     GEMM [batch*seq, hidden] @ [hidden, hidden]
Post Attn RS:      TP Reduce-Scatter after attention
Post Attn Residual: Residual addition (memory-bound)
```

#### MLP Block
```
Pre MLP Norm:      LayerNorm timing from lookup table
Pre MLP AG:        TP All-Gather for MLP input
MLP Up Proj:       GEMM [batch*seq, hidden] @ [hidden, ffn_hidden]
MLP Down Proj:     GEMM [batch*seq, ffn_hidden] @ [ffn_hidden, hidden]
Post MLP RS:       TP Reduce-Scatter after MLP
Post MLP Residual: Residual addition
```

#### MoE Block (if applicable)
```
Router GEMM:       [batch*seq, hidden] @ [hidden, n_experts]
Router TopK:       TopK selection (k=2 typically)
Router Permutation: Token reordering for expert processing
Pre MLP A2A:       All-to-All for expert parallelism
Expert GEMMs:      Per-expert Up/Down projections
Post MLP A2A:      All-to-All gather from experts
```

### 2. GEMM Timing Model

```python
def compute_gemm_time_s(m: int, n: int, k: int, dtype: DType) -> float:
    """
    GEMM: C[m,n] = A[m,k] @ B[k,n]

    FLOPs = 2 * m * n * k  (multiply-accumulate)
    Utilization = lookup from gemm_util.parquet based on (m,n,k,device,dtype)
    Time = FLOPs / (Peak FLOPs * Utilization)
    """
```

**Key insight:** GEMM utilization varies significantly with problem size. Small GEMMs (e.g., router) may achieve only 30-40% of peak, while large GEMMs can reach 90%+.

### 3. Communication Timing Model

Communication follows **ring algorithm** for TP/DP collectives:

```
Ring All-Gather time = Latency × (N-1) + BW × (N-1)/N

where:
  Latency = hop latency × protocol multiplier
  BW = (message_size / bandwidth) × protocol efficiency
  N = degree of parallelism
```

**Protocol selection** (NCCL):
- LL (Low Latency): <32KB, 50% BW efficiency, 2.0× latency
- LL128: 32KB-1MB, 70% BW efficiency, 1.5× latency
- Simple: >1MB, 90% BW efficiency, 1.0× latency

**Chunking overhead:** Messages >512KB are chunked, adding ~2μs per additional chunk.

---

## Overhead Factors

### 1. DP Communication Overlap Efficiency

```python
def calculate_dp_overlap_efficiency(
    microbatch_compute_time: float,
    comm_time_per_bucket: float,
    n_buckets: int,
    is_first_pp_stage: bool
) -> float:
    """
    Physics: DP communication can only overlap during compute windows.

    First PP stage: 0% overlap (no backward compute to overlap with)
    Other stages: min(comm_time, compute_time) overlaps
    """
```

**Impact:** First pipeline stage can see 2-4% lower MFU due to exposed DP time.

### 2. Kernel Launch Overhead

```python
def calculate_kernel_launch_overhead(
    n_layers: int,
    kernels_per_block: int,
    device: str
) -> float:
    """
    Launch latency per kernel:
      A100: ~8μs
      H100/B200: ~5μs

    Total overhead = (2 × n_layers × kernels_per_block) × launch_latency
    Factor of 2 accounts for forward + backward passes
    """
```

**Impact:** For 80-layer model with TP, ~12ms overhead (~1-2% of iteration time).

### 3. Pipeline Bubble

```python
def calculate_pipeline_bubble(
    pp: int,
    vpp: int,
    n_microbatches: int
) -> float:
    """
    Three phases:
    1. Fill: First (PP-1) microbatches fill pipeline
    2. Steady-state: 1F1B scheduling with bubble = (PP-1)/(VPP * n_microbatches)
    3. Drain: Last (PP-1) microbatches drain pipeline

    Fill/drain phases have higher bubble: (PP-1)/PP
    VPP only reduces steady-state bubble, not fill/drain
    """
```

**Impact:** Most significant for small microbatch counts. For PP=4, n=2048: ~0.37% bubble.

### 4. Norm/RoPE Kernel Timing

Lookup tables replace rough HBM approximation:

```
Old approximation: 2 × data_size / memory_bandwidth
New lookup: empirical_time(batch, seqlen, hidden_dim, dtype, device)

Typical improvement:
  LayerNorm: ~35% more accurate
  RoPE: ~45% more accurate
```

Fallback to bandwidth estimation when lookup unavailable.

---

## Validation Methodology

### Data Sources

1. **MLFlow Metrics**: Real training job MFU via `mlflow_collector.py`
   - Experiment IDs with diverse configurations
   - Metrics: MFU, batch_time, samples/sec, load_balancing_loss

2. **GPU Traces**: Profiling data via `profile_parse/` tools
   - CUDA kernel timing breakdown
   - Communication operation timing
   - Overlap efficiency measurements

3. **Benchmark Data**: Empirical lookup tables
   - `benchmarks/results/gemm_util.parquet`: GEMM utilization curves (H100/B200)
   - `benchmarks/results/sdpa.parquet`: FlashAttention timing (8000+ measurements)
   - `norm_rope_timings.parquet`: LayerNorm/RoPE timing

### Error Metrics

```python
MAPE = (1/n) × Σ |predicted_mfu - actual_mfu| / actual_mfu × 100%
MAE = (1/n) × Σ |predicted_mfu - actual_mfu|
RMSE = sqrt((1/n) × Σ (predicted_mfu - actual_mfu)²)
```

**Baseline:** 5-15% MAPE before improvements
**Target:** <5% MAPE (stretch: <3%)

### Validation Framework

```bash
# Run full validation suite
python validation/run_regression_tests.py

# Generate error report
python validation/generate_error_report.py --output report.html
```

Error analysis breakdown by:
- PP degree (1, 2, 4, 8)
- DP degree (1, 2, 4, 8, 16)
- TP degree (1, 2, 4, 8)
- EP degree (1, 2, 4, 8)
- Model size and dtype

---

## Troubleshooting

### MFU Prediction Too High

**Symptom:** Analytical MFU > Actual MFU by >10%

**Possible causes:**
1. **Framework overhead not modeled**
   - CPU bottlenecks (data loading, preprocessing)
   - Python GIL contention
   - Synchronization overhead

   *Solution:* Check CPU utilization and profiling traces

2. **Communication overhead underestimated**
   - Network congestion or stragglers
   - Suboptimal routing or topology

   *Solution:* Validate actual comm times against model predictions

3. **Memory bandwidth saturation**
   - Memory-bound operations slower than expected
   - Memory fragmentation

   *Solution:* Compare actual HBM bandwidth vs. peak

4. **Insufficient microbatch count**
   - Pipeline bubble larger than modeled
   - Fill/drain phases dominating

   *Solution:* Increase global batch size or reduce microbatch size

### MFU Prediction Too Low

**Symptom:** Analytical MFU < Actual MFU

**Possible causes:**
1. **Compiler optimizations not modeled**
   - Kernel fusion reducing operations
   - Memory access optimizations

   *Solution:* This is rare but acceptable (conservative estimate)

2. **Communication overlapped more than modeled**
   - Framework-specific optimizations
   - Better overlap scheduling

   *Solution:* Profile to verify overlap efficiency

### Large Error on Specific Configurations

**High DP, Low PP configurations:**
- Check DP overlap efficiency
- Verify first pipeline stage exposure

**MoE configurations:**
- Validate router operation timing
- Check load balancing metrics
- Verify expert parallelism communication

**Small microbatch counts:**
- Validate pipeline bubble calculation
- Check fill/drain phase impact

**FP8 training:**
- Ensure FP8 GEMM utilization data available
- Check mixed-precision overhead

---

## References

### Code Files
- `dlcalc/training_3d.py`: Main MFU calculation logic
- `dlcalc/utils/comms.py`: Communication modeling
- `dlcalc/utils/gemm_util.py`: GEMM utilization lookups
- `dlcalc/utils/norm_rope_util.py`: Norm/RoPE timing lookups
- `dlcalc/utils/kernel_launch.py`: Launch overhead modeling
- `dlcalc/utils/overlap.py`: DP overlap efficiency
- `dlcalc/utils/moe_router_util.py`: MoE router operations

### Theoretical Background
- Ring algorithm: NCCL source code and documentation
- Pipeline parallelism: Megatron-LM paper, GPipe paper
- 1F1B scheduling: PipeDream paper
- NCCL protocols: https://github.com/NVIDIA/nccl/issues/281

### Validation Tools
- `validation/mfu_error_tracker.py`: Error calculation and tracking
- `validation/mlflow_collector.py`: MLFlow metrics collection
- `profile_parse/`: GPU trace parsing utilities

---

## Change History

See `CHANGELOG.md` for detailed version history and improvement tracking.

**February 2026:** Major accuracy improvements
- Added DP overlap efficiency modeling (Task 1)
- Added NCCL protocol overhead modeling (Task 2)
- Added kernel launch overhead modeling (Task 3)
- Added Norm/RoPE empirical timing (Task 4)
- Added MoE router operation timing (Task 5)
- Added pipeline fill/drain phases (Task 6)

All improvements validated against real training jobs and GPU profiling data.
