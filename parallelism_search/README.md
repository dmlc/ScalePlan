# Parallelism Search

Automated grid search for optimal parallelism configurations in 3D distributed training.

## Features

- **Exhaustive Search**: Evaluates all valid combinations of TP, EP, PP, CP, and DP
- **Memory Constraints**: Filters configurations exceeding device memory limits
- **MFU Estimation**: Uses physics-based modeling matching `training_3d.py`
- **MLflow Integration**: Compare analytical predictions with benchmark data

## Usage

### 1. Create a search configuration file

See `example_configs/` for example configurations (e.g., `example_configs/config-p6.yaml`).

```yaml
model:
  n_layers: 48
  hidden_sz: 6144
  # ... other model parameters
  moe:
    n_experts: 128
    experts_per_token: 3
    # ... MoE parameters

search:
  num_devices: 512

performance:
  activation_checkpointing_type: none

optimizer:
  optimizer_type: muon

data:
  gbs: 8192
  seqlen: 8192
  microbatch_sz: 4

hardware:
  node_type: p6-b200.48xlarge
```

### 2. Run the search

```bash
python parallelism_search/search_entry.py example_configs/config-p6.yaml --devices 512 --top-k 10 --max-tp 1 --max-mbs 4
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `config_path` | Path to search configuration YAML file (required) |
| `--devices N` | Number of devices to search over (default: 512) |
| `--top-k N` | Number of top configurations to show (default: 100) |
| `--max-tp N` | Maximum tensor parallelism (searches 1 to N) |
| `--max-cp N` | Maximum context parallelism (searches 1 to N) |
| `--max-ep N` | Maximum expert parallelism (searches 1 to N, default: 32) |
| `--max-etp N` | Maximum expert tensor parallelism (searches divisors of tp up to N, default: etp=tp) |
| `--max-mbs N` | Maximum microbatch size (searches 1 to N, default: 1) |
| `--activation-checkpointing TYPE` | Activation checkpointing: `none`, `selective`, `full`, or `all` to search (default: all) |
| `--memory-limit-fraction F` | Fraction of device memory to use as limit (default: 0.9) |
| `--mlflow-experiment-id ID` | MLflow experiment ID for benchmark comparison |
| `--no-save` | Don't save results to files |
| `--output-name NAME` | Custom base name for output files |

### Examples

```bash
# Basic search
python parallelism_search/search_entry.py config.yaml

# Limit TP to max 4 and CP to max 2
python parallelism_search/search_entry.py config.yaml --max-tp 4 --max-cp 2

# Use 70% memory limit instead of 90%
python parallelism_search/search_entry.py config.yaml --memory-limit-fraction 0.7

# With MLflow benchmark comparison
python parallelism_search/search_entry.py config.yaml --mlflow-experiment-id 31340
```

### 3. Results

Output files are saved to `parallelism_search/outputs_v1/`:
- `{config_name}.txt`: Human-readable summary
- `{config_name}.json`: Machine-readable data
- `{config_name}_best.yaml`: Best configuration for copy-paste

## Constraints

The search enforces:
- `world_size = tp × pp × cp × dp = num_devices`
- `dp >= 1`
- `n_layers % pp == 0`
- `n_experts % ep == 0`
- `memory_per_device < memory_limit_fraction × hardware_limit`

## Files

| File | Description |
|------|-------------|
| `searcher.py` | Main search implementation |
| `search_entry.py` | Command-line interface |
| `training_calculator.py` | Thin adapter over the canonical cost model (`dlcalc.training_3d.calculate_training_metrics`) |
| `plot_mfu_comparison.py` | Visualization for analytical vs benchmark MFU |

## Example Output

```
============================================================
MODEL INFORMATION
============================================================
  Total Parameters:  440.68B
  Active Parameters: 16.00B
  Layers: 48
  Hidden Size: 6144
  MoE Experts: 128
  Experts per Token: 3
============================================================

TOP PARALLELISM CONFIGURATIONS
================================================================================

Rank 1:
  Configuration: tp=1, ep=16, pp=4, cp=4, dp=32
  Analytical MFU: 5.58%
  Memory per device: 165.66 GB
  Iteration time: 50.156 s
```
