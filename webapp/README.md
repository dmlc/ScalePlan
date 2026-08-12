# Performance Modeling Studio (Web App)

Interactive single-page UI for exploring 3D-parallelism configurations: drag sliders over the model/data/hardware/parallelism knobs and watch MFU, iteration time, throughput, and per-device memory update live. Includes sweep plots (MFU + memory vs `tp`, `pp`), 2-D MFU heatmaps (`tp × ep`, `pp × ep`), and a one-click grid search that returns the top-K configurations.

The backend wraps the same `parallelism_search.training_calculator.calculate_training_metrics` and `parallelism_search.searcher.ParallelismSearcher` used by the CLI — the UI is just a thin presentation layer on top.

## Run a local server

From the repo root (`performance-modeling/`):

```bash
uv run python webapp/server.py
```

Then open http://127.0.0.1:5057 in your browser.

To use a different port:

```bash
PORT=8080 uv run python webapp/server.py
```

## What's in the UI

- **Base config picker** (top): load any YAML from `example_configs/`. "Reset to file" reverts every slider to the file's pristine values.
- **Model panel**: `n_layers`, `hidden_sz`, `inter_sz`, attention heads, `vocab_sz`, precision, GLU/RoPE/dropout/tie_embeddings toggles, and a full **MoE block** (`n_experts`, `experts_per_token`, `expert_inter_sz`, `capacity_factor`, `moe_frequency`) with an enable/disable switch.
- **Data & Optimizer panel**: `gbs`, `seqlen`, `microbatch_sz`, optimizer (`adam`/`muon`), activation checkpointing strategy.
- **Hardware panel**: dropdown over every node type defined in `dlcalc.utils.hardware.MachineSpec` (`p4d`, `p4de`, `p5`, `p6-b200`, `trn1n`), plus a live spec readout (HBM, peak FLOPs at the chosen precision, intra/inter bandwidth) and a `num_devices` slider.
- **Parallelism panel**: `tp / pp / cp` dropdowns showing only valid divisors, **derived `dp`** with the formula, `ep` (capped to `n_experts`, divisor-only), `etp` (divisors of `tp`), `zero_level`, `n_param_buckets`, `sequence_parallel`. Live "tp · pp · cp · dp = num_devices" readout.
- **Live Results** (sticky on the right; debounced ~120 ms after any change):
  - MFU, iteration time, throughput (tokens/sec and tokens/day).
  - **Memory / device card with a colored bar** — green < 85%, amber > 85%, **pulsing red > 100%** of HBM.
  - **Sweep plots**: MFU + memory vs `tp` (at `pp=1`) and vs `pp` (at the current `tp`); memory bars are colored against the device limit.
  - **MFU heatmaps**: `tp × ep` (at `pp=1, cp=1`) and `pp × ep` (at the current `tp, cp=1`). Cells colored on a green ramp; OOM cells red; best cell outlined in blue. **Click any cell to apply that configuration.**
- **Find Optimal Configuration** (full-width below): runs the real `ParallelismSearcher` with `top_k`, `max_ep/tp/cp/mbs`, activation-checkpointing strategy, and memory-limit fraction. The ranked table has an **Apply** button on each row that pushes the configuration back into the sliders.

## API endpoints

Useful if you want to script the calculator instead of clicking:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/meta` | GET | Available configs, node types, precisions, optimizer types |
| `/api/config/<name>` | GET | Returns parsed YAML for one of `example_configs/*.yaml` |
| `/api/evaluate` | POST | Evaluate a single `(model, data, hardware, parallelism)` config |
| `/api/search` | POST | Run a grid search and return top-K configurations |

`/api/evaluate` example:

```bash
curl -s http://127.0.0.1:5057/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": {...},
    "data": {"gbs": 96, "seqlen": 8192, "microbatch_sz": 1},
    "hardware": {"node_type": "p6-b200.48xlarge"},
    "performance": {"activation_checkpointing_type": "selective"},
    "optimizer": {"optimizer_type": "muon"},
    "parallelism": {"tp": 1, "pp": 1, "cp": 1, "dp": 96, "ep": 8, "etp": 1,
                    "vpp": 1, "sp": true, "zero_level": 1, "n_param_buckets": 5}
  }'
```

## Layout

```
webapp/
├── server.py            # Flask backend wrapping calculate_training_metrics + ParallelismSearcher
├── README.md            # this file
└── static/
    ├── index.html       # single-page UI
    ├── style.css
    └── app.js           # state, controls, live evaluation, sweeps, heatmaps, search
```

No build step — the frontend is plain HTML/CSS/JS plus a CDN-loaded Chart.js. Edit the files and reload the page.
