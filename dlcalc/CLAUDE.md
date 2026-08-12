# CLAUDE.md

## Project Overview
dlcalc is a Python CLI toolkit for deep learning performance calculations and optimization, focused on transformer model training with 3D parallelism (data, tensor, and pipeline parallelism). The package provides command-line tools for performance modeling, topology optimization, and training metrics calculations.

## Architecture and Key Components

### Core Modules Structure
- **dlcalc/training_3d.py**: Main 3D parallelism calculator implementing performance modeling equations for transformer training. Contains the core logic for memory, compute, and communication calculations.
- **dlcalc/topology_visualizer.py**: Kubernetes topology visualization using pyvis for network graph generation
- **dlcalc/topology_evaluate.py**: Evaluates network topologies for optimal training job placement
- **dlcalc/topology_scheduler.py**: Implements topology-aware scheduling algorithms for distributed training
- **dlcalc/utils/**: Shared utilities for parameter parsing and common calculations

### CLI Entry Points
All CLI commands are defined in `pyproject.toml` under `[project.scripts]`:
- `3dtrn`: Performance calculator for 3D parallel training
- `topoviz`, `topoeval`, `topoassign`: Topology optimization tools
- `sps2tpd`, `sps2mfu`: Metric conversion utilities
- `ckpt-summarize`: PyTorch checkpoint analysis

### Configuration System
- Hardware specifications stored in `configs/` as YAML files (e.g., `configs/a100_40gb.yaml`)
- Example training configurations in `examples/` directory
- Configuration loading handled through PyYAML with custom parameter parsing in utils module

## Development Tools
Use the following tools:
* `uv` (NEVER use pip)
* `ruff`
* `mypy`
* `pre-commit`

## Development Philosophy
* Test Driven Development: When you are asked to implement a feature or fix a bug, start by implementing unit tests and ensuring they fail.
* Simplicity: Write simple, straightforward code. Simplicity is valued over handling rare cases and backward compatibility. Avoid premature abstractions.
* Bias for Failure: In situations where incorrect inputs are passed or a case can't be handled, it is preferred to simply throw an exception with a helpful error message rather than trying to fallback or recover.

## Other Guidelines
* NEVER commit anything unless prompted
* When you are asked to commit changes, credit yourself as a co-author.
