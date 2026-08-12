# 🚀 dlcalc

<div align="center">

[![PyPI version](https://badge.fury.io/py/dlcalc.svg)](https://badge.fury.io/py/dlcalc)
![checks](https://github.com/jfc4050/dlcalc/actions/workflows/python-app.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**command-line tools for deep learning optimization**

[Installation](#-installation) • [Tools](#-tools) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

`dlcalc` is a collection of tools for deep learning practitioners, providing calculators and tools for:

- 🧮 **Performance Modeling** - Estimate training throughput, memory usage, and MFU
- 🌐 **Topology Analysis** - Analyze and optimize network topology for distributed training
- 📊 **Metrics Conversion** - Convert between different performance metrics
- 🔍 **Checkpoint Analysis** - Inspect and summarize model checkpoints

## 🔧 Installation

### Via pip (recommended)
```bash
pip install dlcalc
```

or

### From source
```bash
git clone https://github.com/jfc4050/dlcalc
cd dlcalc
pip install -e .
```

After this you should have access to the command line tools described below. Some
people may need to add `--user` to their pip install command for them to properly
go under `$PATH`.


## 🛠 Tools

### 📐 Performance Modeling

#### **3D Training Calculator** (`3dtrn`)
Calculator for estimating performance characteristics of ND parallel transformer training:

```bash
3dtrn examples/llama3_70b.yaml
```

> We recommend to use this with profilers like [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) or [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) to give theoretical grounding to your performance profiling.

### 🌐 Topology Optimization

| Tool | Command | Purpose |
|------|---------|---------|
| **Visualizer** | `topoviz` | Generate network topology graphs from Kubernetes clusters |
| **Evaluator** | `topoeval` | Analyze topology optimality for DP rings |
| **Scheduler** | `topoassign` | Compute topology-aware rank assignments |

```bash
# Visualize cluster topology
topoviz -h

# Evaluate training job topology
topoeval -h

# Generate optimal rank assignments
topoassign -h
```

### 📊 Metrics & KPIs

#### **Samples/Sec → MFU Converter** (`sps2mfu`)
Convert training throughput to Model FLOPs Utilization (MFU):

```bash
sps2mfu --samples-per-sec 100 --seqlen 2048 --model-size 70b \
        --n-accelerators 512 --tflops-per-accelerator 312
```

#### **Samples/Sec → Tokens/Day Converter** (`sps2tpd`)
Calculate daily token throughput:

```bash
sps2tpd --samples-per-sec 100 --seqlen 2048
```

### 🔍 Utilities

#### **Checkpoint Summarizer** (`ckpt-summarize`)
Analyze PyTorch checkpoint contents:

```bash
ckpt-summarize model.pt
```

## 🧑‍💻 Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Run Quality Checks

```bash
# Run all checks (formatting, linting, type checking, tests)
bash checks
```

### Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=dlcalc --cov-report=term-missing
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Support

- 🐛 [Report bugs](https://github.com/jfc4050/dlcalc/issues)
- 💡 [Request features](https://github.com/jfc4050/dlcalc/issues)
- 📖 [Read the docs](https://github.com/jfc4050/dlcalc/wiki)

---

<div align="center">
Made with ❤️ for the deep learning community
</div>
