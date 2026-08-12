"""Parallelism configuration grid search for 3D training."""

import json
import yaml
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from tqdm import tqdm

from dlcalc.utils.hardware import MachineSpec
from parallelism_search.training_calculator import calculate_training_metrics


@dataclass
class ParallelismResult:
    """Result of a parallelism configuration evaluation."""
    config: Dict[str, Any]
    mfu: float
    memory_per_device_gb: float
    iteration_time_s: float
    valid: bool
    error_msg: str = ""
    benchmark_mfu: float = -1.0  # MLflow benchmark MFU


class ParallelismSearcher:
    """Grid search for optimal parallelism configuration."""

    def __init__(
        self,
        config_path: str,
        num_devices: int = 512,
        mlflow_experiment_id: str = None,
        max_ep: int = 32,
        max_tp: int = None,
        max_cp: int = None,
        max_etp: int = None,
        memory_limit_fraction: float = 0.9,
        activation_checkpointing: str = "all",
        max_mbs: int = 1,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.num_devices = num_devices
        self.max_ep = max_ep
        self.max_tp = max_tp
        self.max_cp = max_cp
        self.max_etp = max_etp
        self.memory_limit_fraction = memory_limit_fraction
        # Parse activation checkpointing - can be single value or "all" to iterate
        if activation_checkpointing == "all":
            self.activation_checkpointing_types = ["none", "selective", "full"]
        else:
            self.activation_checkpointing_types = [activation_checkpointing]

        # Microbatch size range
        self.max_mbs = max_mbs
        self.mbs_values = list(range(1, max_mbs + 1))
        self.machine_spec = MachineSpec.from_str(self.cfg["hardware"]["node_type"])
        self.device_memory_limit_gb = self.machine_spec.device_spec.mem_capacity_bytes / (1024**3)

        # Fixed model and data parameters
        self.model_cfg = self.cfg["model"]
        self.data_cfg = self.cfg["data"]
        self.performance_cfg = self.cfg["performance"]

        # MLflow integration
        self.mlflow_experiment_id = mlflow_experiment_id

        # Setup MLflow credentials once if needed
        if self.mlflow_experiment_id:
            try:
                from parallelism_search.collect_mlflow_benchmarks import setup_credentials
                setup_credentials()
                print("MLflow credentials setup completed")
            except Exception as e:
                print(f"Warning: MLflow credentials setup failed: {e}")

        self.results: List[ParallelismResult] = []

    def _format_config(self, config: Dict[str, Any]) -> str:
        """Format a parallelism config dict as a readable string."""
        parts = [
            f"tp={config['tp']}",
            f"ep={config['ep']}",
            f"pp={config['pp']}",
            f"cp={config['cp']}",
            f"dp={config['dp']}",
        ]
        if "etp" in config:
            parts.append(f"etp={config['etp']}")
        if "activation_checkpointing" in config:
            parts.append(f"act_ckpt={config['activation_checkpointing']}")
        if "microbatch_sz" in config:
            parts.append(f"mbs={config['microbatch_sz']}")
        return ", ".join(parts)

    def _print_model_info(self) -> None:
        """Print model architecture information including parameter counts."""
        from dlcalc.utils.configurations import ActivationCheckpointingType
        from dlcalc.utils.model_3d import MoeCfg, ParallelConfig, ThreeDParallelModel

        # Create a minimal model representation to get parameter counts
        # Use tp=1, pp=1, dp=num_devices as baseline (params are unpartitioned)
        expert_mesh = None
        if "moe" in self.model_cfg:
            expert_mesh = ParallelConfig.ExpertParallelCfg(ep=1, tp=1, dp=self.num_devices)

        model_repr = ThreeDParallelModel(
            parallelism_cfg=ParallelConfig(
                tp=1, cp=1, pp=1, dp=self.num_devices,
                expert_mesh=expert_mesh,
                vpp=1, sp_enabled=True,
                zero_level=ParallelConfig.ZeroLevel.PARTITION_OPTIMIZER,
            ),
            sequence_len=self.data_cfg["seqlen"],
            microbatch_sz=self.data_cfg["microbatch_sz"],
            hidden_sz=self.model_cfg["hidden_sz"],
            n_layers=self.model_cfg["n_layers"],
            n_q_heads=self.model_cfg["n_q_heads"],
            n_kv_heads=self.model_cfg["n_kv_heads"],
            head_dim=self.model_cfg["head_dim"],
            inter_sz=self.model_cfg["inter_sz"],
            glu=self.model_cfg["glu"],
            moe_cfg=MoeCfg(
                n_experts=self.model_cfg["moe"]["n_experts"],
                expert_inter_sz=self.model_cfg["moe"]["expert_inter_sz"],
                experts_per_token=self.model_cfg["moe"]["experts_per_token"],
                capacity_factor=self.model_cfg["moe"]["capacity_factor"],
                moe_frequency=self.model_cfg["moe"]["moe_frequency"],
                expert_tp_degree=self.model_cfg["moe"]["expert_tp_degree"],
            ) if "moe" in self.model_cfg else None,
            rotary_embed=self.model_cfg["rotary_embeds"],
            dropout=self.model_cfg["dropout"],
            vocab_sz=self.model_cfg["vocab_sz"],
            tie_embeddings=self.model_cfg["tie_embeddings"],
            act_ckpting_type=ActivationCheckpointingType.from_str(
                self.performance_cfg["activation_checkpointing_type"]
            ),
            n_param_buckets=5,
            optimizer_type=self.cfg.get("optimizer", {}).get("optimizer_type", "adam"),
        )

        total_params = model_repr.get_n_total_params(partitioned=False)
        active_params = model_repr.get_n_active_params(partitioned=False)

        def format_params(n: int) -> str:
            if n >= 1e12:
                return f"{n / 1e12:.2f}T"
            elif n >= 1e9:
                return f"{n / 1e9:.2f}B"
            elif n >= 1e6:
                return f"{n / 1e6:.2f}M"
            else:
                return str(n)

        print(f"\n{'='*60}")
        print("MODEL INFORMATION")
        print(f"{'='*60}")
        print(f"  Total Parameters:  {format_params(total_params)}")
        print(f"  Active Parameters: {format_params(active_params)}")
        print(f"  Layers: {self.model_cfg['n_layers']}")
        print(f"  Hidden Size: {self.model_cfg['hidden_sz']}")
        if "moe" in self.model_cfg:
            print(f"  MoE Experts: {self.model_cfg['moe']['n_experts']}")
            print(f"  Experts per Token: {self.model_cfg['moe']['experts_per_token']}")
        print(f"{'='*60}\n")

    def _generate_parallelism_configs(self) -> List[Dict[str, int]]:
        """Generate all valid parallelism configurations."""
        configs = []

        # Determine tp range based on whether max is specified
        max_tp = self.max_tp if self.max_tp is not None else self.num_devices
        tp_values = range(1, min(max_tp, self.num_devices) + 1)

        # Generate all possible factor combinations for tp*pp*cp*dp = num_devices (no ep)
        for tp in tp_values:
            if self.num_devices % tp != 0:
                continue

            remaining_after_tp = self.num_devices // tp

            for pp in range(1, remaining_after_tp + 1):
                if remaining_after_tp % pp != 0:
                    continue

                # Check if layers can be divided by pp
                if self.model_cfg["n_layers"] % pp != 0:
                    continue

                remaining_after_pp = remaining_after_tp // pp

                # Determine cp range based on whether max is specified
                max_cp = self.max_cp if self.max_cp is not None else remaining_after_pp
                cp_values = range(1, min(max_cp, remaining_after_pp) + 1)

                for cp in cp_values:
                    if remaining_after_pp % cp != 0:
                        continue

                    dp = remaining_after_pp // cp

                    # Constraint: dp must be greater than 1 (require data parallelism)
                    if dp < 1:
                        continue

                    # Verify the total equals num_devices (without ep)
                    total = tp * pp * cp * dp
                    if total != self.num_devices:
                        continue

                    for ep in range(1, self.max_ep + 1):
                        # Calculate max expert parallelism based on available devices
                        # After allocating tp, pp, cp, dp, remaining devices can be used for ep
                        max_ep_for_config = total // (pp * cp)  # Devices available per model replica
                        if ep > max_ep_for_config:
                            continue

                        # Additional constraints for MoE
                        if "moe" in self.model_cfg:
                            # Expert parallelism shouldn't exceed number of experts
                            if ep > self.model_cfg["moe"]["n_experts"]:
                                continue

                            # Number of experts should be divisible by ep
                            if self.model_cfg["moe"]["n_experts"] % ep != 0:
                                continue

                        # Determine etp range - defaults to just tp if not specified
                        if self.max_etp is not None:
                            etp_values = [e for e in range(1, self.max_etp + 1) if tp % e == 0]
                        else:
                            etp_values = [tp]  # Default: etp = tp

                        for etp in etp_values:
                            for act_ckpt in self.activation_checkpointing_types:
                                for mbs in self.mbs_values:
                                    configs.append({
                                        "tp": tp,
                                        "ep": ep,
                                        "pp": pp,
                                        "cp": cp,
                                        "dp": dp,
                                        "etp": etp,
                                        "vpp": 1,  # Keep VPP simple for now
                                        "sp": True,  # Enable sequence parallelism
                                        "zero_level": 1,  # ZeRO stage 1
                                        "activation_checkpointing": act_ckpt,
                                        "microbatch_sz": mbs,
                                    })

        return configs

    def _evaluate_config(self, parallelism_config: Dict[str, Any]) -> ParallelismResult:
        """Evaluate a single parallelism configuration using exact training_3d_v1.py logic."""
        try:
            # Create model config with updated expert_tp_degree
            model_config = self.model_cfg.copy()
            etp = parallelism_config.pop("etp", parallelism_config["tp"])
            if "moe" in model_config:
                model_config["moe"] = model_config["moe"].copy()
                model_config["moe"]["expert_tp_degree"] = etp

            # Extract activation checkpointing and microbatch size from parallelism config
            act_ckpt = parallelism_config.pop("activation_checkpointing", "full")
            mbs = parallelism_config.pop("microbatch_sz", self.data_cfg["microbatch_sz"])

            # Create data config with possibly overridden microbatch size
            data_config = self.data_cfg.copy()
            data_config["microbatch_sz"] = mbs

            # Create full configuration
            full_config = {
                "model": model_config,
                "parallelism": {
                    **parallelism_config,
                    "n_param_buckets": 5  # Default number of parameter buckets (matching s4.yaml)
                },
                "performance": {"activation_checkpointing_type": act_ckpt},
                "data": data_config,
                "hardware": self.cfg["hardware"],
                "optimizer": self.cfg.get("optimizer", {}),
            }

            # Add them back to parallelism_config for the result
            parallelism_config["etp"] = etp
            parallelism_config["activation_checkpointing"] = act_ckpt
            parallelism_config["microbatch_sz"] = mbs

            # Use exact calculation from training_3d_v1.py
            mfu, iteration_time_s, memory_per_device_gb = calculate_training_metrics(full_config)

            # Check memory constraint
            if memory_per_device_gb > self.device_memory_limit_gb * self.memory_limit_fraction:  # 90% memory utilization limit
                return ParallelismResult(
                    config=parallelism_config,
                    mfu=0.0,
                    memory_per_device_gb=memory_per_device_gb,
                    iteration_time_s=float('inf'),
                    valid=False,
                    error_msg=f"Memory exceeds limit: {memory_per_device_gb:.2f}GB > {self.device_memory_limit_gb * 0.9:.2f}GB"
                )

            # Get benchmark MFU from MLflow if available
            benchmark_mfu = -1.0
            if self.mlflow_experiment_id and parallelism_config["cp"]==1:
                # Only consider configurations where cp=1 and pp=1
                n_experts = self.model_cfg["moe"]["n_experts"]
                devices_div8 = self.num_devices // 8
                ep = parallelism_config["ep"]
                tp = parallelism_config["tp"]


                # Create job name: if pp > 1, include pp in the name
                if parallelism_config["pp"] > 1:
                    pp = parallelism_config["pp"]
                    job_name = f"s4-e{n_experts}-{devices_div8}-{ep}ep-{tp}tp-{pp}pp"
                else:
                    job_name = f"s4-e{n_experts}-{devices_div8}-{ep}ep-{tp}tp"

                try:
                    # Site-specific MLflow integration; optional by design.
                    from parallelism_search.collect_mlflow_benchmarks import get_benchmark_mfu

                    benchmark_mfu = get_benchmark_mfu(self.mlflow_experiment_id, job_name)
                except Exception as e:
                    print(f"Failed to get benchmark MFU for {job_name}: {e}")
                    benchmark_mfu = -1.0

            return ParallelismResult(
                config=parallelism_config,
                mfu=mfu,
                memory_per_device_gb=memory_per_device_gb,
                iteration_time_s=iteration_time_s,
                valid=True,
                benchmark_mfu=benchmark_mfu
            )

        except Exception as e:
            return ParallelismResult(
                config=parallelism_config,
                mfu=0.0,
                memory_per_device_gb=0.0,
                iteration_time_s=float('inf'),
                valid=False,
                error_msg=str(e),
                benchmark_mfu=-1.0
            )

    def search(self, top_k: int = 5) -> List[ParallelismResult]:
        """Perform grid search and return top-k configurations."""
        # Print model info
        self._print_model_info()

        print(f"Searching parallelism configurations for {self.num_devices} devices...")
        print(f"Device memory limit: {self.device_memory_limit_gb:.1f} GB")

        # Generate all possible configurations
        configs = self._generate_parallelism_configs()
        print(f"Generated {len(configs)} parallelism configurations to evaluate")

        # Evaluate each configuration
        self.results = []
        valid_configs = 0

        for config in tqdm(configs, desc="Evaluating configurations"):
            result = self._evaluate_config(config)
            self.results.append(result)

            if result.valid:
                valid_configs += 1

        print(f"Found {valid_configs} valid configurations out of {len(configs)} total")

        # Sort by MFU (descending) and filter valid results
        valid_results = [r for r in self.results if r.valid]
        valid_results.sort(key=lambda x: x.mfu, reverse=True)

        return valid_results[:top_k]

    def save_results(self, results: List[ParallelismResult], config_name: str = "search") -> str:
        """Save search results to files in outputs/ directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create outputs directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(current_dir, "outputs_v1/")
        os.makedirs(outputs_dir, exist_ok=True)

        # Generate filenames
        # base_filename = f"{config_name}_{timestamp}"
        base_filename = f"{config_name}"
        txt_filename = os.path.join(outputs_dir, f"{base_filename}.txt")
        json_filename = os.path.join(outputs_dir, f"{base_filename}.json")
        yaml_filename = os.path.join(outputs_dir, f"{base_filename}_best.yaml")

        # Save human-readable text output
        with open(txt_filename, 'w') as f:
            f.write("Parallelism Search Results\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total devices: {self.num_devices}\n")
            f.write(f"Global batch size: {self.data_cfg['gbs']}\n")
            f.write(f"Device memory limit: {self.device_memory_limit_gb:.1f} GB\n")
            f.write(f"Total configurations evaluated: {len(self.results)}\n")
            f.write(f"Valid configurations found: {len([r for r in self.results if r.valid])}\n\n")

            if results:
                f.write(f"TOP {len(results)} PARALLELISM CONFIGURATIONS\n")
                f.write(f"{'='*50}\n\n")

                for i, result in enumerate(results, 1):
                    f.write(f"Rank {i}:\n")
                    f.write(f"  Configuration: {self._format_config(result.config)}\n")
                    f.write(f"  Analytical MFU: {result.mfu:.2f}%\n")
                    if result.benchmark_mfu > 0:
                        f.write(f"  Benchmark MFU: {result.benchmark_mfu:.2f}%\n")
                        diff = result.mfu - result.benchmark_mfu
                        f.write(f"  MFU Difference: {diff:+.2f}% (analytical - benchmark)\n")
                    else:
                        f.write("  Benchmark MFU: Not available\n")
                    f.write(f"  Memory per device: {result.memory_per_device_gb:.2f} GB\n")
                    f.write(f"  Iteration time: {result.iteration_time_s:.3f} s\n\n")
            else:
                f.write("No valid configurations found!\n\n")

            # Add section for configurations with benchmark MFU data
            benchmark_results = [r for r in self.results if r.valid and r.benchmark_mfu > 0]
            if benchmark_results:
                # Sort by benchmark MFU (descending)
                benchmark_results.sort(key=lambda x: x.benchmark_mfu, reverse=True)
                f.write(f"\nCONFIGURATIONS WITH BENCHMARK MFU DATA ({len(benchmark_results)} total)\n")
                f.write(f"{'='*50}\n")
                f.write("(Sorted by Benchmark MFU, descending)\n\n")

                for i, result in enumerate(benchmark_results, 1):
                    f.write(f"Rank {i}:\n")
                    f.write(f"  Configuration: {self._format_config(result.config)}\n")
                    f.write(f"  Benchmark MFU: {result.benchmark_mfu:.2f}%\n")
                    f.write(f"  Analytical MFU: {result.mfu:.2f}%\n")
                    diff = result.mfu - result.benchmark_mfu
                    f.write(f"  MFU Difference: {diff:+.2f}% (analytical - benchmark)\n")
                    f.write(f"  Memory per device: {result.memory_per_device_gb:.2f} GB\n")
                    f.write(f"  Iteration time: {result.iteration_time_s:.3f} s\n\n")

            # Add invalid configurations for debugging
            invalid_results = [r for r in self.results if not r.valid]
            if invalid_results:
                f.write(f"\nINVALID CONFIGURATIONS ({len(invalid_results)} total):\n")
                f.write("-" * 40 + "\n")
                for i, result in enumerate(invalid_results[:10], 1):  # Show first 10
                    f.write(f"{i}. {self._format_config(result.config)}\n")
                    f.write(f"   Error: {result.error_msg}\n")

        # Save JSON output for programmatic access
        def result_to_dict(result):
            """Convert result to dict with clearer field names."""
            result_dict = asdict(result)
            # Rename mfu to analytical_mfu for clarity
            result_dict["analytical_mfu"] = result_dict.pop("mfu")
            return result_dict

        json_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_devices": self.num_devices,
                "global_batch_size": self.data_cfg['gbs'],
                "num_experts": self.model_cfg.get("moe", {}).get("n_experts", "N/A"),
                "device_memory_limit_gb": self.device_memory_limit_gb,
                "total_configurations": len(self.results),
                "valid_configurations": len([r for r in self.results if r.valid])
            },
            "top_results": [result_to_dict(result) for result in results],
            "all_results": [result_to_dict(result) for result in self.results]
        }

        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Save best configuration as YAML for easy copy-paste
        if results:
            best_config = results[0]
            yaml_data = {
                "# Best configuration found by parallelism search": None,
                "# Analytical MFU": f"{best_config.mfu:.2f}%",
                "# Benchmark MFU": f"{best_config.benchmark_mfu:.2f}%" if best_config.benchmark_mfu > 0 else "Not available",
                "# Memory per device": f"{best_config.memory_per_device_gb:.2f} GB",
                "# Iteration time": f"{best_config.iteration_time_s:.3f} s",
                "parallelism": best_config.config
            }

            with open(yaml_filename, 'w') as f:
                # Write comments manually since PyYAML doesn't handle them well
                f.write("# Best configuration found by parallelism search\n")
                f.write(f"# Analytical MFU: {best_config.mfu:.2f}%\n")
                if best_config.benchmark_mfu > 0:
                    f.write(f"# Benchmark MFU: {best_config.benchmark_mfu:.2f}%\n")
                    diff = best_config.mfu - best_config.benchmark_mfu
                    f.write(f"# MFU Difference: {diff:+.2f}% (analytical - benchmark)\n")
                else:
                    f.write("# Benchmark MFU: Not available\n")
                f.write(f"# Memory per device: {best_config.memory_per_device_gb:.2f} GB\n")
                f.write(f"# Iteration time: {best_config.iteration_time_s:.3f} s\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                yaml.dump({"parallelism": best_config.config}, f, default_flow_style=False)

        return base_filename

    def print_results(self, results: List[ParallelismResult], save_to_file: bool = True, config_name: str = "search"):
        """Print the search results in a formatted way and optionally save to file."""
        print(f"\n{'='*80}")
        print("TOP PARALLELISM CONFIGURATIONS")
        print(f"{'='*80}")

        for i, result in enumerate(results, 1):
            print(f"\nRank {i}:")
            print(f"  Configuration: {self._format_config(result.config)}")
            print(f"  Analytical MFU: {result.mfu:.2f}%")
            if result.benchmark_mfu > 0:
                print(f"  Benchmark MFU: {result.benchmark_mfu:.2f}%")
                diff = result.mfu - result.benchmark_mfu
                print(f"  MFU Difference: {diff:+.2f}% (analytical - benchmark)")
            else:
                print("  Benchmark MFU: Not available")
            print(f"  Memory per device: {result.memory_per_device_gb:.2f} GB")
            print(f"  Iteration time: {result.iteration_time_s:.3f} s")

        if not results:
            print("No valid configurations found!")

        # Print some invalid configurations for debugging
        invalid_results = [r for r in self.results if not r.valid]
        if invalid_results:
            print(f"\nSample invalid configurations ({len(invalid_results)} total):")
            for result in invalid_results[:3]:  # Show first 3 invalid ones
                print(f"  {self._format_config(result.config)}")
                print(f"    Error: {result.error_msg}")

        # Save to file if requested
        if save_to_file:
            base_filename = self.save_results(results, config_name)
            print(f"\n{'='*80}")
            print("RESULTS SAVED TO FILES:")
            print(f"{'='*80}")
            print(f"Text output: parallelism_search/outputs_v1/{base_filename}.txt")
            print(f"JSON output: parallelism_search/outputs_v1/{base_filename}.json")
            if results:
                print(f"Best config YAML: parallelism_search/outputs_v1/{base_filename}_best.yaml")

            return base_filename
