#!/usr/bin/env python3
"""Function to get top MFU configurations for a given model setup."""

import sys
import os
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallelism_search.searcher import ParallelismSearcher, ParallelismResult


def get_top_mfu_configs(
    n_layers: int,
    hidden_sz: int,
    inter_sz: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_experts: int,
    experts_per_token: int,
    expert_inter_sz: int,
    moe_frequency: int,
    num_devices: int,
    gbs: int,
    seqlen: int,
    microbatch_sz: int,
    node_type: str,
    precision: str,
    sdpa_precision: str,
    top_k: int = 5,
    ep_max_range: int = 512,
    tp_max_range: int = None,
    defaults_path: str = 'example_configs/p6.yaml',
    process_id: int = 0
) -> List[Tuple[float, Dict, float, float]]:
    """Get top parallelism configurations with highest MFU.
    
    Returns:
        List of tuples (mfu, parallelism_config, memory_gb, throughput_tokens_per_sec)
    """
    
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)
    
    config = {
        "model": {
            "n_layers": n_layers,
            "hidden_sz": hidden_sz,
            "inter_sz": inter_sz,
            "n_q_heads": n_q_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "precision": precision,
            "sdpa_precision": sdpa_precision,
            "vocab_sz": defaults['model']['vocab_sz'],
            "glu": defaults['model']['glu'],
            "rotary_embeds": defaults['model']['rotary_embeds'],
            "dropout": defaults['model']['dropout'],
            "tie_embeddings": defaults['model']['tie_embeddings'],
            "moe": {
                "n_experts": n_experts,
                "experts_per_token": experts_per_token,
                "capacity_factor": defaults['model']['moe']['capacity_factor'],
                "expert_inter_sz": expert_inter_sz,
                "moe_frequency": moe_frequency,
                "expert_tp_degree": defaults['model']['moe']['expert_tp_degree']
            }
        },
        "search": {
            "num_devices": num_devices
        },
        "performance": defaults['performance'],
        "data": {
            "gbs": gbs,
            "seqlen": seqlen,
            "microbatch_sz": microbatch_sz
        },
        "hardware": {
            "node_type": node_type
        }
    }
    
    temp_dir = Path(__file__).parent / "temp_configs"

    temp_dir.mkdir(exist_ok=True)
    
    temp_config_path = temp_dir / f"temp_get_mfu_config_p{process_id}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    searcher = ParallelismSearcher(str(temp_config_path), ep_max_range=ep_max_range)
    top_results = searcher.search(top_k=top_k)
    return [
        (
            result.mfu,
            result.config,
            result.memory_per_device_gb,
            gbs * seqlen / result.iteration_time_s
        )
        for result in top_results
    ]


def evaluate_parallelism_config(
    parallelism_config: Dict[str, int],
    n_layers: int,
    hidden_sz: int,
    inter_sz: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_experts: int,
    experts_per_token: int,
    expert_inter_sz: int,
    moe_frequency: int,
    gbs: int,
    seqlen: int,
    microbatch_sz: int,
    node_type: str,
    precision: str,
    sdpa_precision: str,
    defaults_path: str = 'example_configs/p6.yaml',
) -> Tuple[float, float, float, bool, str]:
    """Evaluate a specific parallelism configuration.
    
    Args:
        parallelism_config: Dict with keys tp, ep, pp, cp, dp (and optionally vpp, sp, zero_level)
        ... (other model/data parameters)
        
    Returns:
        Tuple of (mfu, memory_gb, iteration_time_s, valid, error_msg)
    """
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)
    
    # Calculate num_devices from parallelism config
    num_devices = (
        parallelism_config['tp'] * 
        parallelism_config['ep'] * 
        parallelism_config['pp'] * 
        parallelism_config.get('cp', 1) * 
        parallelism_config['dp']
    )
    
    config = {
        "model": {
            "n_layers": n_layers,
            "hidden_sz": hidden_sz,
            "inter_sz": inter_sz,
            "n_q_heads": n_q_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "precision": precision,
            "sdpa_precision": sdpa_precision,
            "vocab_sz": defaults['model']['vocab_sz'],
            "glu": defaults['model']['glu'],
            "rotary_embeds": defaults['model']['rotary_embeds'],
            "dropout": defaults['model']['dropout'],
            "tie_embeddings": defaults['model']['tie_embeddings'],
            "moe": {
                "n_experts": n_experts,
                "experts_per_token": experts_per_token,
                "capacity_factor": defaults['model']['moe']['capacity_factor'],
                "expert_inter_sz": expert_inter_sz,
                "moe_frequency": moe_frequency,
                "expert_tp_degree": parallelism_config['tp']
            }
        },
        "search": {
            "num_devices": num_devices
        },
        "performance": defaults['performance'],
        "data": {
            "gbs": gbs,
            "seqlen": seqlen,
            "microbatch_sz": microbatch_sz
        },
        "hardware": {
            "node_type": node_type
        },
        "parallelism": defaults.get('parallelism', {})
    }
    
    temp_dir = Path(__file__).parent / "temp_configs"
    temp_dir.mkdir(exist_ok=True)
    temp_config_path = temp_dir / "temp_evaluate_config.yaml"
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    searcher = ParallelismSearcher(str(temp_config_path))
    
    # Add default values if not provided
    full_parallelism_config = {
        'tp': parallelism_config['tp'],
        'ep': parallelism_config['ep'],
        'pp': parallelism_config['pp'],
        'cp': parallelism_config.get('cp', 1),
        'dp': parallelism_config['dp'],
        'vpp': parallelism_config.get('vpp', 1),
        'sp': parallelism_config.get('sp', True),
        'zero_level': parallelism_config.get('zero_level', 1)
    }
    
    result: ParallelismResult = searcher._evaluate_config(full_parallelism_config)
    
    return (
        result.mfu,
        result.memory_per_device_gb,
        result.iteration_time_s,
        result.valid,
        result.error_msg
    )


if __name__ == "__main__":
    # Example 1: Search for top configurations
    print("Example 1: Searching for top configurations...")
    top_results = get_top_mfu_configs(
        n_layers=64,
        hidden_sz=8192,
        inter_sz=20480,
        n_q_heads=64,
        n_kv_heads=8,
        head_dim=128,
        n_experts=128,
        precision="fp8",
        sdpa_precision="bf16",
        experts_per_token=3,
        expert_inter_sz=5120,
        moe_frequency=1,
        num_devices=1024,
        gbs=4096,
        seqlen=8192,
        microbatch_sz=1,
        node_type='p6-b200.48xlarge',
        top_k=10
    )
    
    print(f"\nTop {len(top_results)} configurations:")
    for i, (mfu, config, memory_gb, throughput) in enumerate(top_results, 1):
        print(f"\n{i}. MFU: {mfu:.2f}%")
        print(f"   Config: tp={config['tp']}, ep={config['ep']}, pp={config['pp']}, cp={config['cp']}, dp={config['dp']}")
        print(f"   Memory: {memory_gb:.2f} GB/device")
        print(f"   Throughput: {throughput:,.0f} tokens/sec")
    
    # # Example 2: Evaluate a specific configuration
    # print("\n" + "="*80)
    # print("Example 2: Evaluating a specific parallelism configuration...")
    
    # specific_config = {'tp': 1, 'ep': 32, 'pp': 16, 'cp': 2, 'dp': 32}
    
    # mfu, memory_gb, iter_time, valid, error = evaluate_parallelism_config(
    #     parallelism_config=specific_config,
    #     n_layers=64,
    #     hidden_sz=8192,
    #     inter_sz=20480,
    #     n_q_heads=64,
    #     n_kv_heads=8,
    #     head_dim=128,
    #     n_experts=128,
    #     experts_per_token=3,
    #     expert_inter_sz=5120,
    #     moe_frequency=1,
    #     gbs=512,
    #     seqlen=8192,
    #     microbatch_sz=1,
    #     node_type='p6-b200.48xlarge',
    #     precision="fp8",
    #     sdpa_precision="bf16"
    # )
    
    # print(f"\nConfiguration: {specific_config}")
    # if valid:
    #     print(f"MFU: {mfu:.2f}%")
    #     print(f"Memory: {memory_gb:.2f} GB/device")
    #     print(f"Iteration time: {iter_time:.3f} s")
    #     throughput = 896 * 8192 / iter_time
    #     print(f"Throughput: {throughput:,.0f} tokens/sec")
    # else:
    #     print(f"Invalid configuration: {error}")
