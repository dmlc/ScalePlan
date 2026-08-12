#!/usr/bin/env python3
"""Entry point for parallelism configuration grid search."""

import sys
import os
import argparse

# Add the parent directory to sys.path to import parallelism_search
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parallelism_search.searcher import ParallelismSearcher


def main():
    parser = argparse.ArgumentParser(description="Search for optimal parallelism configuration")
    parser.add_argument("config_path", type=str, help="Path to search configuration YAML file")
    parser.add_argument("--devices", type=int, default=512,
                       help="Number of devices to search over (default: 512)")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top configurations to show")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--output-name", type=str, default="search", help="Base name for output files")
    parser.add_argument("--mlflow-experiment-id", type=str, default=None,
                       help="MLflow experiment ID to fetch benchmark MFU (e.g., '31340')")
    parser.add_argument("--max-ep", type=int, default=32,
                       help="Maximum expert parallelism (searches 1 to max_ep, default: 32)")
    parser.add_argument("--max-tp", type=int, default=None,
                       help="Maximum tensor parallelism degree (searches 1 to max_tp, default: no limit)")
    parser.add_argument("--max-cp", type=int, default=None,
                       help="Maximum context parallelism degree (searches 1 to max_cp, default: no limit)")
    parser.add_argument("--max-etp", type=int, default=None,
                       help="Maximum expert tensor parallelism (searches divisors of tp up to max_etp, default: etp=tp)")
    parser.add_argument("--memory-limit-fraction", type=float, default=0.8,
                       help="Fraction of device memory to use as limit (default: 0.9)")
    parser.add_argument("--activation-checkpointing", type=str, default="all",
                       choices=["none", "selective", "full", "all"],
                       help="Activation checkpointing type: 'none', 'selective', 'full', or 'all' to search over all (default: all)")
    parser.add_argument("--max-mbs", type=int, default=1,
                       help="Maximum microbatch size (searches 1 to max_mbs, default: 1)")
    parser.add_argument("--python-path", type=str, default=None,
                       help="Path to Python executable with sda/mlflow installed (e.g., '~/workspace/test/env/bin/python')")

    args = parser.parse_args()

    searcher = ParallelismSearcher(args.config_path,
                                  num_devices=args.devices,
                                  mlflow_experiment_id=args.mlflow_experiment_id,
                                  max_ep=args.max_ep,
                                  max_tp=args.max_tp,
                                  max_cp=args.max_cp,
                                  max_etp=args.max_etp,
                                  memory_limit_fraction=args.memory_limit_fraction,
                                  activation_checkpointing=args.activation_checkpointing,
                                  max_mbs=args.max_mbs)
    top_results = searcher.search(top_k=args.top_k)

    # Extract config name from path for better file naming
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    if args.output_name == "search":
        args.output_name = config_name

    # Print results and save to file
    base_filename = searcher.print_results(
        top_results,
        save_to_file=not args.no_save,
        config_name=args.output_name
    )

    # Print the best configuration in YAML format for easy copy-paste
    if top_results:
        best_config = top_results[0]
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION (YAML format):")
        print(f"{'='*80}")
        print("parallelism:")
        for key, value in best_config.config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
