#!/usr/bin/env python3
"""Example script for parallelism configuration grid search - module internal."""

import os
from .searcher import ParallelismSearcher


def run_search_example():
    """Run a search example using the test configuration."""
    # Get the path to the examples directory relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(os.path.dirname(current_dir), "examples")
    config_path = os.path.join(examples_dir, "moe_search_test.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    # Create and run searcher
    searcher = ParallelismSearcher(config_path)
    top_results = searcher.search(top_k=10)

    # Print results and save to file
    searcher.print_results(top_results, save_to_file=True, config_name="example_search")

    # Print the best configuration
    if top_results:
        best_config = top_results[0]
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION (YAML format):")
        print(f"{'='*80}")
        print("parallelism:")
        for key, value in best_config.config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    run_search_example()
