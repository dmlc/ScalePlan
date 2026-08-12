#!/usr/bin/env python3
"""
Plot MFU comparison between analytical estimates and benchmark results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from typing import List, Dict

def load_search_results(json_file: str) -> Dict:
    """Load search results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def extract_configs_with_benchmarks(results: List[Dict]) -> List[Dict]:
    """Extract configurations that have benchmark MFU data (not -1)."""
    return [
        result for result in results
        if result.get('benchmark_mfu', -1) > 0
    ]


def format_config_label(config: Dict[str, int]) -> str:
    """Format parallelism configuration as a readable label."""
    return f"ep{config['ep']}/pp{config['pp']}/dp{config['dp']}/tp{config['tp']}/cp{config['cp']}"


def plot_mfu_comparison(json_file: str, output_file: str = None, title_suffix: str = ""):
    """Plot MFU comparison between analytical and benchmark results."""

    # Load data
    data = load_search_results(json_file)

    # Extract configurations with benchmark data
    # Try both top_results and all_results
    configs_with_benchmarks = []

    if 'top_results' in data:
        configs_with_benchmarks.extend(extract_configs_with_benchmarks(data['top_results']))

    if 'all_results' in data:
        all_with_benchmarks = extract_configs_with_benchmarks(data['all_results'])
        # Add any from all_results that aren't already in top_results
        existing_configs = {str(c['config']) for c in configs_with_benchmarks}
        for config in all_with_benchmarks:
            if str(config['config']) not in existing_configs:
                configs_with_benchmarks.append(config)

    if not configs_with_benchmarks:
        print("No configurations with benchmark MFU data found!")
        return

    print(f"Found {len(configs_with_benchmarks)} configurations with benchmark data")

    # Sort by benchmark MFU (descending)
    configs_with_benchmarks.sort(key=lambda x: x['benchmark_mfu'], reverse=True)

    # Extract data for plotting
    config_labels = [format_config_label(c['config']) for c in configs_with_benchmarks]
    analytical_mfus = [c.get('analytical_mfu', c.get('mfu', 0)) for c in configs_with_benchmarks]
    benchmark_mfus = [c['benchmark_mfu'] for c in configs_with_benchmarks]

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Set up bar positions
    x = np.arange(len(config_labels))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, analytical_mfus, width, label='Analytical MFU',
                   alpha=0.8, color='skyblue', edgecolor='navy', linewidth=1)
    bars2 = ax.bar(x + width/2, benchmark_mfus, width, label='Benchmark MFU',
                   alpha=0.8, color='lightcoral', edgecolor='darkred', linewidth=1)

    # Customize the plot
    # Extract metadata for title
    metadata = data.get('metadata', {})
    num_devices = metadata.get('num_devices', 'Unknown')
    num_experts = metadata.get('num_experts', 'Unknown')
    global_batch_size = metadata.get('global_batch_size', 'Unknown')

    # Create enhanced title with device, expert, and batch size information
    title = f'MFU Comparison: Analytical vs Benchmark{title_suffix}\n({num_devices} devices, {num_experts} experts, batch_size={global_batch_size})'

    ax.set_xlabel('Parallelism Configuration (ep/pp/dp/tp/cp)', fontsize=16, fontweight='bold')
    ax.set_ylabel('MFU (%)', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xticks(x)

    ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=25)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=18, fontweight='bold')

    add_value_labels(bars1, analytical_mfus)
    add_value_labels(bars2, benchmark_mfus)

    # Calculate and display differences
    differences = [a - b for a, b in zip(analytical_mfus, benchmark_mfus)]
    avg_diff = np.mean(differences)
    max_diff = max(differences)
    min_diff = min(differences)

    # Add statistics text box
    stats_text = 'MFU Differences (Analytical - Benchmark):\n'
    stats_text += f'Average: {avg_diff:+.2f}%\n'
    stats_text += f'Max: {max_diff:+.2f}%\n'
    stats_text += f'Min: {min_diff:+.2f}%'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # output_file = "parallelism_search/outputs/mfu_comparison_s4_e64_nc32.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


    # Print summary table
    print(f"\n{'='*80}")
    print("MFU COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Analytical':<12} {'Benchmark':<12} {'Difference':<12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    for i, (config, analytical, benchmark) in enumerate(zip(config_labels, analytical_mfus, benchmark_mfus)):
        diff = analytical - benchmark
        print(f"{config:<25} {analytical:<12.2f} {benchmark:<12.2f} {diff:<+12.2f}")

    print("\nSummary Statistics:")
    print(f"Average difference: {avg_diff:+.2f}%")
    print(f"Max difference: {max_diff:+.2f}%")
    print(f"Min difference: {min_diff:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Plot MFU comparison from search results')
    parser.add_argument('json_file', help='Path to JSON results file')
    parser.add_argument('--output', '-o', help='Output file for plot (PNG/PDF/SVG)')
    parser.add_argument('--title-suffix', default='', help='Additional text for plot title')

    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File {args.json_file} not found!")
        return 1

    try:
        plot_mfu_comparison(args.json_file, args.output, args.title_suffix)
        return 0
    except Exception as e:
        print(f"Error creating plot: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
