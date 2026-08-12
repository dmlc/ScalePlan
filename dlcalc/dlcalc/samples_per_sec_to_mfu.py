"""Utility to convert samples/second to MFU (Model FLOPs Utilization)."""

import re
from argparse import ArgumentParser

from dlcalc.utils.printing import (
    _GRAY,
    _RED,
    format_number,
    print_info,
    print_kv,
    print_metric,
    print_section_separator,
    print_success,
    print_warning,
)

n_params_pattern = re.compile(r"([\d\.]+)([a-z])")
n_accelerators_pattern = re.compile(r"(\d+)x(\d+)")


def parse_n_params(n_params_str: str) -> int:
    SUFFIX_TO_FACTOR = {
        "m": 1e6,
        "b": 1e9,
        "t": 1e12,
    }
    n_params_str = n_params_str.lower()
    match = re.match(n_params_pattern, n_params_str)

    if not match:
        raise ValueError(f"unable to parse n_params str {n_params_str}")

    prefix = float(match.group(1))

    suffix = match.group(2).lower()
    if suffix not in SUFFIX_TO_FACTOR:
        raise ValueError(
            f"unable to parse n_params str '{n_params_str}' because of unrecognized suffix '{suffix}'"
        )
    factor = SUFFIX_TO_FACTOR[suffix]

    return int(prefix * factor)


def parse_n_accelerators(n_accelerators_str: str) -> int:
    n_accelerators_str = n_accelerators_str.lower()
    if n_accelerators_str.isnumeric():
        return int(n_accelerators_str)
    elif "x" in n_accelerators_str:
        # axb format
        match = re.match(n_accelerators_pattern, n_accelerators_str)
        if not match:
            raise ValueError(f"unable to parse n_accelerators format '{n_accelerators_str}'")
        multiplicand_1 = int(match.group(1))
        multiplicand_2 = int(match.group(2))

        return multiplicand_1 * multiplicand_2
    else:
        raise ValueError(f"unable to parse n_accelerators input '{n_accelerators_str}'")


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "-s",
        "--samples-per-sec",
        type=float,
        required=True,
        help="training throughput in samples/sec",
    )
    parser.add_argument(
        "-l",
        "--seqlen",
        type=int,
        required=True,
        help="sequence length in tokens (i.e. tokens/sample)",
    )
    parser.add_argument(
        "-m",
        "--model-size",
        type=str,
        required=True,
        help="model size (for example 100m, 100b, or 100t)",
    )
    parser.add_argument(
        "-n",
        "--n-accelerators",
        type=str,
        required=True,
        help="number of accelerators used for training. accepts formats like '16' or '2x8'",
    )
    parser.add_argument(
        "-t",
        "--tflops-per-accelerator",
        type=float,
        required=True,
        help="theoretical tflops per accelerator",
    )
    args = parser.parse_args()

    samples_per_sec = args.samples_per_sec
    tokens_per_sample = args.seqlen
    model_size = parse_n_params(args.model_size)
    n_accelerators = parse_n_accelerators(args.n_accelerators)
    flops_per_accelerator = args.tflops_per_accelerator * 1e12

    tokens_per_sec = tokens_per_sample * samples_per_sec
    flops_per_token = 6 * model_size

    # MFU
    achieved_flops = flops_per_token * tokens_per_sec
    theoretical_flops = flops_per_accelerator * n_accelerators
    mfu = achieved_flops / theoretical_flops

    # tokens/day
    tokens_per_day = tokens_per_sec * 60 * 60 * 24

    # Input Configuration
    print_section_separator()
    print_info("Configuration")
    print_kv("Model Size", f"{format_number(model_size)} parameters")
    print_kv("Sequence Length", f"{tokens_per_sample:,} tokens")
    print_kv("Training Throughput", f"{samples_per_sec:.2f} samples/sec")
    print_kv("Accelerators", f"{n_accelerators} devices")
    print_kv("Peak FLOPS/device", f"{flops_per_accelerator * 1e-12:.2f} TFLOPS")

    # Performance Metrics
    print_section_separator()
    print_info("Performance Metrics")

    # Tokens per second
    tokens_display = f"{format_number(tokens_per_sec)} tokens/sec"
    print_kv("Token Throughput", tokens_display)

    # Tokens per day
    daily_display = f"{format_number(tokens_per_day)} tokens/day"
    print_kv("Daily Volume", daily_display)

    # FLOPS metrics
    print_kv("Achieved FLOPS", f"{achieved_flops * 1e-12:.2f} TFLOPS")
    print_kv("Theoretical FLOPS", f"{theoretical_flops * 1e-12:.2f} TFLOPS")
    print_kv("FLOPs per Token", f"{format_number(flops_per_token)}")

    # MFU Result
    print_section_separator()
    mfu_percent = mfu * 100

    # Color code and status based on MFU
    if mfu_percent > 100:
        print(f"  {_RED}⚠ MFU: {mfu_percent:.2f}% - Check input values!{_GRAY}")
        print(
            f"  {_GRAY}(MFU > 100% indicates incorrect parameters or unrealistic throughput){_GRAY}"
        )
    elif mfu_percent >= 50:
        print_success(f"MFU: {mfu_percent:.2f}%")
    elif mfu_percent >= 30:
        print_metric("MFU", f"{mfu_percent:.2f}%", highlight=True)
    elif mfu_percent >= 20:
        print_warning(f"MFU: {mfu_percent:.2f}%")
    else:
        print(f"  {_RED}✗ MFU: {mfu_percent:.2f}%{_GRAY}")

    print()


if __name__ == "__main__":
    main()
