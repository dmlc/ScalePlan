"""Benchmark NCCL all_to_all_single across message sizes (MoE dispatch/combine).

Motivation:
    dlcalc/utils/comms.py:get_all_to_all_comm_time_s charges the intra-node
    all-to-all at the FULL NVLink unidirectional peak (900 GB/s on B200). The
    18b rank0 traces (dlcalc/tests/data/18b_real_mfu_P6_64node.csv runs) show
    the real per-GPU egress bandwidth of an 8-rank NVSwitch all_to_all_single
    is ~324-366 GB/s at p50 for 300 MB - 1.2 GB buffers -- i.e. an efficiency
    of ~0.36-0.41, making the model ~2.6-3.4x too fast on EP A2A everywhere.
    NCCL implements all_to_all_single as (n-1) point-to-point copies per rank
    (P2P/CE path), which does not saturate NVSwitch the way ring collectives
    do. This script MEASURES the effective bandwidth curve so the model's
    efficiency comes from data, not an asserted constant (GUIDELINES §1).

What it measures:
    torch.distributed.all_to_all_single on a contiguous bf16 buffer, sweeping
    total buffer size per rank (the model's `size` argument = the full local
    dispatch buffer; each rank sends size/n to every peer, keeping 1/n local).
    Effective per-GPU egress BW = (size * (n-1)/n) / time. Reported per size
    with p50/p10/p90 over timed iterations.

    Group sizes: 2, 4, 8 ranks within one node (subgroups of the 8-GPU world;
    ranks beyond the group idle at a barrier). Multi-node EP (e.g. ep32) needs
    a multi-node run of this same script -- the sweep auto-detects when
    WORLD_SIZE spans nodes and labels rows inter_node=True.

Usage (single 8-GPU B200 node, e.g. the sleeper pod):
    torchrun --nproc_per_node=8 benchmarks/a2a_benchmark.py \
        --output benchmarks/results/a2a_b200.parquet

    # quick smoke (fewer sizes/iters):
    torchrun --nproc_per_node=8 benchmarks/a2a_benchmark.py --smoke

Multi-node (for the ep32 inter-node regime), from a 4-node allocation:
    torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:29500 benchmarks/a2a_benchmark.py \
        --output benchmarks/results/a2a_b200_4node.parquet

Only rank 0 writes the parquet (+ a .csv next to it).
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import asdict, dataclass

import pandas as pd
import torch
import torch.distributed as dist

# Total per-rank buffer sizes (bytes of the LOCAL dispatch buffer, i.e. the
# model's `size`). The 18b traces sit at 302 MB (mbs1) / 604 MB (mbs2) /
# 1208 MB (mbs4); 700m dispatch buffers are ~19 MB; include a small-message
# tail where latency and protocol switches dominate.
DEFAULT_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 302, 512, 604, 1024, 1208]
SMOKE_SIZES_MB = [8, 64, 302]

# The exact 18b trace points (seq*mbs*topk*hidden bf16 dispatch buffers) --
# always included so the model's cross-check test can pin them.
TRACE_POINTS_MB = [302, 604, 1208]


@dataclass
class BenchRow:
    device: str
    gpu_name: str
    world_size: int
    group_size: int
    inter_node: bool
    buffer_mb: float
    buffer_bytes: int
    dtype: str
    n_iter: int
    time_ms_p50: float
    time_ms_p10: float
    time_ms_p90: float
    # effective per-GPU egress bandwidth: (bytes * (n-1)/n) / t
    egress_gbps_p50: float
    egress_gbps_p90t: float  # bandwidth at the p90 (slow-tail) time
    # efficiency vs the link peak the model would charge (900 GB/s NVLink uni
    # for intra-node B200; EFA for inter-node -- filled in analysis, not here)
    status: str


def _detect_device_label() -> str:
    name = torch.cuda.get_device_name(0).lower()
    for tag in ("b200", "h200", "h100", "a100"):
        if tag in name:
            return tag
    return name.replace(" ", "_")


def bench_one(
    group: dist.ProcessGroup | None,
    group_size: int,
    buffer_bytes: int,
    n_warmup: int,
    n_iter: int,
) -> tuple[float, float, float]:
    """Time all_to_all_single on `group`; returns (p50, p10, p90) ms."""
    numel = buffer_bytes // 2  # bf16
    # round down to a multiple of group_size so the split is even
    numel = (numel // group_size) * group_size
    send = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
    recv = torch.empty_like(send)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(n_warmup):
        dist.all_to_all_single(recv, send, group=group)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_iter):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start.record()
        dist.all_to_all_single(recv, send, group=group)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    p50 = statistics.median(times_ms)
    p10 = times_ms[max(0, int(0.10 * len(times_ms)) - 1)]
    p90 = times_ms[min(len(times_ms) - 1, int(0.90 * len(times_ms)))]
    del send, recv
    torch.cuda.empty_cache()
    return p50, p10, p90


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/a2a_timings.parquet")
    parser.add_argument(
        "--sizes-mb",
        type=float,
        nargs="*",
        default=None,
        help="Override the size sweep (per-rank buffer, MB).",
    )
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Override group sizes (default: 2,4,8 capped at world).",
    )
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="3 sizes, 5 iters.")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    torch.cuda.set_device(local_rank)

    gpus_per_node = torch.cuda.device_count()
    n_nodes = max(1, world // gpus_per_node)

    device_label = _detect_device_label()
    gpu_name = torch.cuda.get_device_name(0)

    sizes_mb = args.sizes_mb or (SMOKE_SIZES_MB if args.smoke else DEFAULT_SIZES_MB)
    sizes_mb = sorted(set(sizes_mb) | set(TRACE_POINTS_MB if not args.smoke else []))
    n_iter = 5 if args.smoke else args.n_iter

    group_sizes = args.group_sizes or [g for g in (2, 4, 8, 16, 32, world) if g <= world]
    group_sizes = sorted(set(group_sizes))

    if rank == 0:
        print(f"world={world} nodes={n_nodes} device={device_label} ({gpu_name})")
        print(f"group sizes {group_sizes}, sizes {sizes_mb} MB, {n_iter} iters")

    rows: list[BenchRow] = []
    for gsize in group_sizes:
        # ranks [0, gsize) form the measured group; everyone else idles.
        group_ranks = list(range(gsize))
        group = dist.new_group(ranks=group_ranks)
        in_group = rank < gsize
        # a group spans nodes iff it needs GPUs from >1 node
        inter_node = gsize > gpus_per_node

        for size_mb in sizes_mb:
            buffer_bytes = int(size_mb * 1e6)
            p50 = p10 = p90 = float("nan")
            status = "OK"
            if in_group:
                try:
                    p50, p10, p90 = bench_one(group, gsize, buffer_bytes, args.n_warmup, n_iter)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    status = f"FAILED: {type(e).__name__}: {str(e)[:80]}"
            dist.barrier()  # keep out-of-group ranks in step

            if rank == 0:
                egress_bytes = buffer_bytes * (gsize - 1) / gsize
                row = BenchRow(
                    device=device_label,
                    gpu_name=gpu_name,
                    world_size=world,
                    group_size=gsize,
                    inter_node=inter_node,
                    buffer_mb=size_mb,
                    buffer_bytes=buffer_bytes,
                    dtype="bf16",
                    n_iter=n_iter,
                    time_ms_p50=p50,
                    time_ms_p10=p10,
                    time_ms_p90=p90,
                    egress_gbps_p50=(egress_bytes / (p50 / 1e3) / 1e9)
                    if p50 == p50 and p50 > 0
                    else float("nan"),
                    egress_gbps_p90t=(egress_bytes / (p90 / 1e3) / 1e9)
                    if p90 == p90 and p90 > 0
                    else float("nan"),
                    status=status,
                )
                rows.append(row)
                print(
                    f"  n={gsize} {size_mb:7.0f} MB  p50={p50:8.3f} ms  "
                    f"egress={row.egress_gbps_p50:6.1f} GB/s  {status}"
                )

        dist.destroy_process_group(group)

    if rank == 0:
        df = pd.DataFrame([asdict(r) for r in rows])
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_parquet(args.output, index=False)
        df.to_csv(os.path.splitext(args.output)[0] + ".csv", index=False)
        print(f"\nwrote {args.output}  ({len(df)} rows)")
        ok = df[(df.status == "OK") & (df.group_size == min(8, world))]
        big = ok[ok.buffer_mb >= 256]
        if len(big):
            print(
                f"8-rank egress @ >=256MB: p50 median "
                f"{big.egress_gbps_p50.median():.0f} GB/s "
                f"(trace cross-check: 324-366 GB/s on the 18b runs)"
            )

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
