"""Benchmark NCCL all_reduce / all_gather / reduce_scatter at DP bucket sizes.

Motivation:
    dlcalc/utils/comms.py prices DP gradient reduce-scatter / param all-gather
    with ring-algorithm equations at protocol line efficiencies (LL128 0.84375
    etc.) on top of the EFA/NVLink peaks. The 18b trace audit shows modeled DP
    collective wire time at ~0.5-0.7x the quiet-step measured buckets, and the
    700m runs (world 32 = exactly one of this pod's subgroup sizes) are the
    worst golden-set regime. This script measures the real end-to-end
    bandwidth of the three collectives vs (message size x group size) on the
    same fabric/NCCL build the benchmark jobs use, so the model's efficiency
    factors come from data (GUIDELINES §1).

What it measures:
    For each group size (8 = intra-node; 16/32 = spanning 2/4 nodes) and each
    payload size, CUDA-event-timed p50/p10/p90 of:
      * all_reduce(bf16)          — zero_level=0 gradient path
      * reduce_scatter_tensor     — zero_level=1 gradient bucket path
      * all_gather_into_tensor    — zero_level=1 param bucket path
    Payload = the FULL unsharded tensor (the model's `size`); algbw and the
    ring-equation busbw are both reported:
      all_reduce:      busbw = 2*(n-1)/n * size / t
      reduce_scatter:  busbw =   (n-1)/n * size / t
      all_gather:      busbw =   (n-1)/n * size / t

Usage (all 4 workers simultaneously; PET_* env provides the rendezvous):
    torchrun benchmarks/dp_collectives_benchmark.py \
        --output /scratch/$USER/dp_collectives_b200.parquet

    # single node:
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
        benchmarks/dp_collectives_benchmark.py --smoke

Only rank 0 writes the parquet (+ .csv).
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import asdict, dataclass

import pandas as pd
import torch
import torch.distributed as dist

# Full-tensor payload sizes. Golden-set DP buckets: 5p3b/18b grad buckets are
# ~0.5-2.3 GiB unsharded (params/8 buckets * 2-4 B); 700m buckets are ~50-200MB.
# Include a small tail for protocol-switch behavior.
DEFAULT_SIZES_MB = [4, 16, 64, 128, 256, 512, 1024, 2048]
SMOKE_SIZES_MB = [64, 512]

COLLECTIVES = ("all_reduce", "reduce_scatter", "all_gather")


@dataclass
class BenchRow:
    device: str
    gpu_name: str
    world_size: int
    group_size: int
    inter_node: bool
    collective: str
    payload_mb: float
    payload_bytes: int
    dtype: str
    n_iter: int
    time_ms_p50: float
    time_ms_p10: float
    time_ms_p90: float
    algbw_gbps_p50: float  # payload / t
    busbw_gbps_p50: float  # ring-equation bus bandwidth
    status: str


def _detect_device_label() -> str:
    name = torch.cuda.get_device_name(0).lower()
    for tag in ("b200", "h200", "h100", "a100"):
        if tag in name:
            return tag
    return name.replace(" ", "_")


def _bus_factor(collective: str, n: int) -> float:
    if collective == "all_reduce":
        return 2.0 * (n - 1) / n
    return (n - 1) / n  # reduce_scatter / all_gather


def bench_one(
    group: dist.ProcessGroup,
    group_size: int,
    collective: str,
    payload_bytes: int,
    n_warmup: int,
    n_iter: int,
) -> tuple[float, float, float]:
    numel = payload_bytes // 2  # bf16
    numel = (numel // group_size) * group_size
    full = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
    shard = torch.empty(numel // group_size, dtype=torch.bfloat16, device="cuda")

    if collective == "all_reduce":

        def op() -> None:
            dist.all_reduce(full, group=group)
    elif collective == "reduce_scatter":

        def op() -> None:
            dist.reduce_scatter_tensor(shard, full, group=group)
    elif collective == "all_gather":

        def op() -> None:
            dist.all_gather_into_tensor(full, shard, group=group)
    else:
        raise ValueError(collective)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(n_warmup):
        op()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_iter):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start.record()
        op()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    p50 = statistics.median(times_ms)
    p10 = times_ms[max(0, int(0.10 * len(times_ms)) - 1)]
    p90 = times_ms[min(len(times_ms) - 1, int(0.90 * len(times_ms)))]
    # release the buffers (op's closure holds them; rebinding drops the refs)
    op = full = shard = None  # noqa: F841
    torch.cuda.empty_cache()
    return p50, p10, p90


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/dp_collectives_timings.parquet")
    parser.add_argument("--sizes-mb", type=float, nargs="*", default=None)
    parser.add_argument("--group-sizes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Rank stride for group construction. stride=1: contiguous (dense-DP"
        " groups, ring threads NVLink within each node). stride=8: one rank per"
        " node (expert-DP groups with ep=8 — every hop crosses EFA).",
    )
    parser.add_argument(
        "--concurrent-groups",
        action="store_true",
        help="With --stride s: run ALL s disjoint strided groups simultaneously"
        " (group g = ranks {g, g+s, g+2s, ...}), like real training where every"
        " GPU's expert-DP group reduces at once and the groups share each node's"
        " NIC. Reports rank0's group time (all groups are symmetric).",
    )
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
    torch.cuda.set_device(local_rank)

    gpus_per_node = torch.cuda.device_count()
    device_label = _detect_device_label()
    gpu_name = torch.cuda.get_device_name(0)

    sizes_mb = args.sizes_mb or (SMOKE_SIZES_MB if args.smoke else DEFAULT_SIZES_MB)
    n_iter = 5 if args.smoke else args.n_iter
    group_sizes = args.group_sizes or [g for g in (8, 16, world) if g <= world]
    group_sizes = sorted(set(group_sizes))

    if rank == 0:
        print(f"world={world} device={device_label} ({gpu_name})")
        print(f"groups {group_sizes}, sizes {sizes_mb} MB, {n_iter} iters")

    rows: list[BenchRow] = []
    for gsize in group_sizes:
        if args.concurrent_groups and args.stride > 1:
            # all `stride` disjoint groups exist and run at once; this rank
            # joins the group of its residue class. NCCL requires every rank
            # to call new_group for every group, in the same order.
            my_group = None
            for g0 in range(args.stride):
                ranks_g = [g0 + r * args.stride for r in range(gsize)]
                if ranks_g[-1] >= world:
                    my_group = None
                    break
                pg = dist.new_group(ranks=ranks_g)
                if rank in ranks_g:
                    my_group = pg
            if my_group is None:
                if rank == 0:
                    print(f"  skip n={gsize} stride={args.stride}: exceeds world")
                continue
            group = my_group
            in_group = True  # every rank belongs to exactly one residue group
            group_ranks = [rank % args.stride + r * args.stride for r in range(gsize)]
        else:
            group_ranks = [r * args.stride for r in range(gsize)]
            if group_ranks[-1] >= world:
                if rank == 0:
                    print(f"  skip n={gsize} stride={args.stride}: needs rank {group_ranks[-1]}")
                continue
            group = dist.new_group(ranks=group_ranks)
            in_group = rank in group_ranks
        inter_node = (gsize * args.stride) > gpus_per_node

        for collective in COLLECTIVES:
            for size_mb in sizes_mb:
                payload = int(size_mb * 1e6)
                p50 = p10 = p90 = float("nan")
                status = "OK"
                if in_group:
                    try:
                        p50, p10, p90 = bench_one(
                            group, gsize, collective, payload, args.n_warmup, n_iter
                        )
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        status = f"FAILED: {type(e).__name__}: {str(e)[:80]}"
                dist.barrier()

                if rank == 0:
                    bf = _bus_factor(collective, gsize)
                    ok_time = p50 == p50 and p50 > 0
                    rows.append(
                        BenchRow(
                            device=device_label,
                            gpu_name=gpu_name,
                            world_size=world,
                            group_size=gsize,
                            inter_node=inter_node,
                            collective=collective,
                            payload_mb=size_mb,
                            payload_bytes=payload,
                            dtype="bf16",
                            n_iter=n_iter,
                            time_ms_p50=p50,
                            time_ms_p10=p10,
                            time_ms_p90=p90,
                            algbw_gbps_p50=(payload / (p50 / 1e3) / 1e9)
                            if ok_time
                            else float("nan"),
                            busbw_gbps_p50=(payload * bf / (p50 / 1e3) / 1e9)
                            if ok_time
                            else float("nan"),
                            status=status,
                        )
                    )
                    r = rows[-1]
                    print(
                        f"  n={gsize:2d} {collective:14} {size_mb:7.0f} MB  "
                        f"p50={p50:8.3f} ms  busbw={r.busbw_gbps_p50:7.1f} GB/s  {status}"
                    )

        dist.destroy_process_group(group)

    if rank == 0:
        df = pd.DataFrame([asdict(r) for r in rows])
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_parquet(args.output, index=False)
        df.to_csv(os.path.splitext(args.output)[0] + ".csv", index=False)
        print(f"\nwrote {args.output}  ({len(df)} rows)")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
