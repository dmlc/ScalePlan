"""Utilities for modeling computation-communication overlap efficiency."""


def calculate_dp_overlap_efficiency(
    microbatch_compute_time: float,
    comm_time_per_bucket: float,
    n_buckets: int,
    is_first_pp_stage: bool,
) -> float:
    """
    Calculate DP communication overlap efficiency.

    Physics: Overlap only possible during compute windows.
    For gradient bucketing, each bucket can overlap with compute
    if compute_time >= comm_time for that bucket.

    During backward pass:
    - Gradients are computed layer by layer
    - Each gradient bucket's reduce-scatter can overlap with subsequent layer computations
    - First PP stage: After completing backward, no more compute remains to overlap with
      the final bucket's communication, leading to exposed communication time
    - Other PP stages: Continue processing other microbatches, providing compute to overlap with

    Args:
        microbatch_compute_time: Total compute time for a microbatch (forward + backward)
        comm_time_per_bucket: Communication time for a single bucket (AG or RS)
        n_buckets: Number of gradient/parameter buckets
        is_first_pp_stage: Whether this is the first pipeline parallel stage

    Returns:
        Fraction of comm time that is overlapped [0.0, 1.0]
        - 0.0 = fully exposed (no overlap)
        - 1.0 = fully overlapped (no exposed time)
    """
    if is_first_pp_stage:
        # First PP stage has no backward compute to overlap with for the final buckets
        # After completing its backward pass, it must wait for DP communication to complete
        return 0.0

    # Available overlap window per bucket
    # Assumes communication is interleaved with computation across buckets
    compute_window_per_bucket = microbatch_compute_time / n_buckets

    # Calculate exposed time per bucket
    # If comm_time > compute_window, the excess is exposed
    exposed_time_per_bucket = max(0.0, comm_time_per_bucket - compute_window_per_bucket)

    # Total overlap efficiency
    total_comm_time = comm_time_per_bucket * n_buckets
    total_exposed_time = exposed_time_per_bucket * n_buckets

    if total_comm_time == 0:
        return 1.0

    return 1.0 - (total_exposed_time / total_comm_time)


def calculate_exposed_dp_time(
    dp_comm_time_per_bucket: float,
    n_buckets: int,
    microbatch_compute_time: float,
    pp_degree: int,
) -> float:
    """
    Calculate the exposed (non-overlapped) DP communication time.

    This accounts for the fact that the first PP stage has fully exposed communication,
    while other stages can overlap communication with computation.

    Args:
        dp_comm_time_per_bucket: Communication time per bucket (seconds)
        n_buckets: Number of buckets
        microbatch_compute_time: Compute time per microbatch (seconds)
        pp_degree: Pipeline parallelism degree

    Returns:
        Exposed communication time (seconds)
    """
    total_comm_time = dp_comm_time_per_bucket * n_buckets

    # Calculate overlap efficiency for non-first stages
    overlap_efficiency_other_stages = calculate_dp_overlap_efficiency(
        microbatch_compute_time=microbatch_compute_time,
        comm_time_per_bucket=dp_comm_time_per_bucket,
        n_buckets=n_buckets,
        is_first_pp_stage=False,
    )

    # First stage: fully exposed (efficiency = 0.0)
    # Other stages: use calculated overlap efficiency
    # Weight by number of stages
    if pp_degree == 1:
        # No pipeline parallelism - treat as first stage
        avg_overlap_efficiency = 0.0
    else:
        # Average across all PP stages
        avg_overlap_efficiency = (
            0.0 * 1 + overlap_efficiency_other_stages * (pp_degree - 1)
        ) / pp_degree

    exposed_time = total_comm_time * (1.0 - avg_overlap_efficiency)
    return exposed_time
