"""Thin adapter over the canonical cost model in :mod:`dlcalc.training_3d`.

This module used to carry a full second copy of the 3dtrn cost model (~1000 lines
of duplicated physics) for the grid search. The copy drifted from 3dtrn more than
once -- the expert-GEMM 1/EP over-credit (Effect A), the expert/dense DP-reduction
split (Effect B), the shared-expert FFN (Effect C) -- each time silently skewing
search rankings until a parity test caught it. There is now ONE implementation:
``dlcalc.training_3d.calculate_training_metrics``, which runs the same code path as
the ``3dtrn`` CLI with the report suppressed.

Keep this module as the search/webapp-facing entry point (it pins the tuple return
shape those callers expect); put new physics in ``dlcalc``, never here.
"""

from typing import Any

from dlcalc.training_3d import calculate_training_metrics as _calculate_training_metrics


def calculate_training_metrics(
    cfg: dict[str, Any],
) -> tuple[float, float, float]:
    """Calculate training metrics using the canonical 3dtrn model.

    Args:
        cfg: Complete configuration dictionary

    Returns:
        Tuple of (mfu_percentage, iteration_time_s, memory_per_device_gb)
    """
    metrics = _calculate_training_metrics(cfg)
    return metrics.mfu_pct, metrics.iteration_time_s, metrics.memory_per_device_gb
