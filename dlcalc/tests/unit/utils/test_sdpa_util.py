"""Unit tests for sdpa_util.py (measured SDPA fwd/bwd lookup)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from dlcalc.utils.hardware import DType, MachineSpec
from dlcalc.utils.sdpa_util import (
    MACHINE_SPEC_TO_GPU_MODEL,
    _lookup_closest,
    get_sdpa_bwd_time_s,
    get_sdpa_fwd_time_s,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A tiny parquet-like frame covering 2 shapes and 2 devices."""
    return pd.DataFrame([
        # A100, seq=2048
        dict(device="a100", dtype="bf16", backend="te",
             seq_len=2048, micro_bs=1, n_q_heads=8, n_kv_heads=1, head_dim=128,
             fwd_time_ms_med=0.30, bwd_time_ms_med=0.31, status="OK"),
        dict(device="a100", dtype="bf16", backend="te",
             seq_len=2048, micro_bs=1, n_q_heads=64, n_kv_heads=8, head_dim=128,
             fwd_time_ms_med=0.60, bwd_time_ms_med=1.40, status="OK"),
        # B200, same shapes
        dict(device="b200", dtype="bf16", backend="te",
             seq_len=2048, micro_bs=1, n_q_heads=8, n_kv_heads=1, head_dim=128,
             fwd_time_ms_med=0.20, bwd_time_ms_med=0.60, status="OK"),
        dict(device="b200", dtype="bf16", backend="te",
             seq_len=2048, micro_bs=1, n_q_heads=64, n_kv_heads=8, head_dim=128,
             fwd_time_ms_med=0.55, bwd_time_ms_med=2.17, status="OK"),
    ])


class TestLookupClosest:
    def test_exact_match_wins(self, sample_df: pd.DataFrame) -> None:
        row = _lookup_closest(
            sample_df, device="a100", dtype="bf16",
            seq_len=2048, micro_bs=1, n_q_heads=64, n_kv_heads=8, head_dim=128,
        )
        assert row is not None
        assert row["bwd_time_ms_med"] == 1.40

    def test_closest_match_for_unknown_shape(self, sample_df: pd.DataFrame) -> None:
        # Ask for 60 heads - closer to 64 than to 8
        row = _lookup_closest(
            sample_df, device="a100", dtype="bf16",
            seq_len=2048, micro_bs=1, n_q_heads=60, n_kv_heads=8, head_dim=128,
        )
        assert row["n_q_heads"] == 64

    def test_unknown_device_returns_none(self, sample_df: pd.DataFrame) -> None:
        assert _lookup_closest(
            sample_df, device="mi300", dtype="bf16",
            seq_len=2048, micro_bs=1, n_q_heads=8, n_kv_heads=1, head_dim=128,
        ) is None


class TestGetSdpaBwdTime:
    def test_returns_seconds(self, monkeypatch, sample_df: pd.DataFrame) -> None:
        monkeypatch.setattr(
            "dlcalc.utils.sdpa_util._load_sdpa_timings", lambda: sample_df
        )
        spec = MagicMock(spec=MachineSpec)
        spec.name = "p4d.24xlarge"
        t = get_sdpa_bwd_time_s(
            seq_len=2048, micro_bs=1, n_q_heads=64, n_kv_heads=8, head_dim=128,
            machine_spec=spec, dtype=DType.BF16,
        )
        assert t == pytest.approx(1.40e-3)

    def test_unknown_machine_returns_none(self, monkeypatch, sample_df: pd.DataFrame) -> None:
        monkeypatch.setattr(
            "dlcalc.utils.sdpa_util._load_sdpa_timings", lambda: sample_df
        )
        spec = MagicMock(spec=MachineSpec)
        spec.name = "unobtainium.12xlarge"
        assert get_sdpa_bwd_time_s(
            seq_len=2048, micro_bs=1, n_q_heads=64, n_kv_heads=8, head_dim=128,
            machine_spec=spec, dtype=DType.BF16,
        ) is None

    def test_fp32_falls_back_to_bf16(self, monkeypatch, sample_df: pd.DataFrame) -> None:
        monkeypatch.setattr(
            "dlcalc.utils.sdpa_util._load_sdpa_timings", lambda: sample_df
        )
        spec = MagicMock(spec=MachineSpec)
        spec.name = "p4d.24xlarge"
        # FP32 -> "bf16" in DTYPE_TO_STR, should hit the bf16 rows
        t = get_sdpa_bwd_time_s(
            seq_len=2048, micro_bs=1, n_q_heads=64, n_kv_heads=8, head_dim=128,
            machine_spec=spec, dtype=DType.FP32,
        )
        assert t == pytest.approx(1.40e-3)


class TestGetSdpaFwdTime:
    def test_same_row_as_bwd(self, monkeypatch, sample_df: pd.DataFrame) -> None:
        monkeypatch.setattr(
            "dlcalc.utils.sdpa_util._load_sdpa_timings", lambda: sample_df
        )
        spec = MagicMock(spec=MachineSpec)
        spec.name = "p6-b200.48xlarge"
        t_fwd = get_sdpa_fwd_time_s(
            seq_len=2048, micro_bs=1, n_q_heads=64, n_kv_heads=8, head_dim=128,
            machine_spec=spec, dtype=DType.BF16,
        )
        t_bwd = get_sdpa_bwd_time_s(
            seq_len=2048, micro_bs=1, n_q_heads=64, n_kv_heads=8, head_dim=128,
            machine_spec=spec, dtype=DType.BF16,
        )
        assert t_fwd == pytest.approx(0.55e-3)
        assert t_bwd == pytest.approx(2.17e-3)


class TestMachineSpecMapping:
    def test_covers_common_instance_types(self) -> None:
        for name in ["p4d.24xlarge", "p5.48xlarge", "p5en.48xlarge", "p6-b200.48xlarge"]:
            assert name in MACHINE_SPEC_TO_GPU_MODEL
