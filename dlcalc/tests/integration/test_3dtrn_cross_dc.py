"""Integration tests for 3dtrn with cross-DC support."""

import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


def run_3dtrn(config_file: str, timeout: int = 30) -> tuple[float | None, float | None, str]:
    """Run 3dtrn and extract MFU and total memory values from output.

    Args:
        config_file: Path to YAML configuration file
        timeout: Maximum time to wait for command completion

    Returns:
        Tuple of (MFU percentage or None, total memory in GiB or None, full output)
    """
    try:
        result = subprocess.run(
            ["3dtrn", config_file], capture_output=True, text=True, timeout=timeout
        )

        output = result.stdout + result.stderr

        # Find MFU in output - looking for "Theoretical MFU: XX.XX%" (with potential ANSI codes)
        # Remove ANSI escape codes first for easier matching
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", output)
        mfu_match = re.search(r"Theoretical MFU:\s*([0-9]+\.[0-9]+)%", clean_output)
        mfu_value = float(mfu_match.group(1)) if mfu_match else None

        # Find total memory in output - looking for "Total Memory Required: XX.XXX GiB"
        mem_match = re.search(r"Total Memory Required:\s*([0-9]+\.[0-9]+)\s*GiB", clean_output)
        mem_value = float(mem_match.group(1)) if mem_match else None

        return mfu_value, mem_value, output

    except subprocess.TimeoutExpired:
        pytest.fail(f"Timeout ({timeout}s) running {config_file}")
    except FileNotFoundError:
        pytest.skip("3dtrn command not found - install package first")
    except Exception as e:
        pytest.fail(f"Error running {config_file}: {e}")


class TestCrossDCIntegration:
    """Integration tests for cross-DC training performance modeling."""

    def test_cross_dc_config_parsing(self):
        """Test that 3dtrn correctly parses and uses cross_dc configuration."""
        # Create a test config with cross-DC parameters
        config = {
            "model": {
                "n_layers": 32,
                "hidden_sz": 4096,
                "inter_sz": 11008,
                "n_q_heads": 32,
                "n_kv_heads": 8,
                "head_dim": 128,
                "vocab_sz": 32000,
                "glu": True,
                "rotary_embeds": True,
                "dropout": False,
                "tie_embeddings": True,
            },
            "parallelism": {
                "tp": 8,
                "pp": 2,
                "dp": 16,
                "vpp": 1,
                "sp": True,
                "zero_level": 1,
                "n_param_buckets": 5,
            },
            "performance": {
                "activation_checkpointing_type": "selective",
            },
            "data": {
                "gbs": 256,
                "seqlen": 2048,
                "microbatch_sz": 1,
            },
            "hardware": {
                "node_type": "p5.48xlarge",
            },
            "cross_dc": {
                "n_dcs": 3,
                "interconnect_bandwidth_gbps": 800,
                "interconnect_latency_s": 0.0035,
            },
        }

        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Run 3dtrn with cross-DC config
            mfu, memory, output = run_3dtrn(config_path)

            # Check that output contains cross-DC information
            assert "cross-DC" in output or "Cross-DC" in output

            # MFU should be lower than baseline due to cross-DC overhead
            # (We can't check exact values without running baseline, but it should be reasonable)
            assert mfu is not None
            assert 0 < mfu < 100  # MFU should be between 0 and 100%

        finally:
            # Clean up temporary file
            Path(config_path).unlink()

    def test_cross_dc_degradation_reporting(self):
        """Test that cross-DC configuration shows degradation in output."""
        # Configuration with cross-DC
        config = {
            "model": {
                "n_layers": 32,
                "hidden_sz": 4096,
                "inter_sz": 11008,
                "n_q_heads": 32,
                "n_kv_heads": 8,
                "head_dim": 128,
                "vocab_sz": 32000,
                "glu": True,
                "rotary_embeds": True,
                "dropout": False,
                "tie_embeddings": True,
            },
            "parallelism": {
                "tp": 8,
                "pp": 2,
                "dp": 16,
                "vpp": 1,
                "sp": True,
                "zero_level": 1,
                "n_param_buckets": 5,
            },
            "performance": {
                "activation_checkpointing_type": "selective",
            },
            "data": {
                "gbs": 256,
                "seqlen": 2048,
                "microbatch_sz": 1,
            },
            "hardware": {
                "node_type": "p5.48xlarge",
            },
            "cross_dc": {
                "n_dcs": 3,
                "interconnect_bandwidth_gbps": 800,
                "interconnect_latency_s": 0.0035,
            },
        }

        # Run with cross-DC
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            mfu, _, output = run_3dtrn(config_path)
        finally:
            Path(config_path).unlink()

        # Verify cross-DC impact is reported
        assert mfu is not None
        assert "Cross-DC Impact" in output or "cross-DC" in output
        assert "degradation" in output.lower() or "slower" in output.lower()

        # Should show specific cross-DC metrics
        assert "Cross-DC Configuration" in output or "cross_dc" in output
        assert "800" in output  # bandwidth
        assert "3.5" in output or "0.0035" in output  # latency

    def test_cross_dc_output_format(self):
        """Test that cross-DC output includes expected information."""
        config = {
            "model": {
                "n_layers": 16,
                "hidden_sz": 2048,
                "inter_sz": 5504,
                "n_q_heads": 16,
                "n_kv_heads": 8,
                "head_dim": 128,
                "vocab_sz": 32000,
                "glu": True,
                "rotary_embeds": True,
                "dropout": False,
                "tie_embeddings": True,
            },
            "parallelism": {
                "tp": 4,
                "pp": 2,
                "dp": 8,
                "vpp": 1,
                "sp": True,
                "zero_level": 1,
                "n_param_buckets": 5,
            },
            "performance": {
                "activation_checkpointing_type": "selective",
            },
            "data": {
                "gbs": 64,
                "seqlen": 1024,
                "microbatch_sz": 1,
            },
            "hardware": {
                "node_type": "p4d.24xlarge",
            },
            "cross_dc": {
                "n_dcs": 2,
                "interconnect_bandwidth_gbps": 400,
                "interconnect_latency_s": 0.005,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            mfu, memory, output = run_3dtrn(config_path)

            # Check for expected cross-DC output elements
            assert "Cross-DC Configuration" in output or "cross_dc" in output
            assert "degradation" in output.lower()

            # Should show bandwidth and latency information
            assert "800" in output or "400" in output  # bandwidth
            assert "0.005" in output or "5" in output  # latency

        finally:
            Path(config_path).unlink()
