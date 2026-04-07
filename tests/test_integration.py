"""End-to-end integration tests using a fake trainer subprocess.

No MLX or GPU required. The fake trainer prints MLX-format loss lines,
writes checkpoint files, and injects NaN at a configurable iteration.
These tests prove the full recovery loop works through real process
management, not mocked internals.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

FAKE_TRAINER = str(Path(__file__).parent / "fake_trainer.py")


def run_nan_safe(tmp_path: Path, *, experts: list[str], nan_at_iter: int = 0,
                 max_seed_attempts: int = 3, iters: int = 50,
                 save_every: int = 10) -> dict:
    """Run nan-safe-train against the fake trainer and return the receipt."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "model": "fake/test-model",
        "iters": iters,
        "seed": 42,
        "save_every": save_every,
    }))

    for expert in experts:
        data_dir = tmp_path / "data" / expert
        data_dir.mkdir(parents=True)
        (data_dir / "train.jsonl").write_text("")
        (data_dir / "valid.jsonl").write_text("")

    receipt_path = tmp_path / "receipt.json"
    env = dict(os.environ)
    # Ensure src-layout package is discoverable by subprocess
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    if nan_at_iter > 0:
        env["NAN_AT_ITER"] = str(nan_at_iter)
    else:
        env.pop("NAN_AT_ITER", None)

    result = subprocess.run(
        [
            sys.executable, "-m", "nan_safe_trainer",
            "--config", str(config_path),
            "--experts", *experts,
            "--data-dir", str(tmp_path / "data"),
            "--adapter-dir", str(tmp_path / "adapters"),
            "--receipt-path", str(receipt_path),
            "--max-seed-attempts", str(max_seed_attempts),
            "--save-every", str(save_every),
            "--trainer-cmd", sys.executable, FAKE_TRAINER,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert receipt_path.exists(), f"No receipt written. stderr:\n{result.stderr}"
    return json.loads(receipt_path.read_text())


class TestEndToEnd:
    def test_clean_run_no_nan(self, tmp_path):
        """Training completes without NaN on first attempt."""
        receipt = run_nan_safe(tmp_path, experts=["clean_expert"])

        assert receipt["status"] == "completed"
        expert = receipt["experts"]["clean_expert"]
        assert expert["status"] == "completed"
        assert len(expert["attempts"]) == 1
        assert expert["attempts"][0]["nan_detected"] is False

        # Final adapter exists
        adapter = tmp_path / "adapters" / "clean_expert" / "adapters.safetensors"
        assert adapter.exists()

    def test_nan_recovery_and_completion(self, tmp_path):
        """NaN at iter 30, recovery from checkpoint 20, completes on retry."""
        receipt = run_nan_safe(
            tmp_path, experts=["recover_expert"],
            nan_at_iter=30, save_every=10, iters=50,
        )

        expert = receipt["experts"]["recover_expert"]

        # First attempt hit NaN
        attempt_1 = expert["attempts"][0]
        assert attempt_1["nan_detected"] is True
        assert attempt_1["seed"] == 42

        # Should have recovered and retried
        assert len(expert["attempts"]) >= 2
        attempt_2 = expert["attempts"][1]
        assert attempt_2["seed"] == 43  # seed incremented

        # Overall should complete
        assert expert["status"] == "completed"
        assert receipt["status"] == "completed"

    def test_receipt_structure_complete(self, tmp_path):
        """Receipt has all required top-level and per-expert fields."""
        receipt = run_nan_safe(tmp_path, experts=["schema_expert"])

        # Top-level
        assert "status" in receipt
        assert "started_at" in receipt
        assert "finished_at" in receipt
        assert "config_path" in receipt
        assert "experts_requested" in receipt
        assert receipt["experts_requested"] == ["schema_expert"]

        # Per-expert
        expert = receipt["experts"]["schema_expert"]
        assert "status" in expert
        assert "attempts" in expert
        assert "started_at" in expert
        assert "finished_at" in expert

        # Per-attempt
        attempt = expert["attempts"][0]
        assert "seed" in attempt
        assert "nan_detected" in attempt
        assert "started_at" in attempt

    def test_multiple_experts_sequential(self, tmp_path):
        """Multiple experts train sequentially, each gets its own receipt entry."""
        receipt = run_nan_safe(
            tmp_path, experts=["expert_a", "expert_b"],
        )

        assert receipt["status"] == "completed"
        assert "expert_a" in receipt["experts"]
        assert "expert_b" in receipt["experts"]
        assert receipt["experts"]["expert_a"]["status"] == "completed"
        assert receipt["experts"]["expert_b"]["status"] == "completed"

    def test_checkpoint_files_managed(self, tmp_path):
        """After NaN recovery, adapter dir has checkpoint files."""
        receipt = run_nan_safe(
            tmp_path, experts=["ckpt_expert"],
            nan_at_iter=30, save_every=10, iters=50,
        )

        adapter_dir = tmp_path / "adapters" / "ckpt_expert"
        assert adapter_dir.exists()
        # Should have the final adapter from the successful retry
        assert (adapter_dir / "adapters.safetensors").exists()

    def test_seed_rotation_visible_in_receipt(self, tmp_path):
        """Receipt shows seed incrementing across NaN recovery attempts."""
        receipt = run_nan_safe(
            tmp_path, experts=["seed_expert"],
            nan_at_iter=30, save_every=10, iters=50,
        )

        expert = receipt["experts"]["seed_expert"]
        seeds = [a["seed"] for a in expert["attempts"]]
        # First seed is 42 (from config), should increment after NaN
        assert seeds[0] == 42
        if len(seeds) > 1:
            assert seeds[1] == 43
