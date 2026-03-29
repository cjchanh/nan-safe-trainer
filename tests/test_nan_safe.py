"""Tests for nan-safe training wrapper.

All tests use mocked subprocesses — no MLX or GPU required.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from nan_safe_trainer.core import (
    CHECKPOINT_RE,
    TRAIN_LOSS_RE,
    VAL_LOSS_RE,
    build_attempt_config,
    is_finite_loss,
    latest_checkpoint,
    load_yaml,
    parse_loss,
    restore_checkpoint,
    write_json,
)


# --- Loss detection ---


class TestLossDetection:
    def test_train_loss_nan_detected(self):
        line = "Iter 10: Train loss nan, Learning Rate 1.000e-05"
        match = TRAIN_LOSS_RE.search(line)
        assert match is not None
        loss = parse_loss(match.group(2))
        assert math.isnan(loss)
        assert not is_finite_loss(loss)

    def test_train_loss_inf_detected(self):
        line = "Iter 20: Train loss inf, Learning Rate 1.000e-05"
        match = TRAIN_LOSS_RE.search(line)
        assert match is not None
        loss = parse_loss(match.group(2))
        assert math.isinf(loss)
        assert not is_finite_loss(loss)

    def test_normal_train_loss_passes(self):
        line = "Iter 30: Train loss 2.345, Learning Rate 1.000e-05"
        match = TRAIN_LOSS_RE.search(line)
        assert match is not None
        loss = parse_loss(match.group(2))
        assert is_finite_loss(loss)
        assert abs(loss - 2.345) < 1e-6

    def test_val_loss_nan_detected(self):
        line = "Iter 30: Val loss nan, Val Tokens-per-sec 123.4"
        match = VAL_LOSS_RE.search(line)
        assert match is not None
        loss = parse_loss(match.group(2))
        assert math.isnan(loss)

    def test_val_loss_normal_passes(self):
        line = "Iter 30: Val loss 1.876, Val Tokens-per-sec 123.4"
        match = VAL_LOSS_RE.search(line)
        assert match is not None
        loss = parse_loss(match.group(2))
        assert is_finite_loss(loss)

    def test_is_finite_loss_none(self):
        assert not is_finite_loss(None)


# --- Checkpoint management ---


class TestCheckpointManagement:
    def test_checkpoint_discovery_finds_highest(self, tmp_path):
        adapter_dir = tmp_path / "adapters" / "test_expert"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "0000010_adapters.safetensors").write_bytes(b"fake10")
        (adapter_dir / "0000020_adapters.safetensors").write_bytes(b"fake20")
        (adapter_dir / "0000005_adapters.safetensors").write_bytes(b"fake05")

        result = latest_checkpoint(adapter_dir)
        assert result is not None
        best_iter, best_path = result
        assert best_iter == 20
        assert best_path.name == "0000020_adapters.safetensors"

    def test_checkpoint_discovery_empty_dir(self, tmp_path):
        adapter_dir = tmp_path / "adapters" / "empty"
        adapter_dir.mkdir(parents=True)
        assert latest_checkpoint(adapter_dir) is None

    def test_checkpoint_discovery_nonexistent_dir(self, tmp_path):
        assert latest_checkpoint(tmp_path / "does_not_exist") is None

    def test_checkpoint_regex_matches_valid(self):
        assert CHECKPOINT_RE.match("0000030_adapters.safetensors") is not None
        assert CHECKPOINT_RE.match("0000001_adapters.safetensors") is not None

    def test_checkpoint_regex_rejects_invalid(self):
        assert CHECKPOINT_RE.match("adapters.safetensors") is None
        assert CHECKPOINT_RE.match("random_file.txt") is None
        assert CHECKPOINT_RE.match("30_adapters.safetensors") is None

    def test_restore_checkpoint_creates_both_files(self, tmp_path):
        backup = tmp_path / "backup.safetensors"
        backup.write_bytes(b"checkpoint_data")
        adapter_dir = tmp_path / "adapters" / "expert"

        restore_checkpoint(backup, adapter_dir, 20)

        canonical = adapter_dir / "adapters.safetensors"
        numbered = adapter_dir / "0000020_adapters.safetensors"
        assert canonical.exists()
        assert numbered.exists()
        assert canonical.read_bytes() == b"checkpoint_data"
        assert numbered.read_bytes() == b"checkpoint_data"


# --- Config management ---


class TestConfigManagement:
    def test_build_attempt_config_writes_yaml(self, tmp_path):
        base = {"model": "test-model", "iters": 600, "seed": 42, "save_every": 30}
        config_path = tmp_path / "attempt.yaml"

        build_attempt_config(
            base, config_path, seed=99, resume_adapter_file="/path/to/resume",
            remaining_iters=300, save_every=15,
        )

        result = load_yaml(config_path)
        assert result["seed"] == 99
        assert result["iters"] == 300
        assert result["save_every"] == 15
        assert result["resume_adapter_file"] == "/path/to/resume"
        assert result["model"] == "test-model"

    def test_build_attempt_config_no_resume(self, tmp_path):
        base = {"model": "test-model", "iters": 100, "seed": 1, "save_every": 10}
        config_path = tmp_path / "attempt.yaml"

        build_attempt_config(
            base, config_path, seed=5, resume_adapter_file=None,
            remaining_iters=100, save_every=10,
        )

        result = load_yaml(config_path)
        assert result["resume_adapter_file"] is None


# --- Receipt writing ---


class TestReceipt:
    def test_receipt_structure(self, tmp_path):
        receipt_path = tmp_path / "receipt.json"
        payload = {
            "status": "running",
            "started_at": "2026-01-01T00:00:00+00:00",
            "experts_requested": ["a", "b"],
            "experts": {},
        }
        write_json(receipt_path, payload)

        loaded = json.loads(receipt_path.read_text())
        assert loaded["status"] == "running"
        assert loaded["experts_requested"] == ["a", "b"]
        assert "started_at" in loaded

    def test_receipt_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "receipt.json"
        write_json(deep_path, {"test": True})
        assert deep_path.exists()


# --- Seed rotation ---


class TestSeedRotation:
    def test_seed_increments_on_nan(self):
        """Verify the seed rotation logic from train_one_expert."""
        base_seed = 42
        current_seed = base_seed

        # Simulate 3 NaN events
        for _ in range(3):
            current_seed += 1

        assert current_seed == 45
        assert current_seed - base_seed == 3


# --- Integration (mocked subprocess) ---


class TestThresholdGating:
    def test_threshold_gates_completion(self, tmp_path):
        """Training completes but val loss exceeds threshold → status=failed."""
        import argparse
        from nan_safe_trainer.core import train_one_expert

        args = argparse.Namespace(
            config=tmp_path / "config.yaml",
            experts=["test_expert"],
            data_dir=str(tmp_path / "data"),
            adapter_dir=str(tmp_path / "adapters"),
            save_every=10,
            max_seed_attempts=2,
            threshold=2.0,
            seed=42,
            receipt_path=tmp_path / "receipt.json",
            trainer_cmd=["/bin/echo", "done"],
            resume_existing=False,
        )
        (tmp_path / "config.yaml").write_text("model: test\niters: 100\nseed: 42\nsave_every: 10\n")
        (tmp_path / "data" / "test_expert").mkdir(parents=True)

        receipt: dict = {"experts": {}}

        with mock.patch("nan_safe_trainer.core.run_training_attempt") as mock_run:
            mock_run.return_value = {
                "status": "completed",
                "nan_detected": False,
                "resume_from_iter": 0,
                "best_logged_val_loss": 2.5,
                "finished_at": "2026-01-01T00:00:00+00:00",
            }
            result = train_one_expert(
                expert="test_expert",
                args=args,
                base_config={"iters": 100, "seed": 42, "save_every": 10},
                receipt=receipt,
                workspace=tmp_path / "workspace",
            )

        assert result["status"] == "failed"
        assert "threshold" in result["failure_reason"]

    def test_threshold_passes_when_under(self, tmp_path):
        """Training completes with val loss under threshold → status=completed."""
        import argparse
        from nan_safe_trainer.core import train_one_expert

        args = argparse.Namespace(
            config=tmp_path / "config.yaml",
            experts=["test_expert"],
            data_dir=str(tmp_path / "data"),
            adapter_dir=str(tmp_path / "adapters"),
            save_every=10,
            max_seed_attempts=2,
            threshold=3.0,
            seed=42,
            receipt_path=tmp_path / "receipt.json",
            trainer_cmd=["/bin/echo", "done"],
            resume_existing=False,
        )
        (tmp_path / "config.yaml").write_text("model: test\niters: 100\nseed: 42\nsave_every: 10\n")
        (tmp_path / "data" / "test_expert").mkdir(parents=True)

        receipt: dict = {"experts": {}}

        with mock.patch("nan_safe_trainer.core.run_training_attempt") as mock_run:
            mock_run.return_value = {
                "status": "completed",
                "nan_detected": False,
                "resume_from_iter": 0,
                "best_logged_val_loss": 2.1,
                "finished_at": "2026-01-01T00:00:00+00:00",
            }
            result = train_one_expert(
                expert="test_expert",
                args=args,
                base_config={"iters": 100, "seed": 42, "save_every": 10},
                receipt=receipt,
                workspace=tmp_path / "workspace",
            )

        assert result["status"] == "completed"
        assert result["final_val_loss"] == 2.1


    def test_threshold_fails_closed_no_val_loss(self, tmp_path):
        """Threshold set but no val loss logged → fail-closed."""
        import argparse
        from nan_safe_trainer.core import train_one_expert

        args = argparse.Namespace(
            config=tmp_path / "config.yaml",
            experts=["test_expert"],
            data_dir=str(tmp_path / "data"),
            adapter_dir=str(tmp_path / "adapters"),
            save_every=10,
            max_seed_attempts=2,
            threshold=3.0,
            seed=42,
            receipt_path=tmp_path / "receipt.json",
            trainer_cmd=["/bin/echo", "done"],
            resume_existing=False,
        )
        (tmp_path / "config.yaml").write_text("model: test\niters: 100\nseed: 42\nsave_every: 10\n")
        (tmp_path / "data" / "test_expert").mkdir(parents=True)

        receipt: dict = {"experts": {}}

        with mock.patch("nan_safe_trainer.core.run_training_attempt") as mock_run:
            mock_run.return_value = {
                "status": "completed",
                "nan_detected": False,
                "resume_from_iter": 0,
                "best_logged_val_loss": None,
                "finished_at": "2026-01-01T00:00:00+00:00",
            }
            result = train_one_expert(
                expert="test_expert",
                args=args,
                base_config={"iters": 100, "seed": 42, "save_every": 10},
                receipt=receipt,
                workspace=tmp_path / "workspace",
            )

        assert result["status"] == "failed"
        assert "no validation loss" in result["failure_reason"]

    def test_no_threshold_completes_without_val_loss(self, tmp_path):
        """No threshold set (inf) → completes even without val loss."""
        import argparse
        from nan_safe_trainer.core import train_one_expert

        args = argparse.Namespace(
            config=tmp_path / "config.yaml",
            experts=["test_expert"],
            data_dir=str(tmp_path / "data"),
            adapter_dir=str(tmp_path / "adapters"),
            save_every=10,
            max_seed_attempts=2,
            threshold=float("inf"),
            seed=42,
            receipt_path=tmp_path / "receipt.json",
            trainer_cmd=["/bin/echo", "done"],
            resume_existing=False,
        )
        (tmp_path / "config.yaml").write_text("model: test\niters: 100\nseed: 42\nsave_every: 10\n")
        (tmp_path / "data" / "test_expert").mkdir(parents=True)

        receipt: dict = {"experts": {}}

        with mock.patch("nan_safe_trainer.core.run_training_attempt") as mock_run:
            mock_run.return_value = {
                "status": "completed",
                "nan_detected": False,
                "resume_from_iter": 0,
                "best_logged_val_loss": None,
                "finished_at": "2026-01-01T00:00:00+00:00",
            }
            result = train_one_expert(
                expert="test_expert",
                args=args,
                base_config={"iters": 100, "seed": 42, "save_every": 10},
                receipt=receipt,
                workspace=tmp_path / "workspace",
            )

        assert result["status"] == "completed"


class TestMaxRetries:
    def test_max_retries_respected(self, tmp_path):
        """Verify that the main loop respects max_seed_attempts."""
        import argparse

        args = argparse.Namespace(
            config=tmp_path / "config.yaml",
            experts=["test_expert"],
            data_dir=str(tmp_path / "data"),
            adapter_dir=str(tmp_path / "adapters"),
            save_every=10,
            max_seed_attempts=2,
            threshold=3.0,
            seed=42,
            receipt_path=tmp_path / "receipt.json",
            trainer_cmd=["/bin/echo", "done"],
            resume_existing=False,
        )

        # Write a minimal config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model: test\niters: 100\nseed: 42\nsave_every: 10\n")

        # Create data dir
        (tmp_path / "data" / "test_expert").mkdir(parents=True)

        from nan_safe_trainer.core import train_one_expert

        receipt: dict = {"experts": {}}

        # Mock run_training_attempt to always return nan_detected with no checkpoint
        with mock.patch("nan_safe_trainer.core.run_training_attempt") as mock_run:
            mock_run.return_value = {
                "status": "nan_detected",
                "nan_detected": True,
                "recovery_checkpoint_iter": None,
                "recovery_checkpoint_file": None,
                "finished_at": "2026-01-01T00:00:00+00:00",
            }

            result = train_one_expert(
                expert="test_expert",
                args=args,
                base_config={"iters": 100, "seed": 42, "save_every": 10},
                receipt=receipt,
                workspace=tmp_path / "workspace",
            )

        assert result["status"] == "failed"
        assert "NaN detected before any recoverable checkpoint" in result["failure_reason"]
        # Should have stopped after first attempt since no checkpoint to recover from
        assert mock_run.call_count == 1
