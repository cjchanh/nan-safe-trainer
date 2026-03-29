"""Typed receipt contracts for nan-safe training runs.

These dataclasses define the stable fields in the JSON receipt.
Use them for deserialization and validation of receipt files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttemptRecord:
    """One training attempt within an expert's run."""
    attempt: int
    seed: int
    nan_detected: bool
    status: str
    started_at: str
    finished_at: str | None = None
    resume_from_iter: int = 0
    resume_from_file: str | None = None
    remaining_target_iters: int = 0
    exit_code: int | None = None
    nan_detected_at_iter: int | None = None
    last_finite_train_loss: float | None = None
    last_finite_train_iter: int | None = None
    best_logged_val_loss: float | None = None
    best_logged_val_iter: int | None = None
    last_logged_val_loss: float | None = None
    last_logged_val_iter: int | None = None
    recovery_checkpoint_iter: int | None = None
    recovery_checkpoint_file: str | None = None
    final_adapter_file: str | None = None
    effective_total_iters: int | None = None
    log_path: str | None = None
    pid: int | None = None
    failure_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttemptRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ExpertResult:
    """Result for a single expert across all retry attempts."""
    status: str
    adapter_path: str
    target_iters: int
    threshold_val_loss: float
    started_at: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    finished_at: str | None = None
    seed_used: int | None = None
    best_checkpoint_iter: int | None = None
    resume_checkpoint_iter: int | None = None
    final_val_loss: float | None = None
    failure_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpertResult:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        attempts = [AttemptRecord.from_dict(a) for a in data.get("attempts", [])]
        filtered = {k: v for k, v in data.items() if k in known and k != "attempts"}
        return cls(attempts=attempts, **filtered)


@dataclass
class TrainingReceipt:
    """Top-level receipt for a nan-safe training run."""
    status: str
    started_at: str
    config_path: str
    adapter_dir: str
    data_dir: str
    save_every: int
    max_seed_attempts: int
    threshold: float
    experts_requested: list[str]
    experts: dict[str, ExpertResult] = field(default_factory=dict)
    finished_at: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingReceipt:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        experts = {
            k: ExpertResult.from_dict(v)
            for k, v in data.get("experts", {}).items()
        }
        filtered = {k: v for k, v in data.items() if k in known and k != "experts"}
        return cls(experts=experts, **filtered)

    @classmethod
    def load(cls, path: str) -> TrainingReceipt:
        import json
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text()))
