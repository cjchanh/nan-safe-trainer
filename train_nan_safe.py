#!/usr/bin/env python3
"""NaN-safe wrapper for MLX LoRA training.

Monitors a training subprocess's stdout for NaN/Inf loss values.
When detected: kills the process, restores the last known-good checkpoint,
increments the random seed, and retries. Writes a structured JSON receipt
documenting every attempt and outcome.

Works with any MLX LoRA trainer that prints loss in the standard format:
    Iter N: Train loss X.XXX, ...
    Iter N: Val loss X.XXX, ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RECEIPT = Path("nan_safe_receipt.json")

TRAIN_LOSS_RE = re.compile(r"Iter (\d+): Train loss ([^,]+),")
VAL_LOSS_RE = re.compile(r"Iter (\d+): Val loss ([^,]+),")
SAVE_RE = re.compile(r"Iter (\d+): Saved adapter weights to ")
CHECKPOINT_RE = re.compile(r"^(\d{7})_adapters\.safetensors$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def parse_loss(token: str) -> float:
    return float(token.strip())


def is_finite_loss(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def latest_checkpoint(adapter_dir: Path) -> tuple[int, Path] | None:
    best_iter = -1
    best_path: Path | None = None
    if not adapter_dir.exists():
        return None
    for child in adapter_dir.iterdir():
        match = CHECKPOINT_RE.match(child.name)
        if not match or not child.is_file():
            continue
        checkpoint_iter = int(match.group(1))
        if checkpoint_iter > best_iter:
            best_iter = checkpoint_iter
            best_path = child
    if best_path is None:
        return None
    return best_iter, best_path


def copy_checkpoint(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def restore_checkpoint(backup_file: Path, adapter_dir: Path, checkpoint_iter: int) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    canonical = adapter_dir / "adapters.safetensors"
    numbered = adapter_dir / f"{checkpoint_iter:07d}_adapters.safetensors"
    shutil.copy2(backup_file, canonical)
    shutil.copy2(backup_file, numbered)


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def build_attempt_config(
    base_config: dict[str, Any],
    config_path: Path,
    *,
    seed: int,
    resume_adapter_file: str | None,
    remaining_iters: int,
    save_every: int,
) -> Path:
    payload = dict(base_config)
    payload["seed"] = seed
    payload["resume_adapter_file"] = resume_adapter_file
    payload["iters"] = remaining_iters
    payload["save_every"] = save_every
    payload["steps_per_eval"] = save_every
    save_yaml(config_path, payload)
    return config_path


def initial_resume_state(
    expert: str,
    *,
    adapter_dir: Path,
    target_iters: int,
    resume_existing: bool,
    workspace: Path,
) -> dict[str, Any]:
    if not resume_existing:
        return {"iter": 0, "file": None, "source": "base_model"}

    existing = latest_checkpoint(adapter_dir)
    if existing is None:
        return {"iter": 0, "file": None, "source": "base_model"}

    checkpoint_iter, checkpoint_path = existing
    if checkpoint_iter >= target_iters:
        return {
            "iter": checkpoint_iter,
            "file": str(checkpoint_path),
            "source": "existing_checkpoint",
            "restored": False,
        }

    backup_path = workspace / "resume" / expert / f"initial_{checkpoint_iter:07d}.safetensors"
    copy_checkpoint(checkpoint_path, backup_path)
    return {
        "iter": checkpoint_iter,
        "file": str(backup_path),
        "source": "existing_checkpoint",
        "restored": False,
    }


def run_training_attempt(
    *,
    expert: str,
    attempt_index: int,
    seed: int,
    adapter_dir: Path,
    trainer_cmd: list[str],
    temp_config: Path,
    workspace: Path,
    resume_iter: int,
    resume_file: str | None,
    remaining_iters: int,
) -> dict[str, Any]:
    log_path = workspace / "logs" / expert / f"attempt_{attempt_index:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = trainer_cmd + ["--config", str(temp_config)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    summary: dict[str, Any] = {
        "attempt": attempt_index,
        "seed": seed,
        "resume_from_iter": resume_iter,
        "resume_from_file": resume_file,
        "remaining_target_iters": remaining_iters,
        "status": "running",
        "nan_detected": False,
        "exit_code": None,
        "started_at": now_iso(),
        "last_finite_train_loss": None,
        "last_finite_train_iter": None,
        "last_logged_val_loss": None,
        "best_logged_val_loss": None,
        "recovery_checkpoint_iter": resume_iter if resume_iter > 0 else None,
        "recovery_checkpoint_file": resume_file,
        "log_path": str(log_path),
    }

    last_train_loss: float | None = None
    last_train_global_iter: int | None = None
    terminate_requested = False

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        summary["pid"] = proc.pid

        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                log_file.write(raw_line)
                log_file.flush()
                sys.stdout.write(f"[{expert} a{attempt_index} s{seed}] {raw_line}")
                sys.stdout.flush()

                train_match = TRAIN_LOSS_RE.search(raw_line)
                if train_match:
                    local_iter = int(train_match.group(1))
                    global_iter = resume_iter + local_iter
                    last_train_global_iter = global_iter
                    last_train_loss = parse_loss(train_match.group(2))
                    if is_finite_loss(last_train_loss):
                        summary["last_finite_train_loss"] = last_train_loss
                        summary["last_finite_train_iter"] = global_iter
                    else:
                        summary["nan_detected"] = True
                        summary["nan_detected_at_iter"] = global_iter
                        summary["status"] = "nan_detected"
                        if not terminate_requested:
                            terminate_requested = True
                            terminate_process(proc)
                    continue

                val_match = VAL_LOSS_RE.search(raw_line)
                if val_match:
                    local_iter = int(val_match.group(1))
                    global_iter = resume_iter + local_iter
                    val_loss = parse_loss(val_match.group(2))
                    if is_finite_loss(val_loss):
                        summary["last_logged_val_loss"] = val_loss
                        summary["last_logged_val_iter"] = global_iter
                        best_logged = summary["best_logged_val_loss"]
                        if best_logged is None or val_loss < best_logged:
                            summary["best_logged_val_loss"] = val_loss
                            summary["best_logged_val_iter"] = global_iter
                    else:
                        summary["nan_detected"] = True
                        summary["nan_detected_at_iter"] = global_iter
                        summary["status"] = "nan_detected"
                        if not terminate_requested:
                            terminate_requested = True
                            terminate_process(proc)
                    continue

                save_match = SAVE_RE.search(raw_line)
                if save_match:
                    local_iter = int(save_match.group(1))
                    global_iter = resume_iter + local_iter
                    checkpoint_file = adapter_dir / f"{local_iter:07d}_adapters.safetensors"
                    if last_train_global_iter != global_iter or not is_finite_loss(last_train_loss):
                        summary["nan_detected"] = True
                        summary["nan_detected_at_iter"] = global_iter
                        summary["status"] = "nan_detected"
                        if not terminate_requested:
                            terminate_requested = True
                            terminate_process(proc)
                        continue

                    backup_file = (
                        workspace
                        / "resume"
                        / expert
                        / f"global_{global_iter:07d}_seed_{seed:04d}_attempt_{attempt_index:02d}.safetensors"
                    )
                    copy_checkpoint(checkpoint_file, backup_file)
                    summary["recovery_checkpoint_iter"] = global_iter
                    summary["recovery_checkpoint_file"] = str(backup_file)
                    continue
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

        summary["exit_code"] = proc.wait()

    if summary["nan_detected"]:
        backup_file = summary.get("recovery_checkpoint_file")
        checkpoint_iter = summary.get("recovery_checkpoint_iter")
        if backup_file and checkpoint_iter:
            restore_checkpoint(Path(backup_file), adapter_dir, int(checkpoint_iter))
        summary["finished_at"] = now_iso()
        if summary["status"] == "running":
            summary["status"] = "nan_detected"
        return summary

    summary["finished_at"] = now_iso()
    if summary["exit_code"] != 0:
        summary["status"] = "failed"
        return summary

    adapter_file = adapter_dir / "adapters.safetensors"
    if not adapter_file.exists():
        summary["status"] = "failed"
        summary["failure_reason"] = f"missing final adapter file: {adapter_file}"
        return summary

    summary["status"] = "completed"
    summary["final_adapter_file"] = str(adapter_file)
    summary["effective_total_iters"] = resume_iter + remaining_iters
    return summary


def train_one_expert(
    *,
    expert: str,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    receipt: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    target_iters = int(base_config["iters"])
    base_seed = int(args.seed if args.seed is not None else base_config["seed"])
    adapter_dir = Path(args.adapter_dir) / expert
    expert_receipt: dict[str, Any] = {
        "status": "running",
        "adapter_path": str(adapter_dir),
        "target_iters": target_iters,
        "threshold_val_loss": args.threshold,
        "attempts": [],
        "started_at": now_iso(),
    }

    state = initial_resume_state(
        expert,
        adapter_dir=adapter_dir,
        target_iters=target_iters,
        resume_existing=args.resume_existing,
        workspace=workspace,
    )
    current_resume_iter = int(state["iter"])
    current_resume_file = state["file"]
    current_seed = base_seed

    if current_resume_iter >= target_iters:
        expert_receipt["status"] = "completed"
        expert_receipt["seed_used"] = current_seed
        expert_receipt["best_checkpoint_iter"] = current_resume_iter
        expert_receipt["finished_at"] = now_iso()
        return expert_receipt

    trainer_cmd = args.trainer_cmd or [sys.executable, "-m", "mlx_lm.lora"]

    for attempt_index in range(1, args.max_seed_attempts + 1):
        receipt["current_expert"] = expert
        receipt["current_attempt"] = attempt_index
        write_json(args.receipt_path, receipt)

        remaining_iters = target_iters - current_resume_iter

        # Build per-attempt config with adapter and data paths resolved
        attempt_config = dict(base_config)
        attempt_config["data"] = str(Path(args.data_dir) / expert)
        attempt_config["adapter_path"] = str(adapter_dir)
        temp_config = workspace / "configs" / expert / f"attempt_{attempt_index:02d}.yaml"
        build_attempt_config(
            attempt_config,
            temp_config,
            seed=current_seed,
            resume_adapter_file=current_resume_file,
            remaining_iters=remaining_iters,
            save_every=args.save_every,
        )

        attempt = run_training_attempt(
            expert=expert,
            attempt_index=attempt_index,
            seed=current_seed,
            adapter_dir=adapter_dir,
            trainer_cmd=trainer_cmd,
            temp_config=temp_config,
            workspace=workspace,
            resume_iter=current_resume_iter,
            resume_file=current_resume_file,
            remaining_iters=remaining_iters,
        )
        expert_receipt["attempts"].append(attempt)
        receipt["experts"][expert] = expert_receipt
        write_json(args.receipt_path, receipt)

        if attempt["status"] == "completed":
            expert_receipt["seed_used"] = current_seed
            expert_receipt["best_checkpoint_iter"] = target_iters
            expert_receipt["resume_checkpoint_iter"] = attempt["resume_from_iter"]
            expert_receipt["status"] = "completed"
            expert_receipt["finished_at"] = now_iso()
            return expert_receipt

        if attempt["status"] == "nan_detected":
            checkpoint_iter = attempt.get("recovery_checkpoint_iter")
            checkpoint_file = attempt.get("recovery_checkpoint_file")
            if checkpoint_iter is None or checkpoint_file is None:
                expert_receipt["status"] = "failed"
                expert_receipt["failure_reason"] = "NaN detected before any recoverable checkpoint was captured"
                expert_receipt["finished_at"] = now_iso()
                return expert_receipt
            current_resume_iter = int(checkpoint_iter)
            current_resume_file = checkpoint_file
            current_seed += 1
            continue

        expert_receipt["status"] = "failed"
        expert_receipt["failure_reason"] = attempt.get("failure_reason", "training subprocess failed")
        expert_receipt["finished_at"] = now_iso()
        return expert_receipt

    expert_receipt["status"] = "failed"
    expert_receipt["failure_reason"] = f"exhausted {args.max_seed_attempts} seed attempts"
    expert_receipt["best_checkpoint_iter"] = current_resume_iter if current_resume_iter > 0 else None
    expert_receipt["seed_used"] = current_seed - 1
    expert_receipt["finished_at"] = now_iso()
    return expert_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NaN-safe wrapper for MLX LoRA training with automatic checkpoint recovery"
    )
    parser.add_argument("--config", type=Path, required=True, help="MLX LoRA YAML config file")
    parser.add_argument("--experts", nargs="+", required=True, help="Expert names to train")
    parser.add_argument("--data-dir", required=True, help="Base data directory (expects <data-dir>/<expert>/ subdirs)")
    parser.add_argument("--adapter-dir", default="./adapters", help="Base adapter output directory (default: ./adapters)")
    parser.add_argument("--save-every", type=int, default=30, help="Checkpoint save interval in iterations")
    parser.add_argument("--max-seed-attempts", type=int, default=5, help="Max retry attempts per expert before giving up")
    parser.add_argument("--threshold", type=float, default=3.0, help="Validation loss threshold for success")
    parser.add_argument("--seed", type=int, default=None, help="Override base seed from config")
    parser.add_argument("--receipt-path", type=Path, default=DEFAULT_RECEIPT, help="Path to write the JSON receipt")
    parser.add_argument(
        "--trainer-cmd", nargs="+", default=None,
        help="Training command (default: python -m mlx_lm.lora). The wrapper appends --config <path>."
    )
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest existing numbered checkpoint when present.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.config.exists():
        raise SystemExit(f"missing config: {args.config}")
    if args.max_seed_attempts < 1:
        raise SystemExit("--max-seed-attempts must be at least 1")
    if args.save_every < 1:
        raise SystemExit("--save-every must be at least 1")

    base_config = load_yaml(args.config)
    receipt: dict[str, Any] = {
        "status": "running",
        "started_at": now_iso(),
        "config_path": str(args.config),
        "adapter_dir": args.adapter_dir,
        "data_dir": args.data_dir,
        "save_every": args.save_every,
        "max_seed_attempts": args.max_seed_attempts,
        "threshold": args.threshold,
        "experts_requested": args.experts,
        "experts": {},
    }
    write_json(args.receipt_path, receipt)

    try:
        with tempfile.TemporaryDirectory(prefix="nan_safe_training_") as tmpdir:
            workspace = Path(tmpdir)
            overall_ok = True
            for expert in args.experts:
                result = train_one_expert(
                    expert=expert,
                    args=args,
                    base_config=base_config,
                    receipt=receipt,
                    workspace=workspace,
                )
                receipt["experts"][expert] = result
                write_json(args.receipt_path, receipt)
                expert_ok = result["status"] == "completed"
                overall_ok = overall_ok and expert_ok

        receipt.pop("current_expert", None)
        receipt.pop("current_attempt", None)
        receipt["status"] = "completed" if overall_ok else "failed"
        receipt["finished_at"] = now_iso()
        write_json(args.receipt_path, receipt)
        return 0 if overall_ok else 1
    except KeyboardInterrupt:
        receipt["status"] = "interrupted"
        receipt["finished_at"] = now_iso()
        write_json(args.receipt_path, receipt)
        raise
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["finished_at"] = now_iso()
        receipt["error"] = str(exc)
        write_json(args.receipt_path, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
