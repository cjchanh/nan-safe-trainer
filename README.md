[![CI](https://github.com/cjchanh/nan-safe-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/cjchanh/nan-safe-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# nan-safe-trainer

Automatic NaN/Inf detection and checkpoint recovery for LoRA fine-tuning on Apple Silicon.

## The Problem

When fine-tuning language models with LoRA on Apple Silicon using MLX, training runs
can diverge — the loss becomes NaN or Inf, corrupting model weights from that point
forward. This is especially common with 4-bit quantized models under gradient
accumulation.

The standard workflow: watch the terminal, ctrl-C when you see NaN, find the last good
checkpoint, change the random seed, restart manually. At 3am on a long training run,
nobody is watching.

Related discussions in the MLX ecosystem:
- [mlx-examples #620](https://github.com/ml-explore/mlx-examples/issues/620) — NaN training/validation loss
- [mlx-lm #361](https://github.com/ml-explore/mlx-lm/issues/361) — NaN values during fine-tuning
- [mlx-lm discussion #636](https://github.com/ml-explore/mlx-lm/discussions/636) — Debugging NaN

## How It Works

```
spawn training process
    │
    ▼
monitor stdout line-by-line (regex on loss values)
    │
    ├── loss is finite ──► continue monitoring
    │
    └── loss is NaN/Inf ──► kill process immediately
                                │
                                ▼
                          find last good checkpoint
                                │
                                ▼
                          increment random seed
                                │
                                ▼
                          restart from checkpoint
                                │
                                ▼
                          (repeat up to --max-seed-attempts)
```

The wrapper monitors `stdout` for MLX's standard loss output format:

```
Iter 10: Train loss 2.345, Learning Rate 1.000e-05
Iter 30: Val loss 1.876, Val Tokens-per-sec 123.4
```

When NaN or Inf appears in the loss value, the wrapper:

1. Sends SIGTERM to the training subprocess
2. Locates the most recent checkpoint saved before the NaN event
3. Backs up that checkpoint to a workspace directory
4. Writes a temporary config with `seed += 1` and `resume_adapter_file` pointing to the backup
5. Restarts training from the checkpoint with the new seed
6. Repeats until training completes or retries are exhausted

## Installation

```bash
git clone https://github.com/cjchanh/nan-safe-trainer.git
cd nan-safe-trainer
pip install -e .
```

You also need MLX for the actual training:

```bash
pip install mlx mlx-lm
```

## Usage

```bash
nan-safe-train \
  --config examples/example_config.yaml \
  --experts my_expert_a my_expert_b \
  --data-dir ./data \
  --adapter-dir ./adapters \
  --max-seed-attempts 5 \
  --save-every 30
```

Or run as a module:

```bash
python -m nan_safe_trainer --config config.yaml --experts my_expert --data-dir ./data
```

### Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--config` | yes | — | MLX LoRA YAML config file |
| `--experts` | yes | — | Expert names to train (space-separated) |
| `--data-dir` | yes | — | Base data directory (expects `<data-dir>/<expert>/` subdirs with train.jsonl/valid.jsonl) |
| `--adapter-dir` | no | `./adapters` | Base adapter output directory |
| `--save-every` | no | `30` | Checkpoint interval (iterations) |
| `--max-seed-attempts` | no | `5` | Max retries per expert before giving up |
| `--threshold` | no | disabled | Validation loss threshold for success (fail-closed: fails if no val loss logged) |
| `--seed` | no | from config | Override the base seed |
| `--receipt-path` | no | `./nan_safe_receipt.json` | Path for the JSON receipt |
| `--trainer-cmd` | no | `python -m mlx_lm.lora` | Training command (wrapper appends `--config <path>`) |
| `--resume-existing` | no | `true` | Resume from existing checkpoints |

### Data Layout

```
data/
  expert_a/
    train.jsonl
    valid.jsonl
  expert_b/
    train.jsonl
    valid.jsonl
```

### Custom Trainer Command

By default, the wrapper calls `python -m mlx_lm.lora --config <temp_config>`. If you
use a custom training script, pass it via `--trainer-cmd`:

```bash
nan-safe-train \
  --config config.yaml \
  --experts my_expert \
  --data-dir ./data \
  --trainer-cmd python my_custom_trainer.py
```

Your trainer must accept `--config <path>` and print loss in the standard MLX format.

## Receipt Format

Every run writes a structured JSON receipt documenting:

```json
{
  "status": "completed|failed|interrupted",
  "started_at": "2026-01-01T00:00:00+00:00",
  "finished_at": "2026-01-01T01:30:00+00:00",
  "config_path": "config.yaml",
  "experts_requested": ["expert_a", "expert_b"],
  "experts": {
    "expert_a": {
      "status": "completed",
      "target_iters": 600,
      "seed_used": 139,
      "best_checkpoint_iter": 600,
      "attempts": [
        {
          "attempt": 1,
          "seed": 137,
          "nan_detected": true,
          "nan_detected_at_iter": 80,
          "recovery_checkpoint_iter": 60
        },
        {
          "attempt": 2,
          "seed": 138,
          "nan_detected": true,
          "nan_detected_at_iter": 95,
          "recovery_checkpoint_iter": 90
        },
        {
          "attempt": 3,
          "seed": 139,
          "nan_detected": false,
          "status": "completed"
        }
      ]
    }
  }
}
```

## Extras

The package also includes optional utilities (not part of the core training loop):

- `nan_safe_trainer.filter` — Pre-filter training data rows that produce NaN on forward pass. Requires `mlx` and `mlx-lm`.
- `nan_safe_trainer.metal` — Metal GPU memory governance for MLX (set memory/cache limits by profile).
- `nan_safe_trainer.receipt` — Typed dataclasses (`TrainingReceipt`, `ExpertResult`, `AttemptRecord`) for deserializing receipt files.

## Development

```bash
make install   # pip install -e ".[dev]"
make test      # install + pytest (28 tests)
make build     # install + build wheel + sdist
make check     # build + twine check
make clean     # remove build artifacts
```

All targets bootstrap their own dependencies. `make test` from a fresh venv just works.

## What This Does NOT Do

- Does not fix the root cause of NaN (that's a kernel/precision issue in MLX)
- Does not include trained model weights
- Does not include training data
- Does not require cloud services — runs entirely locally

## License

MIT
