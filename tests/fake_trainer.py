#!/usr/bin/env python3
"""Fake MLX trainer that prints loss lines and simulates NaN.

Used by integration tests. Reads config YAML to determine behavior:
- Prints train loss and val loss in MLX format
- Writes numbered checkpoint files to adapter_path
- Injects NaN at a configurable iteration (via NAN_AT_ITER env var)
- On second run (resume detected), completes without NaN
"""

import os
import sys
from pathlib import Path

import yaml


def main():
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    if not config_path:
        print("ERROR: --config required", file=sys.stderr)
        sys.exit(1)

    config = yaml.safe_load(Path(config_path).read_text())
    iters = int(config.get("iters", 100))
    save_every = int(config.get("save_every", 10))
    seed = int(config.get("seed", 42))
    adapter_path = config.get("adapter_path", "./adapters/test")
    resume = config.get("resume_adapter_file")

    nan_at_iter = int(os.environ.get("NAN_AT_ITER", "0"))

    adapter_dir = Path(adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    base_loss = 3.5
    for i in range(1, iters + 1):
        loss = base_loss - (i * 0.01) + (seed % 10) * 0.001

        if nan_at_iter > 0 and i == nan_at_iter and resume is None:
            # Only NaN on first attempt (no resume), succeed on retry
            print(f"Iter {i}: Train loss nan, Learning Rate 1.000e-05")
            sys.stdout.flush()
            continue

        print(f"Iter {i}: Train loss {loss:.4f}, Learning Rate 1.000e-05")
        sys.stdout.flush()

        if i % save_every == 0:
            checkpoint = adapter_dir / f"{i:07d}_adapters.safetensors"
            checkpoint.write_bytes(f"checkpoint_seed{seed}_iter{i}".encode())
            print(f"Iter {i}: Saved adapter weights to {checkpoint}")
            sys.stdout.flush()

        if i % (save_every * 2) == 0:
            val_loss = loss * 0.5
            print(f"Iter {i}: Val loss {val_loss:.4f}, Val Tokens-per-sec 123.4")
            sys.stdout.flush()

    # Write final adapter
    final = adapter_dir / "adapters.safetensors"
    final.write_bytes(f"final_seed{seed}".encode())


if __name__ == "__main__":
    main()
