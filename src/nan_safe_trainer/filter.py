#!/usr/bin/env python3
"""Filter out training rows that produce NaN loss on a specific model.

Loads the model once, then evaluates each row individually.
Writes a clean JSONL with only non-NaN rows.

Requires: mlx, mlx-lm
"""

import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def check_row_loss(model, tokenizer, row: dict, max_seq_length: int = 2048) -> float:
    """Compute loss for a single row. Returns loss value or float('nan')."""
    messages = row.get("messages", [])
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = "\n".join(m.get("content", "") for m in messages)

    tokens = tokenizer.encode(text)
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]
    if len(tokens) < 2:
        return float("nan")

    x = mx.array(tokens[:-1])[None, :]
    y = mx.array(tokens[1:])[None, :]

    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y).mean()
    mx.eval(loss)
    return loss.item()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter NaN-producing rows from training data")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output clean JSONL file")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    model, tokenizer = load(args.model)

    input_path = Path(args.input)
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} rows from {input_path}")

    clean_rows = []
    nan_indices = []

    for i, row in enumerate(rows):
        loss = check_row_loss(model, tokenizer, row, args.max_seq_length)
        if math.isnan(loss) or math.isinf(loss):
            nan_indices.append(i)
            if len(nan_indices) <= 20:
                msgs = row.get("messages", [])
                preview = msgs[0].get("content", "")[:80] if msgs else "?"
                print(f"  NaN at row {i}: {preview}...")
        else:
            clean_rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  Checked {i+1}/{len(rows)}, found {len(nan_indices)} NaN rows so far")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r) for r in clean_rows) + "\n")

    print(f"\nDone: {len(rows)} total, {len(nan_indices)} NaN, {len(clean_rows)} clean")
    print(f"Clean data written to {output_path}")
    if nan_indices:
        print(f"NaN row indices: {nan_indices[:50]}")

    return len(nan_indices)


if __name__ == "__main__":
    sys.exit(main())
