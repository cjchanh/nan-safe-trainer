# AGENTS.md - nan-safe-trainer

## Scope

This repository contains the `nan-safe-trainer` Python package for NaN/Inf detection and checkpoint recovery around MLX LoRA training.

## Guardrails

- Keep changes limited to this repository unless the operator gives an explicit broader scope.
- Do not commit or inspect private training data, adapters, model weights, `.env` files, PEM files, or generated receipts.
- Treat `data/`, `adapters/`, `evidence/`, release artifacts, and generated checkpoint files as local/generated surfaces.
- Do not weaken NaN/Inf fail-closed behavior or checkpoint recovery semantics without tests covering the change.

## Verification

Run the canonical gate before closeout:

```bash
python3 -m pytest -q
```

Baseline: 28 tests passing.
