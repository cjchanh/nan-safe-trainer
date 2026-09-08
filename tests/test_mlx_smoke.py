"""Real-MLX smoke: one training step. Skips when mlx is not installed."""

from __future__ import annotations

import math
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")


class _TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = mx.array([[0.5, 0.0], [0.0, 0.5]])
        self.b = mx.zeros((2,))

    def __call__(self, x: Any) -> Any:
        return x @ self.w + self.b


def test_mlx_one_training_step() -> None:
    model = _TinyLinear()
    mx.eval(model.parameters())
    w_before = mx.array(model.w)

    def loss_fn(model: nn.Module, x: Any, y: Any) -> Any:
        return mx.mean((model(x) - y) ** 2)

    x = mx.ones((4, 2))
    y = mx.zeros((4, 2))
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optim.SGD(learning_rate=0.1).update(model, grads)
    mx.eval(model.parameters(), loss)

    assert math.isfinite(float(loss))
    assert not bool(mx.array_equal(model.w, w_before))
