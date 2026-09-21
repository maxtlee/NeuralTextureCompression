"""Tests for the decoder MLP. No pytest required.

Usage:  .venv/bin/python tests/test_color_mlp.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc.mlp import HIDDEN, ColorMLP  # noqa: E402


def test_output_shape_and_range():
    for in_dim in (2, 6, 16):
        mlp = ColorMLP(in_dim)
        out = mlp(torch.randn(500, in_dim))
        assert out.shape == (500, 3)
        assert out.min() >= 0.0 and out.max() <= 1.0
    print("ok: output is (N, 3) and stays in [0, 1]")


def test_architecture_is_fixed():
    mlp = ColorMLP(6)
    layers = [m for m in mlp.net]
    assert isinstance(layers[0], nn.Linear) and layers[0].out_features == HIDDEN
    assert isinstance(layers[1], nn.ReLU)
    assert isinstance(layers[2], nn.Linear) and layers[2].in_features == HIDDEN
    assert layers[2].out_features == HIDDEN
    assert isinstance(layers[3], nn.ReLU)
    assert isinstance(layers[4], nn.Linear) and layers[4].out_features == 3
    assert isinstance(layers[5], nn.Sigmoid)
    print("ok: two 64-wide ReLU hidden layers and a sigmoid RGB head")


def test_parameter_count():
    in_dim = 6
    mlp = ColorMLP(in_dim)
    expected = (in_dim * HIDDEN + HIDDEN) + (HIDDEN * HIDDEN + HIDDEN) + (HIDDEN * 3 + 3)
    assert sum(p.numel() for p in mlp.parameters()) == expected
    print(f"ok: parameter count is {expected} for in_dim={in_dim}")


def test_gradients_flow():
    mlp = ColorMLP(4)
    out = mlp(torch.rand(256, 4))
    out.pow(2).mean().backward()
    for name, param in mlp.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, name
    print("ok: gradients reach every weight")


def test_deterministic():
    torch.manual_seed(0)
    a = ColorMLP(3)
    torch.manual_seed(0)
    b = ColorMLP(3)
    x = torch.rand(64, 3)
    assert torch.equal(a(x), b(x))
    print("ok: same seed gives the same output")


def main() -> int:
    tests = [
        test_output_shape_and_range,
        test_architecture_is_fixed,
        test_parameter_count,
        test_gradients_flow,
        test_deterministic,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
