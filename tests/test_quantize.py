"""Tests for post-training 8-bit quantization. No pytest required.

Usage:  .venv/bin/python tests/test_quantize.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc.model import NeuralTexture  # noqa: E402
from ntc.quantize import (  # noqa: E402
    float_size_bytes,
    quantize_model,
    quantize_uint8,
    quantized_size_bytes,
)


def test_rounding_error_bounded():
    x = torch.randn(1000) * 3 + 1
    q, lo, scale, x_hat = quantize_uint8(x)
    assert q.dtype == torch.uint8
    assert q.min() >= 0 and q.max() <= 255
    assert x_hat.min() >= lo - 1e-6 and x_hat.max() <= lo + 255 * scale + 1e-6
    assert (x - x_hat).abs().max() <= scale / 2 + 1e-6
    print(f"ok: max error <= half a quantization step ({scale / 2:.2e})")


def test_constant_array():
    x = torch.full((64,), 0.3)
    q, lo, scale, x_hat = quantize_uint8(x)
    assert torch.allclose(x_hat, x)
    print("ok: constant array quantizes exactly")


def test_quantize_model_grid_only():
    torch.manual_seed(0)
    model = NeuralTexture(resolutions=(8, 16), feat_dim=2)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    quantize_model(model, quantize_mlp=False)
    # Grids changed (unless degenerate), MLP untouched.
    assert not torch.equal(model.grid.grids[0], before["grid.grids.0"])
    for name, p in model.mlp.named_parameters():
        assert torch.equal(p, before[f"mlp.{name}"])
    print("ok: grid-only quantization leaves the MLP in float32")


def test_quantize_model_all():
    torch.manual_seed(0)
    model = NeuralTexture(resolutions=(8, 16), feat_dim=2)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    quantize_model(model, quantize_mlp=True)
    for name, p in model.mlp.named_parameters():
        assert not torch.equal(p, before[f"mlp.{name}"])
    print("ok: quantize_mlp also quantizes the decoder")


def test_size_accounting():
    model = NeuralTexture(resolutions=(16, 32, 64), feat_dim=2)
    grid_values = 2 * (16 * 16 + 32 * 32 + 64 * 64)
    mlp_values = sum(p.numel() for p in model.mlp.parameters())
    assert float_size_bytes(model) == (grid_values + mlp_values) * 4
    assert quantized_size_bytes(model, quantize_mlp=False) == grid_values * 1 + 3 * 8 + mlp_values * 4
    assert quantized_size_bytes(model, quantize_mlp=True) == (grid_values + mlp_values) * 1 + (3 + 6) * 8
    print("ok: quantized size is 1 byte per value plus range endpoints")


def test_quality_drop_is_small():
    torch.manual_seed(0)
    # Fit a tiny smooth target, then quantize and compare.
    heights = torch.linspace(0, 1, 16).reshape(16, 1, 1).expand(16, 16, 3)
    from ntc.train import build_dataset, train

    model = NeuralTexture(resolutions=(8, 16), feat_dim=2)
    coords, targets = build_dataset(heights)
    train(model, coords, targets, steps=300, batch=256, lr=1e-2, log_every=0)

    from ntc.train import decode

    def quality():
        pred = decode(model, 16, 16)
        return float(torch.mean((pred - heights) ** 2))

    err_float = quality()
    quantize_model(model, quantize_mlp=False)
    err_quant = quality()
    assert err_quant < err_float * 3.0, (err_float, err_quant)
    print(f"ok: MSE grows only mildly after quantization ({err_float:.2e} -> {err_quant:.2e})")


def main() -> int:
    tests = [
        test_rounding_error_bounded,
        test_constant_array,
        test_quantize_model_grid_only,
        test_quantize_model_all,
        test_size_accounting,
        test_quality_drop_is_small,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
