"""Tests for the model + training loop. No pytest required.

Usage:  .venv/bin/python tests/test_train.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc.metrics import psnr  # noqa: E402
from ntc.model import MODEL_SIZES, NeuralTexture  # noqa: E402
from ntc.train import build_dataset, decode, train  # noqa: E402


def smooth_texture(height=32, width=32):
    u = torch.linspace(0, 1, width).reshape(1, width, 1)
    v = torch.linspace(0, 1, height).reshape(height, 1, 1)
    return torch.cat([u.expand(height, width, 1),
                      v.expand(height, width, 1),
                      0.5 * (u + v).expand(height, width, 1)], dim=2).contiguous()


def test_dataset_matches_texture():
    texture = smooth_texture(16, 24)
    coords, targets = build_dataset(texture)
    assert coords.shape == (16 * 24, 2)
    assert targets.shape == (16 * 24, 3)
    assert coords.min() >= 0.0 and coords.max() <= 1.0
    assert torch.equal(targets, texture.reshape(-1, 3))
    print("ok: dataset is every texel center paired with its color")


def test_model_forward_shape():
    model = NeuralTexture(resolutions=(8,), feat_dim=2)
    out = model(torch.rand(100, 2))
    assert out.shape == (100, 3)
    assert out.min() >= 0 and out.max() <= 1
    print("ok: model maps (N, 2) to (N, 3) colors in [0, 1]")


def test_named_sizes_build():
    for size, cfg in MODEL_SIZES.items():
        model = NeuralTexture.from_size(size)
        assert model.grid.resolutions == cfg["resolutions"]
        assert model.mlp.in_dim == model.grid.out_dim
    print("ok: the named sizes build with matching grid/MLP widths")


def test_training_reduces_loss():
    torch.manual_seed(0)
    texture = smooth_texture(32, 32)
    coords, targets = build_dataset(texture)
    model = NeuralTexture(resolutions=(8, 16), feat_dim=2)
    history = train(model, coords, targets, steps=400, batch=1024, lr=1e-2, log_every=100)

    assert history[0]["psnr"] < 25.0
    assert history[-1]["psnr"] > history[0]["psnr"] + 10.0
    full = decode(model, 32, 32)
    assert psnr(texture, full) > 30.0
    print(f"ok: training lifts PSNR {history[0]['psnr']:.1f} -> {history[-1]['psnr']:.1f} "
          f"(full image {psnr(texture, full):.1f} dB)")


def test_decode_shape():
    model = NeuralTexture(resolutions=(8,), feat_dim=2)
    assert decode(model, 16, 20).shape == (16, 20, 3)
    print("ok: decode renders the full image")


def main() -> int:
    tests = [
        test_dataset_matches_texture,
        test_model_forward_shape,
        test_named_sizes_build,
        test_training_reduces_loss,
        test_decode_shape,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
