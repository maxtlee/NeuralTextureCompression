"""Tests for the multi-resolution feature grid. No pytest required.

Usage:  .venv/bin/python tests/test_feature_grid.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import get_device, sample_bilinear, texel_centers  # noqa: E402
from ntc.feature_grid import FeatureGrid  # noqa: E402


def test_output_shape():
    grid = FeatureGrid(resolutions=(16, 32, 64), feat_dim=2)
    assert grid.out_dim == 6
    out = grid(torch.rand(500, 2))
    assert out.shape == (500, 6)
    print("ok: output shape is (N, levels * feat_dim)")


def test_parameter_count():
    grid = FeatureGrid(resolutions=(16, 32, 64), feat_dim=2)
    expected = 2 * (16 * 16 + 32 * 32 + 64 * 64)
    assert grid.num_parameters == expected, (grid.num_parameters, expected)
    print("ok: parameter count matches R^2 * feat_dim summed over levels")


def test_matches_full_resolution_sampler():
    torch.manual_seed(0)
    resolution = 16
    grid = FeatureGrid(resolutions=(resolution,), feat_dim=3)
    with torch.no_grad():
        grid.grids[0].copy_(torch.randn(1, 3, resolution, resolution) * 0.1)

    uv = torch.rand(2000, 2)
    got = grid(uv)
    reference = sample_bilinear(grid.grids[0][0].permute(1, 2, 0), uv)
    assert torch.allclose(got, reference, atol=1e-6), (got - reference).abs().max()
    print("ok: one 3-channel level matches the full-resolution bilinear sampler")


def test_border_clamping():
    resolution = 8
    grid = FeatureGrid(resolutions=(resolution,), feat_dim=1)
    with torch.no_grad():
        grid.grids[0].copy_(torch.arange(resolution * resolution).float().reshape(1, 1, resolution, resolution))

    corners = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    out = grid(corners).reshape(-1)
    # uv=0 maps to the first texel and uv=1 to the last; outside those clamps.
    texels = grid.grids[0].reshape(-1)
    assert out[1].item() == texels[0].item()
    assert out[2].item() == texels[-1].item()
    assert out[0].item() == texels[0].item()
    assert out[3].item() == texels[-1].item()
    print("ok: coordinates outside [0, 1] clamp to the border")


def test_gradients_flow_to_every_level():
    grid = FeatureGrid(resolutions=(16, 32), feat_dim=2)
    grid(torch.rand(256, 2)).pow(2).mean().backward()
    for level in grid.grids:
        assert level.grad is not None and level.grad.abs().sum() > 0
    print("ok: gradients reach every level")


def test_texel_center_reproduces_a_level():
    # Sampling a level exactly at its own texel centers returns the stored features.
    resolution = 12
    grid = FeatureGrid(resolutions=(resolution,), feat_dim=2)
    with torch.no_grad():
        grid.grids[0].copy_(torch.randn(1, 2, resolution, resolution))
    uv = texel_centers(resolution, resolution)
    out = grid(uv).reshape(resolution, resolution, 2)
    assert torch.allclose(out, grid.grids[0][0].permute(1, 2, 0), atol=1e-6)
    print("ok: level sampled at texel centers equals the stored grid")


def main() -> int:
    device = get_device()
    print(f"device: {device}")
    tests = [
        test_output_shape,
        test_parameter_count,
        test_matches_full_resolution_sampler,
        test_border_clamping,
        test_gradients_flow_to_every_level,
        test_texel_center_reproduces_a_level,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
