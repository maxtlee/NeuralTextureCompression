"""Feature-grid shapes, storage cost, and a consistency check against the sampler.

The three architectures to be trained later differ only in their grid setup:
  small   grids (64)              feat_dim 2
  medium  grids (16, 32, 64)      feat_dim 2
  large   grids (16, 32, 64, 128) feat_dim 4

Usage:  .venv/bin/python scripts/run_feature_grid.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import get_device, sample_bilinear  # noqa: E402
from ntc.feature_grid import FeatureGrid  # noqa: E402

CONFIGS = {
    "small": dict(resolutions=(64,), feat_dim=2),
    "medium": dict(resolutions=(16, 32, 64), feat_dim=2),
    "large": dict(resolutions=(16, 32, 64, 128), feat_dim=4),
}


def main() -> int:
    device = get_device()
    print(f"device: {device}")
    print(f"{'model':8s} {'grids':22s} {'feat_dim':>8s} {'out_dim':>8s} {'params':>9s} {'KB (f32)':>9s}")
    for name, cfg in CONFIGS.items():
        grid = FeatureGrid(**cfg).to(device)
        print(
            f"{name:8s} {str(cfg['resolutions']):22s} {cfg['feat_dim']:8d} "
            f"{grid.out_dim:8d} {grid.num_parameters:9d} "
            f"{grid.num_parameters * 4 / 1024:9.1f}"
        )

    # One level with feat_dim=3 is just a 3-channel image; the grid lookup must
    # match the full-resolution sampler with the same convention.
    torch.manual_seed(0)
    resolution = 16
    grid = FeatureGrid(resolutions=(resolution,), feat_dim=3).to(device)
    with torch.no_grad():
        grid.grids[0].copy_(torch.randn(1, 3, resolution, resolution, device=device) * 0.1)

    uv = torch.rand(1000, 2, device=device)
    got = grid(uv)
    texels = grid.grids[0][0].permute(1, 2, 0)  # (R, R, 3)
    reference = sample_bilinear(texels, uv)
    max_error = (got - reference).abs().max().item()
    print(f"\nsampler consistency: max abs error {max_error:.2e}")
    if max_error > 1e-5:
        print("ERROR: feature-grid lookup does not match the sampler")
        return 1

    # Gradients must reach every level.
    grid = FeatureGrid(resolutions=(16, 32), feat_dim=2).to(device)
    loss = grid(torch.rand(512, 2, device=device)).pow(2).mean()
    loss.backward()
    for i, level in enumerate(grid.grids):
        if level.grad is None or level.grad.abs().sum() == 0:
            print(f"ERROR: no gradient on level {i}")
            return 1
    print("gradients reach every level: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
