"""Decoder sizes and the combined grid + MLP storage for the three architectures.

  small   grids (64)              feat_dim 2
  medium  grids (16, 32, 64)      feat_dim 2
  large   grids (16, 32, 64, 128) feat_dim 4

Usage:  .venv/bin/python scripts/run_color_mlp.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import FeatureGrid, get_device, raw_bytes  # noqa: E402
from ntc.mlp import ColorMLP  # noqa: E402

RAW_TEXTURE = raw_bytes(512, 512)

CONFIGS = {
    "small": dict(resolutions=(64,), feat_dim=2),
    "medium": dict(resolutions=(16, 32, 64), feat_dim=2),
    "large": dict(resolutions=(16, 32, 64, 128), feat_dim=4),
}


def main() -> int:
    device = get_device()
    print(f"device: {device}")
    print(f"{'model':7s} {'out_dim':>7s} {'grid':>8s} {'mlp':>6s} {'total':>8s} {'KB (f32)':>9s} {'vs raw':>7s}")
    for name, cfg in CONFIGS.items():
        grid = FeatureGrid(**cfg)
        mlp = ColorMLP(grid.out_dim)
        grid_params = grid.num_parameters
        mlp_params = sum(p.numel() for p in mlp.parameters())
        total = grid_params + mlp_params
        print(
            f"{name:7s} {grid.out_dim:7d} {grid_params:8d} {mlp_params:6d} {total:8d} "
            f"{total * 4 / 1024:9.1f} {RAW_TEXTURE / (total * 4):6.1f}x"
        )

    # End-to-end shape check: grid features decoded to RGB.
    grid = FeatureGrid(resolutions=(16, 32, 64), feat_dim=4).to(device)
    mlp = ColorMLP(grid.out_dim).to(device)
    colors = mlp(grid(torch.rand(1000, 2, device=device)))
    assert colors.shape == (1000, 3)
    assert colors.min() >= 0.0 and colors.max() <= 1.0
    print(f"\nforward: (1000, 2) -> features {grid.out_dim} -> colors {tuple(colors.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
