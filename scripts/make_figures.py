"""Generate the committed figures used by the writeup.

Writes full side-by-side original|S3TC comparisons and 4x nearest-neighbor
zoom crops (to make 4x4 block artifacts visible) under writeup/assets/.

Usage:  .venv/bin/python scripts/make_figures.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import S3TC, get_device, load_image, save_image
from ntc.viz import hstack  # noqa: E402

ASSETS = ROOT / "writeup" / "assets"
ZOOM = 96      # crop size in texels
SCALE = 4      # nearest-neighbor upscale factor


def zoom_crop(image: torch.Tensor, size: int = ZOOM, scale: int = SCALE) -> torch.Tensor:
    """Center crop, upscaled with nearest-neighbor so texel/blocks are visible."""
    height, width = image.shape[:2]
    top = (height - size) // 2
    left = (width - size) // 2
    crop = image[top:top + size, left:left + size]
    crop = crop.permute(2, 0, 1).unsqueeze(0)
    crop = F.interpolate(crop, scale_factor=scale, mode="nearest")
    return crop[0].permute(1, 2, 0)


def main() -> int:
    device = get_device()
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    for path in sorted((ROOT / "assets" / "provided").glob("*.png")):
        texture = load_image(path).to(device)
        recon = S3TC().compress(texture).decode()

        save_image(hstack([texture, recon]), ASSETS / f"{path.stem}_s3tc.png")
        save_image(
            hstack([zoom_crop(texture), zoom_crop(recon)]),
            ASSETS / f"{path.stem}_zoom.png",
        )
        print(f"wrote {path.stem}_s3tc.png, {path.stem}_zoom.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
