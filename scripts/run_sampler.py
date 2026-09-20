"""Verify the full-resolution bilinear sampler and write round-trip images.

Sampling a texture at its own texel centers reproduces it exactly, which
confirms the texel-center convention every compressor relies on.

Usage:  .venv/bin/python scripts/run_sampler.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import FullResSampler, get_device, load_image, psnr, save_image, texel_centers
from ntc.viz import hstack  # noqa: E402


def main() -> int:
    device = get_device()
    out_dir = ROOT / "experiments" / "sampler"
    print(f"device: {device}")

    textures = sorted((ROOT / "assets" / "provided").glob("*.png"))
    for path in textures:
        texture = load_image(path).to(device)
        height, width = texture.shape[:2]

        sampler = FullResSampler().compress(texture)
        uv = texel_centers(height, width, device=device)
        recon = sampler.sample(uv).reshape(height, width, 3)

        quality = psnr(texture, recon)
        save_image(hstack([texture, recon]), out_dir / f"{path.stem}_roundtrip.png")
        print(
            f"{path.name:14s} {width}x{height}  "
            f"round-trip PSNR={quality} dB  "
            f"({sampler.size_bytes / 1024:.1f} KB raw)"
        )
        if quality != float("inf"):
            print(f"  ERROR: round-trip is not exact (PSNR {quality})")
            return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
