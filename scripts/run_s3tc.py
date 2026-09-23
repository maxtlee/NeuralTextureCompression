"""Compress textures with S3TC/DXT1 and report PSNR + size.

Writes side-by-side original|reconstruction images and a metrics JSON under
experiments/s3tc/. By default it runs the three provided textures; pass --image
to compress any of your own.

Usage:
  .venv/bin/python scripts/run_s3tc.py
  .venv/bin/python scripts/run_s3tc.py --image path/to/your.png
  .venv/bin/python scripts/run_s3tc.py --image photo.jpg --size-px 256 --name photo
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import (  # noqa: E402
    S3TC,
    compression_factor,
    get_device,
    load_image,
    load_texture,
    psnr,
    raw_bytes,
    save_image,
)
from ntc.viz import hstack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="compress a custom image instead of the provided set")
    parser.add_argument("--name", default=None, help="output stem for --image (default: filename)")
    parser.add_argument("--size-px", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    out_dir = ROOT / "experiments" / "s3tc"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        items = [(args.name or Path(args.image).stem,
                  load_texture(args.image, size=args.size_px).to(device))]
    else:
        items = [(path.stem, load_image(path).to(device))
                 for path in sorted((ROOT / "assets" / "provided").glob("*.png"))]

    results = []
    print(f"device: {device}")
    print(f"{'texture':10s} {'PSNR (dB)':>10s} {'S3TC (KB)':>10s} {'raw (KB)':>10s} {'ratio':>7s}")

    for stem, texture in items:
        height, width = texture.shape[:2]
        s3tc = S3TC().compress(texture)
        recon = s3tc.decode()

        quality = psnr(texture, recon)
        raw = raw_bytes(height, width)
        ratio = compression_factor(raw, s3tc.size_bytes)

        save_image(hstack([texture, recon]), out_dir / f"{stem}_s3tc.png")
        results.append({
            "texture": stem,
            "width": width,
            "height": height,
            "psnr_db": quality,
            "s3tc_bytes": s3tc.size_bytes,
            "raw_bytes": raw,
            "compression_factor": ratio,
        })
        print(f"{stem:10s} {quality:10.3f} {s3tc.size_bytes / 1024:10.1f} "
              f"{raw / 1024:10.1f} {ratio:6.2f}x")

    prefix = f"{items[0][0]}_" if args.image else ""
    (out_dir / f"{prefix}metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir}/{prefix}metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
