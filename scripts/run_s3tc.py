"""Compress textures with S3TC/DXT1 and report PSNR + size.

Writes side-by-side original|reconstruction images and a metrics JSON under
experiments/s3tc/.

Usage:  .venv/bin/python scripts/run_s3tc.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import S3TC, compression_factor, get_device, load_image, psnr, raw_bytes, save_image
from ntc.viz import hstack  # noqa: E402


def main() -> int:
    device = get_device()
    out_dir = ROOT / "experiments" / "s3tc"
    results = []
    print(f"device: {device}")
    print(f"{'texture':10s} {'PSNR (dB)':>10s} {'S3TC (KB)':>10s} {'raw (KB)':>10s} {'ratio':>7s}")

    for path in sorted((ROOT / "assets" / "provided").glob("*.png")):
        texture = load_image(path).to(device)
        height, width = texture.shape[:2]

        s3tc = S3TC().compress(texture)
        recon = s3tc.decode()

        quality = psnr(texture, recon)
        raw = raw_bytes(height, width)
        ratio = compression_factor(raw, s3tc.size_bytes)

        save_image(hstack([texture, recon]), out_dir / f"{path.stem}_s3tc.png")
        results.append(
            {
                "texture": path.stem,
                "width": width,
                "height": height,
                "psnr_db": quality,
                "s3tc_bytes": s3tc.size_bytes,
                "raw_bytes": raw,
                "compression_factor": ratio,
            }
        )
        print(
            f"{path.stem:10s} {quality:10.3f} {s3tc.size_bytes / 1024:10.1f} "
            f"{raw / 1024:10.1f} {ratio:6.2f}x"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir}/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
