"""Evaluate the seam-aware S3TC variants against the baseline.

Runs the baseline encoder and the diffusion / boundary / ICM variants on the
provided textures, measures PSNR and the cross-boundary seam error, writes
side-by-side and zoomed comparisons under experiments/s3tc_opt/, and a summary
plot used by the writeup at writeup/assets/s3tc_opt_seam.png.

Usage:  .venv/bin/python scripts/run_s3tc_optimizations.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import (  # noqa: E402
    S3TCOpt,
    compression_factor,
    get_device,
    load_image,
    psnr,
    raw_bytes,
    save_image,
    seam_error,
)
from ntc.viz import hstack  # noqa: E402

CONFIGS = ["baseline", "diffusion", "boundary", "icm"]
OUT = ROOT / "experiments" / "s3tc_opt"
ASSETS = ROOT / "writeup" / "assets"
PLOT = ASSETS / "s3tc_opt_seam.png"


def zoom_crop(image: torch.Tensor, size: int = 96, scale: int = 4) -> torch.Tensor:
    height, width = image.shape[:2]
    top, left = (height - size) // 2, (width - size) // 2
    crop = image[top:top + size, left:left + size]
    crop = crop.permute(2, 0, 1).unsqueeze(0)
    crop = F.interpolate(crop, scale_factor=scale, mode="nearest")
    return crop[0].permute(1, 2, 0)


def main() -> int:
    device = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    print(f"{'texture':10s} {'method':10s} {'PSNR (dB)':>10s} {'seam':>10s}")

    results = []
    for path in sorted((ROOT / "assets" / "provided").glob("*.png")):
        texture = load_image(path).to(device)
        height, width = texture.shape[:2]
        row = [texture]
        zooms = [zoom_crop(texture)]
        for method in CONFIGS:
            decoded = S3TCOpt(method=method).compress(texture).decode()
            quality = psnr(texture, decoded)
            seam = seam_error(texture, decoded)
            row.append(decoded)
            zooms.append(zoom_crop(decoded))
            results.append({
                "texture": path.stem,
                "method": method,
                "psnr_db": quality,
                "seam_error": seam,
                "s3tc_bytes": (height * width) // 2,
                "raw_bytes": raw_bytes(height, width),
                "compression_factor": compression_factor(raw_bytes(height, width),
                                                         (height * width) // 2),
            })
            print(f"{path.stem:10s} {method:10s} {quality:10.3f} {seam:10.6f}")

        save_image(hstack(row, gap=6), OUT / f"{path.stem}_methods.png")
        save_image(hstack(zooms, gap=6), ASSETS / f"{path.stem}_methods.png")

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))

    # Summary plot: PSNR against seam error, one point per method per texture.
    markers = {"gradient": "o", "bricks": "s", "clouds": "^"}
    colors = {"baseline": "#444444", "diffusion": "#e08a1e", "boundary": "#2f8f4e",
              "icm": "#2f5d8a"}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for entry in results:
        ax.scatter(entry["seam_error"], entry["psnr_db"],
                   marker=markers[entry["texture"]], color=colors[entry["method"]],
                   s=45, edgecolor="white", linewidth=0.6)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=m)
               for m, c in colors.items()]
    handles += [plt.Line2D([], [], marker=mk, linestyle="", color="#888",
                           label=t) for t, mk in markers.items()]
    ax.legend(handles=handles, ncol=2, fontsize=8, frameon=False)
    ax.set_xlabel("seam error (lower is better)")
    ax.set_ylabel("PSNR dB (higher is better)")
    ax.set_title("S3TC endpoint selection: seam error vs quality")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"\nwrote {OUT}/metrics.json and {PLOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
