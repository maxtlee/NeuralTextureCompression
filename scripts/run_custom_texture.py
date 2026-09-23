"""Fit the representation to any image and save the comparison images.

Given any RGB image, this center-crops it to a square, resizes it, runs the S3TC
baseline, and fits the neural sizes, then writes a comparison image (original vs
each representation), a zoomed crop, the PSNR-over-training curves, and a metrics
JSON into the output directory.

Usage:
  .venv/bin/python scripts/run_custom_texture.py --image path/to/your.png
  .venv/bin/python scripts/run_custom_texture.py --image photo.jpg --sizes small large
  .venv/bin/python scripts/run_custom_texture.py --image t.png --size-px 256 --steps 1000
  .venv/bin/python scripts/run_custom_texture.py --image t.png --no-s3tc --out experiments/custom
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import (  # noqa: E402
    S3TC,
    get_device,
    load_texture,
    psnr,
    raw_bytes,
    save_image,
)
from ntc.codec import NeuralTextureCodec  # noqa: E402
from ntc.quantize import quantized_size_bytes  # noqa: E402
from ntc.viz import hstack, zoom_crop  # noqa: E402

SIZE_STYLE = {"small": "-", "medium": "--", "large": ":"}
SIZE_COLOR = {"small": "#2f8f4e", "medium": "#e08a1e", "large": "#2f5d8a"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="path to any RGB image")
    parser.add_argument("--name", default=None, help="output stem (default: image filename)")
    parser.add_argument("--size-px", type=int, default=512,
                        help="size the (square) texture is resized to (default 512)")
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large"],
                        choices=["small", "medium", "large"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--no-s3tc", action="store_true", help="skip the S3TC baseline")
    parser.add_argument("--out", default=str(ROOT / "experiments" / "custom"))
    args = parser.parse_args()

    device = get_device()
    stem = args.name or Path(args.image).stem
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    texture = load_texture(args.image, size=args.size_px).to(device)
    height, width = texture.shape[:2]
    raw = raw_bytes(height, width)
    print(f"device: {device}")
    print(f"image: {args.image} -> {width}x{height}")

    panels = [texture]
    names = ["original"]
    metrics = {"image": str(args.image), "texture": stem, "width": width, "height": height}

    if not args.no_s3tc:
        s3tc = S3TC().compress(texture)
        recon = s3tc.decode()
        panels.append(recon)
        names.append("S3TC")
        metrics["s3tc"] = {
            "psnr_db": psnr(texture, recon),
            "size_bytes": s3tc.size_bytes,
            "compression_factor": raw / s3tc.size_bytes,
        }
        print(f"  S3TC    {metrics['s3tc']['psnr_db']:6.2f} dB  "
              f"{s3tc.size_bytes / 1024:7.1f} KB  {raw / s3tc.size_bytes:5.1f}x")

    curves = {}
    curve_steps = {}
    for size in args.sizes:
        scores, steps_log, psnr_log, shown = [], [], [], None
        for seed in range(args.seeds):
            codec = NeuralTextureCodec(size=size, steps=args.steps, batch=args.batch,
                                       lr=args.lr, seed=seed, log_every=max(1, args.steps // 20))
            codec.compress(texture)
            recon = codec.decode()
            scores.append(psnr(texture, recon))
            if seed == 0:
                shown = recon
                size_bytes, params = codec.size_bytes, codec.num_parameters
                steps_log = [h["step"] for h in codec.history]
                psnr_log = [h["psnr"] for h in codec.history]
        panels.append(shown)
        names.append(f"neural {size}")
        mean = statistics.mean(scores)
        curves[size] = psnr_log
        curve_steps[size] = steps_log
        metrics[size] = {
            "params": params,
            "size_bytes": size_bytes,
            "compression_factor": raw / size_bytes,
            "psnr_db": mean,
            "psnr_seeds": scores,
            "grid8_size_bytes": quantized_size_bytes(codec.model, quantize_mlp=False),
        }
        print(f"  {size:7s} {mean:6.2f} dB  {size_bytes / 1024:7.1f} KB  {raw / size_bytes:5.1f}x")

    save_image(hstack(panels, gap=6), out / f"{stem}_comparison.png")
    save_image(hstack([zoom_crop(p) for p in panels], gap=6), out / f"{stem}_zoom.png")
    (out / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for size, values in curves.items():
        ax.plot(curve_steps[size], values, color=SIZE_COLOR[size], linestyle=SIZE_STYLE[size],
                label=size)
    ax.set_xlabel("step")
    ax.set_ylabel("PSNR from batch loss (dB)")
    ax.set_title(f"Training on {stem}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{stem}_training.png", dpi=150)

    shown_out = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown_out}/{stem}_comparison.png  (panels: {', '.join(names)})")
    print(f"      {shown_out}/{stem}_zoom.png, {stem}_training.png, {stem}_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
