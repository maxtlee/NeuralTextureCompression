"""Test the neural representation on textures sourced online.

Runs S3TC and the three neural sizes on every texture in assets/own/, records
float and 8-bit quality, a simple high-frequency detail score, and writes:

  experiments/own/metrics.json
  writeup/assets/own_size_quality.png

Usage:
  .venv/bin/python scripts/run_own_textures.py
  .venv/bin/python scripts/run_own_textures.py --seeds 0,1
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import S3TC, get_device, load_image, psnr, raw_bytes  # noqa: E402
from ntc.codec import NeuralTextureCodec  # noqa: E402
from ntc.quantize import quantize_model, quantized_size_bytes  # noqa: E402

SIZES = ["small", "medium", "large"]
SIZE_MARKER = {"small": "o", "medium": "s", "large": "^"}
SIZE_STYLE = {"small": "-", "medium": "--", "large": ":"}
PALETTE = ["#2f5d8a", "#8a2f2f", "#2f8f4e", "#8a6d2f", "#6a2f8a",
           "#2f8a8a", "#8a2f6a", "#555555"]
EXPERIMENTS = ROOT / "experiments" / "own"
ASSETS = ROOT / "writeup" / "assets"


def detail_score(texture: torch.Tensor) -> float:
    """Mean absolute Laplacian of the grayscale image: high-frequency energy."""
    gray = texture.mean(dim=2, keepdim=True).permute(2, 0, 1).unsqueeze(0)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                          device=texture.device).reshape(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
    return float(lap.abs().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = get_device()
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "assets" / "own").glob("*.png"))
    print(f"device: {device}  textures: {len(paths)}  seeds: {seeds}")

    results = []
    for path in paths:
        texture = load_image(path).to(device)
        height, width = texture.shape[:2]
        raw = raw_bytes(height, width)
        detail = detail_score(texture)

        s3tc = S3TC().compress(texture)
        baseline = {"psnr_db": psnr(texture, s3tc.decode()), "size_bytes": s3tc.size_bytes,
                    "compression_factor": raw / s3tc.size_bytes}

        row = {"texture": path.stem, "detail": detail, "s3tc": baseline, "sizes": {}}
        for size in SIZES:
            float_scores, grid_scores = [], []
            for seed in seeds:
                codec = NeuralTextureCodec(size=size, steps=args.steps, batch=args.batch,
                                           lr=args.lr, seed=seed, log_every=0)
                codec.compress(texture)
                float_scores.append(psnr(texture, codec.decode()))
                state = {k: v.detach().clone() for k, v in codec.model.state_dict().items()}
                quantize_model(codec.model, quantize_mlp=False)
                grid_scores.append(psnr(texture, codec.decode()))
                codec.model.load_state_dict(state)
            row["sizes"][size] = {
                "float_mean": statistics.mean(float_scores),
                "grid8_mean": statistics.mean(grid_scores),
                "size_bytes": codec.size_bytes,
                "grid8_bytes": quantized_size_bytes(codec.model, quantize_mlp=False),
            }
        results.append(row)
        print(f"  {path.stem:9s} detail {detail:.4f} | S3TC {baseline['psnr_db']:5.1f} | "
              + " | ".join(f"{s} {row['sizes'][s]['float_mean']:5.1f}/{row['sizes'][s]['grid8_mean']:5.1f}"
                           for s in SIZES))

    (EXPERIMENTS / "metrics.json").write_text(json.dumps(results, indent=2))

    colors = {row["texture"]: PALETTE[i % len(PALETTE)] for i, row in enumerate(results)}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for row in results:
        color = colors[row["texture"]]
        xs = [row["sizes"][s]["size_bytes"] / 1024 for s in SIZES]
        ys = [row["sizes"][s]["float_mean"] for s in SIZES]
        ax.plot(xs, ys, color=color, marker="o", linewidth=1.3, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5, label=row["texture"])
        ax.scatter([row["s3tc"]["size_bytes"] / 1024], [row["s3tc"]["psnr_db"]],
                   marker="*", color=color, s=120, edgecolor="white", linewidth=0.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("stored size (KB, log scale)")
    ax.set_ylabel("PSNR dB (float32, 2 seeds)")
    ax.set_title("Sourced textures: size vs quality (line = small/medium/large, star = S3TC)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "own_size_quality.png", dpi=150)
    print(f"\nwrote {EXPERIMENTS}/metrics.json and {ASSETS}/own_size_quality.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
