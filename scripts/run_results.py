"""Size-vs-quality comparison of the learned representation against S3TC.

Trains the three architectures on each of the three textures, averaged over
several seeds to smooth out GPU nondeterminism, and writes:

  experiments/results/raw.json              all runs, aggregates, training curves, S3TC
  writeup/assets/{texture}_neural.png       original vs the three reconstructions
  writeup/assets/p6_training.png            PSNR over training, all 9 combinations
  writeup/assets/p6_size_quality.png        PSNR vs stored size, S3TC overlaid

Training takes a few minutes; use --plot-only to rebuild the figures from
experiments/results/raw.json without retraining.

Usage:
  .venv/bin/python scripts/run_results.py
  .venv/bin/python scripts/run_results.py --seeds 0,1,2 --steps 2000
  .venv/bin/python scripts/run_results.py --plot-only
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import S3TC, get_device, load_image, psnr, raw_bytes, save_image  # noqa: E402
from ntc.codec import NeuralTextureCodec  # noqa: E402
from ntc.viz import hstack  # noqa: E402

TEXTURES = ["gradient", "bricks", "clouds"]
SIZES = ["small", "medium", "large"]
TEXTURE_COLOR = {"gradient": "#2f5d8a", "bricks": "#8a2f2f", "clouds": "#2f8f4e"}
TEXTURE_MARKER = {"gradient": "o", "bricks": "s", "clouds": "^"}
SIZE_COLOR = {"small": "#2f8f4e", "medium": "#e08a1e", "large": "#2f5d8a"}
SIZE_STYLE = {"small": "-", "medium": "--", "large": ":"}

EXPERIMENTS = ROOT / "experiments" / "results"
ASSETS = ROOT / "writeup" / "assets"


def run_experiments(args, seeds, device):
    runs = []
    baseline = {}
    curves = {}
    for texture_name in TEXTURES:
        texture = load_image(ROOT / "assets" / "provided" / f"{texture_name}.png").to(device)
        height, width = texture.shape[:2]

        s3tc = S3TC().compress(texture)
        baseline[texture_name] = {
            "psnr_db": psnr(texture, s3tc.decode()),
            "size_bytes": s3tc.size_bytes,
            "compression_factor": raw_bytes(height, width) / s3tc.size_bytes,
        }

        panels = [texture]
        for size in SIZES:
            scores, history, shown = [], [], None
            for seed in seeds:
                codec = NeuralTextureCodec(size=size, steps=args.steps, batch=args.batch,
                                           lr=args.lr, seed=seed, log_every=100)
                codec.compress(texture)
                recon = codec.decode()
                scores.append(psnr(texture, recon))
                history.append([h["psnr"] for h in codec.history])
                if seed == seeds[0]:
                    shown = recon
                    size_bytes, params = codec.size_bytes, codec.num_parameters
            panels.append(shown)
            curves.setdefault(texture_name, {})[size] = history
            runs.append({
                "texture": texture_name, "size": size, "params": params,
                "size_bytes": size_bytes, "raw_bytes": raw_bytes(height, width),
                "compression_factor": raw_bytes(height, width) / size_bytes,
                "psnr_mean": statistics.mean(scores),
                "psnr_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "psnr_seeds": scores,
            })
            print(f"  {texture_name:9s} {size:7s} PSNR {runs[-1]['psnr_mean']:6.2f} "
                  f"+/- {runs[-1]['psnr_std']:.2f}  {size_bytes / 1024:6.1f} KB  "
                  f"{runs[-1]['compression_factor']:.1f}x")

        save_image(hstack(panels, gap=6), ASSETS / f"{texture_name}_neural.png")

    return {
        "baseline_s3tc": baseline,
        "neural": runs,
        "training_curves": curves,
        "config": {"seeds": seeds, "steps": args.steps, "batch": args.batch, "lr": args.lr},
    }


def plot_training(payload):
    steps = list(range(0, payload["config"]["steps"], 100))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for texture_name, by_size in payload["training_curves"].items():
        for size, history in by_size.items():
            mean = [sum(c[i] for c in history) / len(history) for i in range(len(steps))]
            lo = [min(c[i] for c in history) for i in range(len(steps))]
            hi = [max(c[i] for c in history) for i in range(len(steps))]
            ax.plot(steps, mean, color=TEXTURE_COLOR[texture_name], linestyle=SIZE_STYLE[size],
                    linewidth=1.5, label=f"{texture_name} / {size}")
            ax.fill_between(steps, lo, hi, color=TEXTURE_COLOR[texture_name], alpha=0.10, linewidth=0)
    ax.set_xlabel("step")
    ax.set_ylabel("PSNR from batch loss (dB)")
    ax.set_title("Training curves, mean of 5 seeds (shaded: min-max)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "p6_training.png", dpi=150)


def plot_size_quality(payload):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for entry in payload["neural"]:
        for score in entry["psnr_seeds"]:
            ax.scatter(entry["size_bytes"] / 1024, score,
                       marker=TEXTURE_MARKER[entry["texture"]], color=SIZE_COLOR[entry["size"]],
                       s=16, alpha=0.30, linewidth=0)
        ax.errorbar(entry["size_bytes"] / 1024, entry["psnr_mean"], yerr=entry["psnr_std"],
                    marker=TEXTURE_MARKER[entry["texture"]], color=SIZE_COLOR[entry["size"]],
                    linestyle="", capsize=2, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    for texture_name, base in payload["baseline_s3tc"].items():
        ax.scatter(base["size_bytes"] / 1024, base["psnr_db"],
                   marker=TEXTURE_MARKER[texture_name], color="#111111", s=55,
                   edgecolor="white", linewidth=0.6, zorder=5)
    # Color carries the size/baseline, so the legend uses color patches; the
    # marker shapes stay reserved for the texture and are keyed separately.
    size_handles = [Patch(facecolor=c, label=s) for s, c in SIZE_COLOR.items()]
    size_handles.append(Patch(facecolor="#111111", label="S3TC (fixed 6x)"))
    texture_handles = [plt.Line2D([], [], marker=m, linestyle="none", markerfacecolor="white",
                                  markeredgecolor="#444444", markeredgewidth=1.1, markersize=8,
                                  label=t)
                       for t, m in TEXTURE_MARKER.items()]
    leg_size = ax.legend(handles=size_handles, title="size / baseline", loc="lower center",
                         ncol=4, fontsize=8, frameon=False)
    ax.add_artist(leg_size)
    ax.legend(handles=texture_handles, title="texture", loc="upper center", ncol=3,
              fontsize=8, frameon=False)
    ax.set_xscale("log")
    ax.set_xlabel("stored size (KB, log scale, lower is better)")
    ax.set_ylabel("PSNR dB (higher is better)")
    ax.set_title("Size vs quality, neural models and S3TC (5 seeds)")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(ASSETS / "p6_size_quality.png", dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    raw_path = EXPERIMENTS / "raw.json"

    if args.plot_only:
        payload = json.loads(raw_path.read_text())
        print(f"loaded {raw_path}")
    else:
        payload = run_experiments(args, seeds, get_device())
        raw_path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {raw_path}")

    plot_training(payload)
    plot_size_quality(payload)
    print(f"wrote p6_training.png and p6_size_quality.png to {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
