"""Fit a neural texture to one image and report quality and size.

Trains the chosen architecture on every texel of one texture (Adam, random
minibatches of the fixed texel set) and writes a reconstruction comparison and a
PSNR-over-training curve under experiments/train/.

Usage:
  .venv/bin/python scripts/run_train.py
  .venv/bin/python scripts/run_train.py --texture clouds --size large --steps 2000
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import get_device, load_image, load_texture, psnr, raw_bytes, save_image  # noqa: E402
from ntc.model import NeuralTexture  # noqa: E402
from ntc.train import build_dataset, decode, train  # noqa: E402
from ntc.viz import hstack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture", default="gradient")
    parser.add_argument("--size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--image", default=None, help="train on a custom image instead of a provided one")
    parser.add_argument("--name", default=None, help="output stem for --image (default: filename)")
    parser.add_argument("--size-px", type=int, default=512)
    args = parser.parse_args()

    device = get_device()
    if args.image:
        texture = load_texture(args.image, size=args.size_px).to(device)
        source = args.name or Path(args.image).stem
    else:
        texture = load_image(ROOT / "assets" / "provided" / f"{args.texture}.png").to(device)
        source = args.texture
    height, width = texture.shape[:2]
    coords, targets = build_dataset(texture)

    torch.manual_seed(0)
    model = NeuralTexture.from_size(args.size).to(device)
    history = train(model, coords, targets, steps=args.steps, batch=args.batch,
                    lr=args.lr, log_every=max(1, args.steps // 20))
    recon = decode(model, height, width)

    quality = psnr(texture, recon)
    total = model.num_parameters
    raw = raw_bytes(height, width)
    print(f"device: {device}")
    print(f"texture {source}  size {args.size}  {width}x{height}")
    print(f"params  {total} ({total * 4 / 1024:.1f} KB f32)  vs raw {raw / 1024:.1f} KB "
          f"= {raw / (total * 4):.1f}x")
    print(f"PSNR    {quality:.2f} dB   (batch-loss PSNR {history[-1]['psnr']:.2f} dB)")

    out = ROOT / "experiments" / "train"
    out.mkdir(parents=True, exist_ok=True)
    save_image(hstack([texture, recon], gap=6), out / f"{source}_{args.size}.png")

    steps = [h["step"] for h in history]
    values = [h["psnr"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(steps, values, color="#2f5d8a")
    ax.set_xlabel("step")
    ax.set_ylabel("PSNR from batch loss (dB)")
    ax.set_title(f"Training {args.size} on {source}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"{source}_{args.size}_psnr.png", dpi=150)
    print(f"wrote {out.relative_to(ROOT)}/{source}_{args.size}.png and _psnr.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
