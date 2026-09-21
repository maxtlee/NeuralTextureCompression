"""Hyperparameter sweeps for the seam-aware S3TC variants.

Varies one knob at a time from the tuned default while holding the rest fixed,
and reports PSNR and seam error per texture. Writes
experiments/s3tc_opt/sweeps.json.

Usage:  .venv/bin/python scripts/sweep_s3tc_optimizations.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import S3TCOpt, get_device, load_image, psnr, seam_error  # noqa: E402

DEFAULTS = {
    "diffusion": dict(alpha=0.1, iters=4),
    "boundary": dict(gamma=0.5, boundary_weight=1.0, iters=4),
    "icm": dict(lam=2.0, beta=0.5, iters=4),
}

AXES = {
    "diffusion": {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
        "iters": [1, 2, 4, 8, 16],
    },
    "boundary": {
        "gamma": [0.25, 0.5, 0.75, 1.0],
        "boundary_weight": [0.5, 1.0, 2.0, 4.0],
        "iters": [1, 2, 4, 8, 16],
    },
    "icm": {
        "lam": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        "beta": [0.2, 0.35, 0.5, 0.65, 0.8],
        "iters": [1, 2, 4, 8, 12],
    },
}


def main() -> int:
    device = get_device()
    textures = {
        name: load_image(ROOT / "assets" / "provided" / f"{name}.png").to(device)
        for name in ("gradient", "bricks", "clouds")
    }
    baseline = {}
    for name, tex in textures.items():
        decoded = S3TCOpt("baseline").compress(tex).decode()
        baseline[name] = (psnr(tex, decoded), seam_error(tex, decoded))
    print("baseline:", {k: (round(v[0], 3), round(v[1], 6)) for k, v in baseline.items()})

    results = {}
    for method, axes in AXES.items():
        results[method] = {}
        for knob, values in axes.items():
            results[method][knob] = []
            print(f"\n{method}.{knob} (default {DEFAULTS[method].get(knob)})")
            for value in values:
                opts = dict(DEFAULTS[method])
                opts[knob] = value
                row = {"value": value, "per_texture": {}}
                for name, tex in textures.items():
                    decoded = S3TCOpt(method, **opts).compress(tex).decode()
                    quality = psnr(tex, decoded)
                    seam = seam_error(tex, decoded)
                    row["per_texture"][name] = {"psnr": quality, "seam": seam}
                g = row["per_texture"]["gradient"]
                c = row["per_texture"]["clouds"]
                row["mean_seam_ratio"] = (
                    (g["seam"] / baseline["gradient"][1])
                    + (c["seam"] / baseline["clouds"][1])
                ) / 2
                row["mean_psnr_delta"] = (
                    (g["psnr"] - baseline["gradient"][0])
                    + (c["psnr"] - baseline["clouds"][0])
                ) / 2
                results[method][knob].append(row)
                print(
                    f"  {knob}={value:<5} "
                    f"gradient {g['psnr']:.3f}/{g['seam']:.6f}  "
                    f"clouds {c['psnr']:.3f}/{c['seam']:.6f}  "
                    f"mean seam x{row['mean_seam_ratio']:.3f}  "
                    f"mean dPSNR {row['mean_psnr_delta']:+.3f}"
                )

    out = ROOT / "experiments" / "s3tc_opt"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweeps.json").write_text(json.dumps(
        {"baseline": {k: {"psnr": v[0], "seam": v[1]} for k, v in baseline.items()},
         "sweeps": results}, indent=2))
    print(f"\nwrote {out}/sweeps.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
