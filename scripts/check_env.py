"""Environment smoke test for A1.

Verifies the venv can import the stack, that the selected device can actually
run kernels (a Pascal sm_61 card with a +cu130 torch build will pass
``cuda.is_available()`` but fail here), and that the provided textures load.

Usage:  .venv/bin/python scripts/check_env.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import get_device  # noqa: E402


def main() -> int:
    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__} (built cuda {torch.version.cuda})")
    print(f"arch list   {torch.cuda.get_arch_list()}")
    print(f"numpy       {np.__version__}")

    device = get_device()
    print(f"device      {device}")

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"gpu         {name} (sm_{cc[0]}{cc[1]})")
        try:
            x = torch.randn(1024, 1024, device="cuda")
            _ = (x @ x).sum().item()
            g = torch.randn(3, 2, 16, 16, device="cuda")
            uv = torch.rand(3, 1, 1, 2, device="cuda") * 2 - 1
            F.grid_sample(g, uv, mode="bilinear", padding_mode="border",
                          align_corners=False)
            torch.cuda.synchronize()
            print("gpu kernels OK")
        except Exception as exc:  # noqa: BLE001
            print(f"GPU KERNEL FAILED: {exc}")
            print("Reinstall torch against the cu126 index (see requirements.txt).")
            return 1

    provided = sorted((ROOT / "assets" / "provided").glob("*.png"))
    if not provided:
        print("no textures found under assets/provided/")
        return 1
    for path in provided:
        from PIL import Image

        arr = np.asarray(Image.open(path))
        print(f"texture     {path.name:14s} {arr.shape} {arr.dtype}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
