"""Image I/O: torch tensors <-> 8-bit RGB PNGs (H, W, 3) in [0, 1]."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path) -> torch.Tensor:
    """Load an image as a float32 (H, W, 3) tensor in [0, 1]."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def save_image(image: torch.Tensor, path) -> None:
    """Write a float (H, W, 3) tensor in [0, 1] as an 8-bit RGB PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round()
    Image.fromarray(arr.astype(np.uint8)).save(path)
