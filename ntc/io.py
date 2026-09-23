"""Image I/O: torch tensors <-> 8-bit RGB PNGs (H, W, 3) in [0, 1]."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path) -> torch.Tensor:
    """Load an image as a float32 (H, W, 3) tensor in [0, 1]."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def load_texture(path, size: int = None, multiple_of: int = 4) -> torch.Tensor:
    """Load any RGB image for the pipeline.

    Center-crops to a square (so aspect ratios are preserved without stretching),
    optionally resizes to ``size`` x ``size``, and crops down to a multiple of
    ``multiple_of`` (the block codecs require multiples of 4). Returns a float32
    (H, W, 3) tensor in [0, 1].
    """
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if size:
        image = image.resize((size, size), Image.LANCZOS)
    width, height = image.size
    width -= width % multiple_of
    height -= height % multiple_of
    if (width, height) != image.size:
        image = image.crop((0, 0, width, height))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def save_image(image: torch.Tensor, path) -> None:
    """Write a float (H, W, 3) tensor in [0, 1] as an 8-bit RGB PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round()
    Image.fromarray(arr.astype(np.uint8)).save(path)
