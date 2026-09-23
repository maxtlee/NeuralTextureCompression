"""Small image montage helpers for experiment outputs."""

import torch
import torch.nn.functional as F


def hstack(images, gap: int = 8, value: float = 1.0) -> torch.Tensor:
    """Horizontally stack same-height (H, W, 3) images with a gap between them."""
    height = images[0].shape[0]
    parts = []
    for i, image in enumerate(images):
        if i:
            parts.append(torch.full((height, gap, 3), value, device=image.device))
        parts.append(image)
    return torch.cat(parts, dim=1)


def zoom_crop(image: torch.Tensor, size: int = 96, scale: int = 4) -> torch.Tensor:
    """Center crop, nearest-neighbor upscaled so texels/blocks are visible."""
    height, width = image.shape[:2]
    size = min(size, height, width)
    top, left = (height - size) // 2, (width - size) // 2
    crop = image[top:top + size, left:left + size].permute(2, 0, 1).unsqueeze(0)
    crop = F.interpolate(crop, scale_factor=scale, mode="nearest")
    return crop[0].permute(1, 2, 0)
