"""Small image montage helpers for experiment outputs."""

import torch


def hstack(images, gap: int = 8, value: float = 1.0) -> torch.Tensor:
    """Horizontally stack same-height (H, W, 3) images with a gap between them."""
    height = images[0].shape[0]
    parts = []
    for i, image in enumerate(images):
        if i:
            parts.append(torch.full((height, gap, 3), value, device=image.device))
        parts.append(image)
    return torch.cat(parts, dim=1)
