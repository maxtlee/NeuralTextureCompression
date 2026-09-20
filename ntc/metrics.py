"""Reconstruction metrics and size accounting."""

import math

import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared error over all elements."""
    return float(torch.mean((a - b) ** 2))


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR in dB. Images are normalized to [0, 1] so max_val=1.0.

    PSNR = 20*log10(max_val) - 10*log10(MSE). Exact matches return +inf.
    """
    err = mse(a, b)
    if err == 0.0:
        return math.inf
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(err)


def raw_bytes(height: int, width: int, channels: int = 3) -> int:
    """Size of the uncompressed 8-bit texture."""
    return height * width * channels


def compression_factor(raw: int, compressed: int) -> float:
    """raw / compressed; e.g. 6.0 means 6x smaller."""
    return raw / compressed


def texel_centers(height: int, width: int, device=None, dtype=torch.float32):
    """All H*W texel-center coordinates (N, 2) in [0, 1].

    Uses the (i + 0.5) / size convention, matching grid_sample align_corners=False.
    """
    v = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    u = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
