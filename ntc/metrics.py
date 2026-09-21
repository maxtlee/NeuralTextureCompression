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


def seam_error(original: torch.Tensor, reconstruction: torch.Tensor,
               block: int = 4) -> float:
    """Cross-boundary gradient error, a measure of blocking artifacts.

    Block compression creates color steps at block boundaries that are not in
    the original. For every pair of adjacent texels that straddle a block
    boundary, compare the reconstruction's color step to the original's:

        e = mean | (rec[a] - rec[b]) - (orig[a] - orig[b]) |

    Subtracting the original's own gradient means a perfect reconstruction
    scores 0 even on a textured image, and only *added* discontinuities count.
    Lower is better; units are normalized color in [0, 1].
    """
    height, width = original.shape[:2]
    if height % block or width % block:
        raise ValueError(f"texture {height}x{width} not divisible by {block}")

    errors = []
    for x in range(block, width, block):
        errors.append(((reconstruction[:, x] - reconstruction[:, x - 1])
                       - (original[:, x] - original[:, x - 1])).abs())
    for y in range(block, height, block):
        errors.append(((reconstruction[y] - reconstruction[y - 1])
                       - (original[y] - original[y - 1])).abs())
    return float(torch.cat([e.reshape(-1) for e in errors]).mean())


def texel_centers(height: int, width: int, device=None, dtype=torch.float32):
    """All H*W texel-center coordinates (N, 2) in [0, 1].

    Uses the (i + 0.5) / size convention, matching grid_sample align_corners=False.
    """
    v = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    u = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
