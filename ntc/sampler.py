"""Full-resolution bilinear texture sampler (uncompressed reference).

Coordinates follow the (i + 0.5) / size texel-center convention, which matches
PyTorch's ``F.grid_sample(..., align_corners=False)``. Every compressor in this
package reads textures through that same convention, so measured differences
come from the representation, not from how the texture is read.
"""

import torch


def sample_bilinear(texels: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``texels`` (H, W, 3) at ``uv`` (N, 2) in [0, 1].

    Out-of-range lookups clamp to the border (replicate padding), like
    ``F.grid_sample(padding_mode="border")``.

    Returns (N, 3) colors.
    """
    height, width = texels.shape[:2]
    uv = uv.reshape(-1, 2)

    # Continuous texel coordinates; texel centers sit at integer positions.
    x = (uv[:, 0] * width - 0.5).clamp(0.0, width - 1.0)
    y = (uv[:, 1] * height - 0.5).clamp(0.0, height - 1.0)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = (x0 + 1.0).clamp(max=width - 1.0)
    y1 = (y0 + 1.0).clamp(max=height - 1.0)
    x0 = x0.to(torch.long)
    y0 = y0.to(torch.long)
    x1 = x1.to(torch.long)
    y1 = y1.to(torch.long)

    s = (x - x0.to(x.dtype)).unsqueeze(1)
    t = (y - y0.to(y.dtype)).unsqueeze(1)

    c00 = texels[y0, x0]
    c10 = texels[y0, x1]
    c01 = texels[y1, x0]
    c11 = texels[y1, x1]

    return (
        (1 - s) * (1 - t) * c00
        + s * (1 - t) * c10
        + (1 - s) * t * c01
        + s * t * c11
    )


class FullResSampler:
    """Bilinear sampler over a full-resolution texture (the reference method).

    Exposes the same ``compress(texture)`` / ``sample(uv)`` interface as every
    compressor, so callers can swap representations transparently.
    """

    def __init__(self):
        self.texels = None

    def compress(self, texture: torch.Tensor) -> "FullResSampler":
        """Store the full-resolution texture (H, W, 3) in [0, 1]."""
        self.texels = texture
        return self

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        """Return (N, 3) colors for (N, 2) coordinates in [0, 1]."""
        return sample_bilinear(self.texels, uv)

    @property
    def size_bytes(self) -> int:
        """Uncompressed size: 3 bytes per texel (8-bit RGB)."""
        height, width = self.texels.shape[:2]
        return height * width * 3
