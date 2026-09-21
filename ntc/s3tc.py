"""S3TC / DXT1 (BC1) block compression implemented from scratch.

Each 4x4 block stores two RGB565 endpoint colors plus a 2-bit palette index per
texel: 2*16 + 16*2 = 64 bits = 8 bytes per 16 texels, a fixed 4 bits/texel
(6:1 against 24-bit RGB).

Endpoints are the extremes along the block's principal color axis (PCA),
refined by a few least-squares refits given the current index assignment. Blocks
are always encoded in the 4-color mode, ordering the endpoints so the first
RGB565 code is the larger one.

``sample(u, v)`` decodes the owning block and reuses the bilinear sampler on the
decoded image, so every compressor reads textures identically.
"""

import torch

from .sampler import sample_bilinear

BLOCK = 4

# palette_j = w0_j * c0 + w1_j * c1
_PALETTE_WEIGHTS = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]]
)
_RGB565_LEVELS = torch.tensor([31.0, 63.0, 31.0])


def quantize_rgb565(color: torch.Tensor):
    """Quantize (..., 3) colors in [0, 1] to RGB565. Returns (codes, dequantized).

    Uses a multiply-round-divide approximation to the hardware code->value
    mapping rather than the exact bit-replication decode.
    """
    levels = _RGB565_LEVELS.to(color.device)
    q = (color * levels).round().clamp(min=0.0)
    q = torch.minimum(q, levels)
    return q, q / levels


def pack_rgb565(q: torch.Tensor) -> torch.Tensor:
    """Pack integer (..., 3) RGB565 components into 16-bit codes."""
    r = q[..., 0].to(torch.long)
    g = q[..., 1].to(torch.long)
    b = q[..., 2].to(torch.long)
    return (r << 11) | (g << 5) | b


def unpack_rgb565(code: torch.Tensor) -> torch.Tensor:
    """Decode 16-bit RGB565 codes to (..., 3) colors in [0, 1]."""
    r = (code >> 11) & 0x1F
    g = (code >> 5) & 0x3F
    b = code & 0x1F
    levels = _RGB565_LEVELS.to(code.device)
    return torch.stack([r, g, b], dim=-1).to(torch.float32) / levels


def to_blocks(texture: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """(H, W, 3) -> (num_blocks, block*block, 3)."""
    height, width = texture.shape[:2]
    if height % block or width % block:
        raise ValueError(f"texture {height}x{width} not divisible by {block}")
    hb, wb = height // block, width // block
    b = texture.reshape(hb, block, wb, block, 3)
    b = b.permute(0, 2, 1, 3, 4)
    return b.reshape(hb * wb, block * block, 3).contiguous()


def from_blocks(blocks: torch.Tensor, height: int, width: int, block: int = BLOCK):
    """(num_blocks, block*block, 3) -> (H, W, 3)."""
    hb, wb = height // block, width // block
    b = blocks.reshape(hb, wb, block, block, 3)
    b = b.permute(0, 2, 1, 3, 4)
    return b.reshape(height, width, 3)


def _palette(c0: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
    """Build the 4-color palette (num_blocks, 4, 3) from two endpoints."""
    c2 = (2 * c0 + c1) / 3.0
    c3 = (c0 + 2 * c1) / 3.0
    return torch.stack([c0, c1, c2, c3], dim=1)


def _assign_indices(points: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Nearest palette entry per texel. points (B, N, 3), palette (B, 4, 3)."""
    dist = ((points.unsqueeze(2) - palette.unsqueeze(1)) ** 2).sum(dim=-1)
    return dist.argmin(dim=2)


def _pca_endpoints(points: torch.Tensor, iters: int = 8):
    """Two endpoints = extremes along the block's principal color axis."""
    mean = points.mean(dim=1, keepdim=True)
    centered = points - mean
    cov = centered.transpose(1, 2) @ centered

    v = torch.ones(points.shape[0], 3, 1, device=points.device, dtype=points.dtype)
    for _ in range(iters):
        v = cov @ v
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)

    t = (centered @ v).squeeze(-1)
    b_idx = torch.arange(points.shape[0], device=points.device)
    return points[b_idx, t.argmin(dim=1)], points[b_idx, t.argmax(dim=1)]


def _least_squares_endpoints(points, indices, fallback0, fallback1):
    """Refit endpoints after an index assignment (normal equations per block)."""
    w = _PALETTE_WEIGHTS.to(points.device)[indices]
    a, b = w[..., 0], w[..., 1]
    saa = (a * a).sum(dim=1)
    sab = (a * b).sum(dim=1)
    sbb = (b * b).sum(dim=1)
    ta = (a.unsqueeze(-1) * points).sum(dim=1)
    tb = (b.unsqueeze(-1) * points).sum(dim=1)

    det = saa * sbb - sab * sab
    ok = det.abs() > 1e-8
    det_safe = torch.where(ok, det, torch.ones_like(det)).unsqueeze(-1)
    c0 = (sbb.unsqueeze(-1) * ta - sab.unsqueeze(-1) * tb) / det_safe
    c1 = (saa.unsqueeze(-1) * tb - sab.unsqueeze(-1) * ta) / det_safe

    mask = ok.unsqueeze(-1)
    c0 = torch.where(mask, c0.clamp(0.0, 1.0), fallback0)
    c1 = torch.where(mask, c1.clamp(0.0, 1.0), fallback1)
    return c0, c1


def finalize_blocks(points, c0, c1):
    """Quantize endpoints to RGB565, order into 4-color mode, assign indices.

    Returns the same (code_hi, code_lo, indices) triple as ``encode_blocks``.
    """
    q0, c0q = quantize_rgb565(c0)
    q1, c1q = quantize_rgb565(c1)
    code0 = pack_rgb565(q0)
    code1 = pack_rgb565(q1)

    # 4-color mode needs code_hi > code_lo; order the dequantized endpoints too.
    swap = code0 < code1
    swap3 = swap.unsqueeze(-1)
    endpoint_hi = torch.where(swap3, c1q, c0q)
    endpoint_lo = torch.where(swap3, c0q, c1q)
    code_hi = torch.where(swap, code1, code0)
    code_lo = torch.where(swap, code0, code1)

    indices = _assign_indices(points, _palette(endpoint_hi, endpoint_lo))
    return code_hi, code_lo, indices


def encode_blocks(texture: torch.Tensor, block: int = BLOCK, refine_iters: int = 2):
    """Encode a texture into DXT1 blocks.

    Returns (code_hi, code_lo, indices):
      code_hi, code_lo: (B,) RGB565 codes, first >= second (4-color mode)
      indices: (B, block*block) values in {0, 1, 2, 3}
    """
    points = to_blocks(texture, block)
    c0, c1 = _pca_endpoints(points)

    for _ in range(refine_iters):
        idx = _assign_indices(points, _palette(c0, c1))
        c0, c1 = _least_squares_endpoints(points, idx, c0, c1)

    return finalize_blocks(points, c0, c1)


def decode_blocks(code_hi, code_lo, indices):
    """Rebuild (B, block*block, 3) colors from DXT1 codes and indices."""
    c0 = unpack_rgb565(code_hi)
    c1 = unpack_rgb565(code_lo)
    palette = _palette(c0, c1)
    block = indices.shape[1]
    rows = torch.arange(indices.shape[0], device=indices.device).unsqueeze(1)
    return palette[rows, indices]


class S3TC:
    """S3TC / DXT1 compressor.

    Exposes the same ``compress(texture)`` / ``sample(uv)`` interface as every
    compressor, so callers can swap representations transparently.
    """

    def __init__(self, block: int = BLOCK, refine_iters: int = 2):
        self.block = block
        self.refine_iters = refine_iters
        self.code_hi = None
        self.code_lo = None
        self.indices = None
        self.height = None
        self.width = None
        self._decoded = None

    def compress(self, texture: torch.Tensor) -> "S3TC":
        """Encode (H, W, 3) texture in [0, 1] into 4x4 DXT1 blocks."""
        self.height, self.width = texture.shape[:2]
        self.code_hi, self.code_lo, self.indices = encode_blocks(
            texture, self.block, self.refine_iters
        )
        self._decoded = None
        return self

    def decode(self) -> torch.Tensor:
        """Decode the full texture (H, W, 3)."""
        if self._decoded is None:
            blocks = decode_blocks(self.code_hi, self.code_lo, self.indices)
            self._decoded = from_blocks(blocks, self.height, self.width, self.block)
        return self._decoded

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        """Decode the block image, then bilinearly sample it."""
        return sample_bilinear(self.decode(), uv)

    @property
    def size_bytes(self) -> int:
        """Fixed 8 bytes per 4x4 block = W*H/2 bytes."""
        return self.code_hi.numel() * 8
