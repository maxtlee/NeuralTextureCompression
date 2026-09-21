"""Learned texture representation: a feature grid decoded by a small MLP.

The model maps a continuous coordinate ``(u, v)`` to a color. The grid holds the
texture's information; the MLP turns blended features into RGB. Fit per texture:
the weights encode one image and nothing else.

The three sizes used for the size-vs-quality comparison differ only in the grid
setup (all share the same 2x64 decoder).
"""

import torch
import torch.nn as nn

from .feature_grid import FeatureGrid
from .mlp import ColorMLP

MODEL_SIZES = {
    "small": {"resolutions": (64,), "feat_dim": 2},
    "medium": {"resolutions": (16, 32, 64), "feat_dim": 2},
    "large": {"resolutions": (16, 32, 64, 128), "feat_dim": 4},
}


class NeuralTexture(nn.Module):
    """A multi-resolution feature grid plus a decoder MLP, fit to one texture."""

    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim: int = 2):
        super().__init__()
        self.grid = FeatureGrid(resolutions=resolutions, feat_dim=feat_dim)
        self.mlp = ColorMLP(self.grid.out_dim)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Map (N, 2) coordinates in [0, 1] to (N, 3) colors in [0, 1]."""
        return self.mlp(self.grid(uv))

    @classmethod
    def from_size(cls, size: str = "small") -> "NeuralTexture":
        """Build one of the named architectures in ``MODEL_SIZES``."""
        if size not in MODEL_SIZES:
            raise ValueError(f"unknown size {size!r}; choose from {list(MODEL_SIZES)}")
        return cls(**MODEL_SIZES[size])

    @property
    def num_parameters(self) -> int:
        """Total stored values in the grid plus the decoder."""
        return sum(p.numel() for p in self.parameters())
