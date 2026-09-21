"""Multi-resolution feature grid with bilinear lookup.

At each resolution the grid stores a learnable ``(1, feat_dim, R, R)`` tensor of
feature vectors. A lookup scales the ``(u, v)`` coordinate into grid space and
bilinearly blends the four surrounding cells, using the same texel-center
convention as the full-resolution sampler (``align_corners=False``, border
clamping). The blended features from every resolution are concatenated, so coarse
levels carry broad structure and fine levels carry detail: with ``L`` levels of
width ``feat_dim`` the result has ``L * feat_dim`` channels.

The concatenated features are meant to be fed to a small decoder that maps them
to a color; the grid is where the texture's information actually lives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGrid(nn.Module):
    """A stack of learnable grids sampled bilinearly at continuous coordinates."""

    def __init__(self, resolutions=(16, 32, 64, 128), feat_dim: int = 2):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.feat_dim = feat_dim
        self.grids = nn.ParameterList(
            nn.Parameter(torch.empty(1, feat_dim, r, r)) for r in self.resolutions
        )
        self.out_dim = feat_dim * len(self.resolutions)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Small random init so the levels start near zero but not symmetric."""
        for grid in self.grids:
            nn.init.uniform_(grid, -0.1, 0.1)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Sample the concatenated features at ``uv`` (N, 2) in [0, 1].

        Returns (N, out_dim).
        """
        uv = uv.reshape(-1, 2)
        # grid_sample wants coordinates in [-1, 1], shape (batch, Hout, Wout, 2).
        coords = (uv * 2.0 - 1.0).reshape(1, -1, 1, 2)
        features = []
        for grid in self.grids:
            sampled = F.grid_sample(
                grid,
                coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            features.append(sampled.reshape(self.feat_dim, -1).t())
        return torch.cat(features, dim=1)

    @property
    def num_parameters(self) -> int:
        """Total number of stored feature values across every level."""
        return sum(grid.numel() for grid in self.grids)
