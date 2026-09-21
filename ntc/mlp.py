"""Shallow decoder mapping feature channels to an RGB color.

Fixed architecture: the input width is the number of concatenated feature
channels from the grid, then two hidden layers of width 64 with ReLU, then a
3-channel output squashed to [0, 1] with a sigmoid. This is the only work the
renderer runs per texel, so it is kept deliberately small.
"""

import torch
import torch.nn as nn

HIDDEN = 64


class ColorMLP(nn.Module):
    """Two hidden ReLU layers of width 64, then a sigmoid RGB output."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, in_dim) features to (N, 3) colors in [0, 1]."""
        return self.net(x)
