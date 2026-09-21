"""Learned texture representation behind the shared compress/sample interface.

Fits one of the named architectures to a texture and reports its stored size the
same way the block compressors do, so the two can be plotted on one size-vs-
quality axis.
"""

import torch

from .model import NeuralTexture
from .train import build_dataset, decode, train


class NeuralTextureCodec:
    """Train a ``NeuralTexture`` on a texture and sample it like any codec."""

    def __init__(self, size: str = "large", steps: int = 2000, batch: int = 16384,
                 lr: float = 1e-2, seed: int = 0, log_every: int = 100,
                 bytes_per_param: int = 4):
        self.size = size
        self.steps = steps
        self.batch = batch
        self.lr = lr
        self.seed = seed
        self.log_every = log_every
        self.bytes_per_param = bytes_per_param
        self.model = None
        self.height = None
        self.width = None
        self.history = []

    def compress(self, texture: torch.Tensor) -> "NeuralTextureCodec":
        """Fit the model to ``texture`` (H, W, 3) in [0, 1]."""
        self.height, self.width = texture.shape[:2]
        torch.manual_seed(self.seed)
        self.model = NeuralTexture.from_size(self.size).to(texture.device)
        coords, targets = build_dataset(texture)
        self.history = train(self.model, coords, targets, steps=self.steps,
                             batch=self.batch, lr=self.lr, log_every=self.log_every,
                             seed=self.seed)
        return self

    def sample(self, uv: torch.Tensor) -> torch.Tensor:
        """Map (N, 2) coordinates in [0, 1] to (N, 3) colors."""
        return self.model(uv)

    @torch.no_grad()
    def decode(self) -> torch.Tensor:
        """Render the full texture (H, W, 3)."""
        return decode(self.model, self.height, self.width)

    @property
    def num_parameters(self) -> int:
        return self.model.num_parameters

    @property
    def size_bytes(self) -> int:
        return self.model.num_parameters * self.bytes_per_param
