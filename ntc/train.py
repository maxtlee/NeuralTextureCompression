"""Training loop: fit a ``NeuralTexture`` to a single target texture.

The training set is every texel location. The full list of ``H*W`` texel-center
coordinates and their ground-truth colors is built once up front and never
changes; each step draws a random minibatch and takes a gradient step on the MSE
reconstruction loss. PSNR follows directly from the loss.
"""

import torch
import torch.nn.functional as F

from .metrics import texel_centers


def build_dataset(texture: torch.Tensor):
    """Return (coords (N, 2), targets (N, 3)) for every texel of ``texture``.

    Coordinates are texel centers in [0, 1]; targets are the flattened colors.
    """
    height, width = texture.shape[:2]
    coords = texel_centers(height, width, device=texture.device, dtype=texture.dtype)
    targets = texture.reshape(-1, 3).contiguous()
    return coords, targets


def train(model, coords, targets, steps: int = 2000, batch: int = 16384,
          lr: float = 1e-2, log_every: int = 100, seed: int = 0):
    """Fit ``model`` with Adam on random minibatches of the fixed training set.

    Returns a history list of dicts with the step, batch MSE loss, and the PSNR
    implied by that loss (-10*log10(loss)).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    count = coords.shape[0]
    batch = min(batch, count)
    generator = torch.Generator(device=coords.device)
    generator.manual_seed(seed)

    history = []
    for step in range(steps):
        index = torch.randint(0, count, (batch,), device=coords.device, generator=generator)
        prediction = model(coords[index])
        loss = F.mse_loss(prediction, targets[index])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_every and (step % log_every == 0 or step == steps - 1):
            value = loss.item()
            history.append({
                "step": step,
                "loss": value,
                "psnr": float("inf") if value == 0 else -10.0 * torch.log10(torch.tensor(value)).item(),
            })
    return history


@torch.no_grad()
def decode(model, height: int, width: int) -> torch.Tensor:
    """Render the full texture by evaluating the model at every texel center."""
    coords = texel_centers(height, width, device=next(model.parameters()).device,
                           dtype=next(model.parameters()).dtype)
    return model(coords).reshape(height, width, 3)
