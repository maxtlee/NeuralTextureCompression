"""Device selection for the NTC project."""

import torch


def get_device() -> str:
    """Return "cuda", "mps", or "cpu" in order of preference."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
