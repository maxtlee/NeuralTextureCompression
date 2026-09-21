"""Post-training uniform 8-bit quantization of a fitted model.

Each stored array is quantized separately over its own range: a float ``x`` is
rounded to the nearest of 256 evenly spaced levels, the index ``q`` is stored,
and decoding uses ``x_hat = lo + q * scale``. That is one byte per value instead
of four, plus the two float32 range endpoints ``(lo, scale)`` per array. The
feature grid is where the size win comes from; quantizing the tiny MLP is
optional.

The model is fitted in float32 first (training is unchanged); this is a separate
post-training step. Quality drops a little, which is the trade for the smaller
representation.
"""

import torch

BYTES_PER_VALUE = 1
BYTES_PER_RANGE = 8  # lo and scale, two float32


def quantize_uint8(tensor: torch.Tensor):
    """Quantize an array to 8-bit uniform levels over its own range.

    Returns ``(q, lo, scale, x_hat)``: the uint8 indices, the range endpoints,
    and the dequantized values used at decode time.
    """
    x = tensor.detach()
    lo = x.min()
    hi = x.max()
    if float(hi - lo) <= 0.0:
        q = torch.zeros_like(x)
        scale = torch.zeros((), device=x.device, dtype=x.dtype)
        return q.to(torch.uint8), float(lo), float(scale), x.clone()

    scale = (hi - lo) / 255.0
    q = torch.round((x - lo) / scale).clamp(0.0, 255.0)
    x_hat = lo + q * scale
    return q.to(torch.uint8), float(lo), float(scale), x_hat


def _quantized_arrays(model, quantize_mlp: bool):
    arrays = list(model.grid.grids)
    if quantize_mlp:
        arrays += list(model.mlp.parameters())
    return arrays


def quantize_model(model, quantize_mlp: bool = False):
    """Quantize the stored arrays in place; returns per-array (value, lo, scale).

    The model then renders with the quantized values, so a decode after this is
    the quantized reconstruction.
    """
    stats = []
    for array in _quantized_arrays(model, quantize_mlp):
        q, lo, scale, x_hat = quantize_uint8(array.data)
        array.data.copy_(x_hat)
        stats.append({"values": array.numel(), "lo": lo, "scale": scale})
    return stats


def quantized_size_bytes(model, quantize_mlp: bool = False) -> int:
    """Stored size after quantization: one byte per value plus range endpoints."""
    arrays = _quantized_arrays(model, quantize_mlp)
    total = sum(a.numel() for a in arrays) * BYTES_PER_VALUE + len(arrays) * BYTES_PER_RANGE
    if not quantize_mlp:
        total += sum(p.numel() for p in model.mlp.parameters()) * 4
    return total


def float_size_bytes(model) -> int:
    """Stored size before quantization: four bytes per value."""
    return sum(p.numel() for p in model.parameters()) * 4
