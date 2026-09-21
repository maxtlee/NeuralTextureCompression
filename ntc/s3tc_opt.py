"""Seam-aware variants of the S3TC encoder.

The baseline encoder fits every 4x4 block independently to minimize its own MSE.
That leaves endpoint jumps at block boundaries, which show up as visible
blocking. These variants keep the same bit budget and decoder but change how the
four-color palette is chosen, using information from neighboring blocks or from
the whole image:

- ``diffusion``: block error diffusion. A block's reconstruction error on its
  right/bottom edges is fed forward into the targets of the blocks to its
  right/below before they are fit, so errors cancel across seams.
- ``boundary``: boundary-weighted refit. Edge texels are refit toward the
  already-decoded colors of the neighboring block, and weighted more heavily.
- ``icm``: joint seam-energy optimization. Iterated conditional modes: each
  block tries a few candidate endpoint pairs and keeps the one minimizing its
  own error plus a penalty on the color mismatch across its seams.

All three return the same (code_hi, code_lo, indices) triple, so the decoder and
the sampler are unchanged. See ``scripts/run_s3tc_optimizations.py`` for
evaluation against the baseline.
"""

import torch

from .s3tc import (
    BLOCK,
    S3TC,
    _assign_indices,
    _least_squares_endpoints,
    _PALETTE_WEIGHTS,
    _palette,
    _pca_endpoints,
    finalize_blocks,
    quantize_rgb565,
    to_blocks,
)


def _gather(palette: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select palette colors per texel. palette (B, 4, 3), indices (B, N)."""
    rows = torch.arange(palette.shape[0], device=palette.device).unsqueeze(1)
    return palette[rows, indices]


def _decoded_and_indices(points, c0, c1):
    """Decode through RGB565-quantized endpoints, as the renderer will.

    Returns (decoded (B, N, 3), indices (B, N)). All three methods optimize this
    actual decode rather than the float palette, since quantization is a real
    part of the artifact.
    """
    _, c0q = quantize_rgb565(c0)
    _, c1q = quantize_rgb565(c1)
    palette = _palette(c0q, c1q)
    indices = _assign_indices(points, palette)
    return _gather(palette, indices), indices


def _least_squares_weighted(points, indices, fallback0, fallback1, weights=None):
    """Weighted least-squares refit of the two endpoints.

    ``weights`` is per-texel (B, N); None means uniform.
    """
    w = _PALETTE_WEIGHTS.to(points.device)[indices]
    a, b = w[..., 0], w[..., 1]
    if weights is None:
        weights = torch.ones(points.shape[:2], device=points.device, dtype=points.dtype)
    if weights.dim() == 3:
        weights = weights.squeeze(-1)

    wa, wb = weights * a, weights * b
    saa = (wa * a).sum(dim=1)
    sab = (wa * b).sum(dim=1)
    sbb = (wb * b).sum(dim=1)
    ta = (wa.unsqueeze(-1) * points).sum(dim=1)
    tb = (wb.unsqueeze(-1) * points).sum(dim=1)

    det = saa * sbb - sab * sab
    ok = det.abs() > 1e-8
    det_safe = torch.where(ok, det, torch.ones_like(det)).unsqueeze(-1)
    c0 = (sbb.unsqueeze(-1) * ta - sab.unsqueeze(-1) * tb) / det_safe
    c1 = (saa.unsqueeze(-1) * tb - sab.unsqueeze(-1) * ta) / det_safe

    mask = ok.unsqueeze(-1)
    c0 = torch.where(mask, c0.clamp(0.0, 1.0), fallback0)
    c1 = torch.where(mask, c1.clamp(0.0, 1.0), fallback1)
    return c0, c1


def _facing(decoded_grid: torch.Tensor):
    """Neighbor colors each edge should match, (hb, wb, 3) each.

    ``decoded_grid`` is (hb, wb, block, block, 3). At image borders the face is
    set to the block's own edge, so it contributes zero mismatch.
    """
    hb, wb = decoded_grid.shape[:2]
    right = decoded_grid[:, :, :, -1, :].clone()
    left = decoded_grid[:, :, :, 0, :].clone()
    bottom = decoded_grid[:, :, -1, :, :].clone()
    top = decoded_grid[:, :, 0, :, :].clone()

    if wb > 1:
        right[:, :-1] = left[:, 1:]
        left[:, 1:] = right[:, :-1]
    if hb > 1:
        bottom[:-1, :] = top[1:, :]
        top[1:, :] = bottom[:-1, :]
    return left, right, top, bottom


def _edge_targets(points, decoded_grid, block):
    """Per-texel boundary target (B, N, 3) and edge mask (B, N, 1).

    The target preserves the *original* cross-boundary gradient: for each edge
    texel, ``target = neighbor_decoded + (own_original_edge - neighbor_original)``.
    So the block is pulled until its step across the seam matches the original's,
    which is exactly the artifact the seam metric measures. At image borders the
    original gradient term is zero, so no pull is applied.
    """
    hb, wb = decoded_grid.shape[:2]
    d_left, d_right, d_top, d_bottom = _facing(decoded_grid)
    orig = points.reshape(hb, wb, block, block, 3)
    o_left, o_right, o_top, o_bottom = _facing(orig)

    o_r = orig[:, :, :, -1, :]
    o_l = orig[:, :, :, 0, :]
    o_b = orig[:, :, -1, :, :]
    o_t = orig[:, :, 0, :, :]

    target_right = d_right + (o_r - o_right)
    target_left = d_left + (o_l - o_left)
    target_bottom = d_bottom + (o_b - o_bottom)
    target_top = d_top + (o_t - o_top)

    total = torch.zeros_like(orig)
    count = torch.zeros(hb, wb, block, block, 1, device=points.device, dtype=points.dtype)
    total[:, :, :, block - 1, :] += target_right
    total[:, :, :, 0, :] += target_left
    total[:, :, block - 1, :, :] += target_bottom
    total[:, :, 0, :, :] += target_top
    count[:, :, :, block - 1, :] += 1
    count[:, :, :, 0, :] += 1
    count[:, :, block - 1, :, :] += 1
    count[:, :, 0, :, :] += 1

    neighbor = (total / count.clamp(min=1.0)).reshape(-1, block * block, 3)
    mask = (count > 0).to(points.dtype).reshape(-1, block * block, 1)
    return neighbor, mask


def encode_error_diffusion(texture, block=BLOCK, iters=4, alpha=0.1):
    """Block error diffusion: push each block's edge error into its neighbors."""
    height, width = texture.shape[:2]
    hb, wb = height // block, width // block
    points = to_blocks(texture, block)
    c0, c1 = _pca_endpoints(points)

    for _ in range(iters):
        decoded, indices = _decoded_and_indices(points, c0, c1)
        error = (decoded - points).reshape(hb, wb, block, block, 3)

        adjusted = points.reshape(hb, wb, block, block, 3).clone()
        if wb > 1:
            right_edge = error[:, :, :, -1, :]
            left_pull = torch.zeros_like(error[:, :, :, 0, :])
            left_pull[:, 1:] = right_edge[:, :-1]
            adjusted[:, :, :, 0, :] += alpha * left_pull
        if hb > 1:
            bottom_edge = error[:, :, -1, :, :]
            top_pull = torch.zeros_like(error[:, :, 0, :, :])
            top_pull[1:, :] = bottom_edge[:-1, :]
            adjusted[:, :, 0, :, :] += alpha * top_pull

        c0, c1 = _least_squares_weighted(
            adjusted.reshape(-1, block * block, 3), indices, c0, c1
        )

    return finalize_blocks(points, c0, c1)


def encode_boundary_refit(texture, block=BLOCK, iters=4, gamma=0.5,
                          boundary_weight=1.0):
    """Boundary-weighted refit: pull edge texels toward the decoded neighbors."""
    height, width = texture.shape[:2]
    hb, wb = height // block, width // block
    points = to_blocks(texture, block)
    c0, c1 = _pca_endpoints(points)

    for _ in range(iters):
        decoded, indices = _decoded_and_indices(points, c0, c1)
        decoded = decoded.reshape(hb, wb, block, block, 3)

        neighbor, mask = _edge_targets(points, decoded, block)
        adjusted = points + gamma * (neighbor - points) * mask
        weights = torch.where(mask.squeeze(-1) > 0, boundary_weight, 1.0)
        c0, c1 = _least_squares_weighted(adjusted, indices, c0, c1, weights)

    return finalize_blocks(points, c0, c1)


def _shift_grid(grid: torch.Tensor, di: int, dj: int) -> torch.Tensor:
    """Neighbor grid sample at (i+di, j+dj), clamped at the border."""
    hb, wb = grid.shape[:2]
    ii = (torch.arange(hb, device=grid.device) + di).clamp(0, hb - 1)
    jj = (torch.arange(wb, device=grid.device) + dj).clamp(0, wb - 1)
    return grid[ii][:, jj]


def encode_icm(texture, block=BLOCK, iters=4, lam=2.0, beta=0.5):
    """Iterated conditional modes over candidate endpoints with a seam penalty."""
    height, width = texture.shape[:2]
    hb, wb = height // block, width // block
    points = to_blocks(texture, block)
    pca0, pca1 = _pca_endpoints(points)
    c0, c1 = pca0.clone(), pca1.clone()

    # The baseline fit is a useful candidate to keep in the search set.
    ls0, ls1 = pca0.clone(), pca1.clone()
    for _ in range(2):
        idx = _assign_indices(points, _palette(ls0, ls1))
        ls0, ls1 = _least_squares_endpoints(points, idx, ls0, ls1)

    for _ in range(iters):
        decoded, _ = _decoded_and_indices(points, c0, c1)
        decoded_grid = decoded.reshape(hb, wb, block, block, 3)

        # Original edges and the current decoded neighbor edges. The seam cost
        # is the *added* step across each boundary: (candidate step) minus
        # (original step), matching the seam metric.
        orig = points.reshape(hb, wb, block, block, 3)
        o_r = orig[:, :, :, -1, :]
        o_l = orig[:, :, :, 0, :]
        o_b = orig[:, :, -1, :, :]
        o_t = orig[:, :, 0, :, :]
        cur_l = decoded_grid[:, :, :, 0, :]
        cur_t = decoded_grid[:, :, 0, :, :]

        c0g, c1g = c0.reshape(hb, wb, 3), c1.reshape(hb, wb, 3)
        cand0 = [c0, pca0, ls0]
        cand1 = [c1, pca1, ls1]
        for di, dj in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cand0.append((1 - beta) * c0 + beta * _shift_grid(c0g, di, dj).reshape(-1, 3))
            cand1.append((1 - beta) * c1 + beta * _shift_grid(c1g, di, dj).reshape(-1, 3))

        best_cost = None
        best0, best1 = c0, c1
        for k in range(len(cand0)):
            trial0, trial1 = cand0[k], cand1[k]
            trial, _ = _decoded_and_indices(points, trial0, trial1)
            local = ((points - trial) ** 2).sum(dim=(1, 2))

            trial_grid = trial.reshape(hb, wb, block, block, 3)
            seam = torch.zeros(hb, wb, device=points.device, dtype=points.dtype)
            if wb > 1:
                step_orig = o_r[:, :-1] - o_l[:, 1:]
                step_cand = trial_grid[:, :-1, :, -1, :] - cur_l[:, 1:]
                seam[:, :-1] += ((step_cand - step_orig) ** 2).sum(-1).sum(dim=2)
            if hb > 1:
                step_orig_b = o_b[:-1, :] - o_t[1:, :]
                step_cand_b = trial_grid[:-1, :, -1, :, :] - cur_t[1:, :]
                seam[:-1, :] += ((step_cand_b - step_orig_b) ** 2).sum(-1).sum(dim=2)
            seam = seam.reshape(-1)

            cost = local + lam * seam
            if best_cost is None:
                best_cost = cost
                best0, best1 = trial0, trial1
            else:
                better = cost < best_cost
                best_cost = torch.where(better, cost, best_cost)
                best0 = torch.where(better.unsqueeze(-1), trial0, best0)
                best1 = torch.where(better.unsqueeze(-1), trial1, best1)
        c0, c1 = best0, best1

    return finalize_blocks(points, c0, c1)


_METHODS = {
    "baseline": None,  # handled by the parent encoder
    "diffusion": encode_error_diffusion,
    "boundary": encode_boundary_refit,
    "icm": encode_icm,
}


class S3TCOpt(S3TC):
    """S3TC compressor with a selectable endpoint-selection method.

    ``method`` is one of "baseline", "diffusion", "boundary", "icm". Remaining
    keyword arguments are forwarded to the chosen encoder.
    """

    def __init__(self, method: str = "baseline", block: int = BLOCK, **options):
        super().__init__(block=block)
        if method not in _METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {list(_METHODS)}")
        self.method = method
        self.options = options

    def compress(self, texture: torch.Tensor) -> "S3TCOpt":
        self.height, self.width = texture.shape[:2]
        if self.method == "baseline":
            from .s3tc import encode_blocks

            self.code_hi, self.code_lo, self.indices = encode_blocks(
                texture, self.block, self.options.get("refine_iters", 2)
            )
        else:
            self.code_hi, self.code_lo, self.indices = _METHODS[self.method](
                texture, self.block, **self.options
            )
        self._decoded = None
        return self
