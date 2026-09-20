"""Sanity tests for the bilinear sampler and S3TC compression. No pytest required.

Usage:  .venv/bin/python tests/test_sampler_s3tc.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import psnr, texel_centers  # noqa: E402
from ntc.s3tc import (  # noqa: E402
    S3TC,
    _palette,
    decode_blocks,
    encode_blocks,
    from_blocks,
    to_blocks,
)
from ntc.sampler import sample_bilinear  # noqa: E402


def test_sampler_matches_grid_sample():
    torch.manual_seed(0)
    for height, width in [(16, 16), (8, 12), (32, 24)]:
        texels = torch.rand(height, width, 3)
        uv = torch.rand(2000, 2)
        got = sample_bilinear(texels, uv)

        grid = (uv * 2 - 1).reshape(1, -1, 1, 2)
        inp = texels.permute(2, 0, 1).unsqueeze(0)
        ref = F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        ref = ref.reshape(3, -1).t()
        assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()
    print("ok: sampler matches F.grid_sample (border, align_corners=False)")


def test_sampler_identity_at_texel_centers():
    texels = torch.rand(20, 30, 3)
    uv = texel_centers(20, 30)
    recon = sample_bilinear(texels, uv).reshape(20, 30, 3)
    assert torch.equal(recon, texels)
    print("ok: sampling at texel centers is exact")


def test_block_roundtrip():
    texture = torch.rand(8, 12, 3)
    blocks = to_blocks(texture)
    assert blocks.shape == (6, 16, 3)
    rebuilt = from_blocks(blocks, 8, 12)
    assert torch.equal(rebuilt, texture)
    print("ok: to_blocks/from_blocks round-trip")


def test_s3tc_codes_and_size():
    torch.manual_seed(1)
    texture = torch.rand(32, 32, 3)
    code_hi, code_lo, indices = encode_blocks(texture)
    blocks = texture.shape[0] * texture.shape[1] // 16
    assert code_hi.shape == (blocks,)
    assert code_lo.shape == (blocks,)
    assert indices.shape == (blocks, 16)
    assert (code_hi >= code_lo).all(), "4-color mode needs code_hi >= code_lo"
    assert indices.min() >= 0 and indices.max() <= 3

    s3tc = S3TC().compress(texture)
    assert s3tc.size_bytes == texture.shape[0] * texture.shape[1] // 2
    print("ok: S3TC blocks, endpoint ordering, and size")


def test_s3tc_decode_and_sample_match():
    torch.manual_seed(2)
    texture = torch.rand(32, 32, 3)
    s3tc = S3TC().compress(texture)
    decoded = s3tc.decode()
    assert decoded.shape == texture.shape

    blocks = decode_blocks(s3tc.code_hi, s3tc.code_lo, s3tc.indices)
    assert torch.allclose(blocks, to_blocks(decoded))

    uv = texel_centers(32, 32)
    via_s3tc = s3tc.sample(uv).reshape(32, 32, 3)
    via_p1 = sample_bilinear(decoded, uv).reshape(32, 32, 3)
    assert torch.allclose(via_s3tc, via_p1)
    print("ok: S3TC sample matches the sampler on the decoded image")


def test_s3tc_quality_flat_color():
    # A constant block must decode almost exactly (endpoints collapse).
    texture = torch.full((16, 16, 3), 0.25)
    s3tc = S3TC().compress(texture)
    quality = psnr(texture, s3tc.decode())
    assert quality > 40.0, quality
    print(f"ok: flat-color S3TC PSNR={quality:.1f} dB")


def test_palette_interpolation():
    # c2 = (2c0 + c1) / 3 and c3 = (c0 + 2c1) / 3.
    c0 = torch.tensor([[0.9, 0.1, 0.4]])
    c1 = torch.tensor([[0.1, 0.8, 0.2]])
    palette = _palette(c0, c1)
    assert torch.allclose(palette[:, 2], (2 * c0 + c1) / 3)
    assert torch.allclose(palette[:, 3], (c0 + 2 * c1) / 3)
    print("ok: palette interpolation (c2, c3)")


def main() -> int:
    tests = [
        test_sampler_matches_grid_sample,
        test_sampler_identity_at_texel_centers,
        test_block_roundtrip,
        test_s3tc_codes_and_size,
        test_s3tc_decode_and_sample_match,
        test_s3tc_quality_flat_color,
        test_palette_interpolation,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
