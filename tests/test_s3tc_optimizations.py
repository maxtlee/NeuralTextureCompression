"""Tests for the seam metric and the seam-aware S3TC variants. No pytest needed.

Usage:  .venv/bin/python tests/test_s3tc_optimizations.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import get_device, load_image, psnr  # noqa: E402
from ntc.metrics import seam_error  # noqa: E402
from ntc.s3tc_opt import S3TCOpt  # noqa: E402

METHODS = ["diffusion", "boundary", "icm"]


def test_seam_zero_for_identical():
    image = torch.rand(32, 32, 3)
    assert seam_error(image, image) == 0.0
    print("ok: seam error is 0 for an identical image")


def test_seam_detects_added_step():
    original = torch.zeros(16, 16, 3)
    reconstruction = original.clone()
    # Add a 0.5 step exactly at the x = 8 block boundary.
    reconstruction[:, 8:] = 0.5
    value = seam_error(original, reconstruction, block=4)
    assert value > 0.05, value
    print(f"ok: seam error detects an added boundary step ({value:.3f})")


def test_seam_ignores_original_gradient():
    # A ramp whose step is exactly the same in both images has no added seam.
    base = torch.linspace(0, 1, 32).reshape(1, 32, 1).repeat(32, 1, 3)
    assert seam_error(base, base) == 0.0
    print("ok: seam error ignores the original gradient")


def test_methods_produce_valid_blocks():
    torch.manual_seed(0)
    texture = torch.rand(32, 32, 3)
    for method in METHODS:
        compressor = S3TCOpt(method=method).compress(texture)
        blocks = texture.shape[0] * texture.shape[1] // 16
        assert compressor.code_hi.shape == (blocks,)
        assert compressor.code_lo.shape == (blocks,)
        assert compressor.indices.shape == (blocks, 16)
        assert (compressor.code_hi >= compressor.code_lo).all()
        assert compressor.indices.min() >= 0 and compressor.indices.max() <= 3
        assert compressor.size_bytes == texture.shape[0] * texture.shape[1] // 2
        assert compressor.decode().shape == texture.shape
    print("ok: all methods produce valid, correctly-sized DXT1 blocks")


def test_methods_are_deterministic():
    torch.manual_seed(0)
    texture = torch.rand(32, 32, 3)
    for method in METHODS:
        a = S3TCOpt(method=method).compress(texture).decode()
        b = S3TCOpt(method=method).compress(texture).decode()
        assert torch.equal(a, b)
    print("ok: all methods are deterministic")


def test_methods_reduce_seam_on_smooth_gradient():
    device = get_device()
    texture = load_image(ROOT / "assets" / "provided" / "gradient.png").to(device)
    baseline = S3TCOpt("baseline").compress(texture).decode()
    baseline_seam = seam_error(texture, baseline)
    baseline_psnr = psnr(texture, baseline)

    for method in METHODS:
        decoded = S3TCOpt(method=method).compress(texture).decode()
        seam = seam_error(texture, decoded)
        quality = psnr(texture, decoded)
        assert seam <= baseline_seam, (method, seam, baseline_seam)
        print(f"ok: {method:9s} seam {seam:.6f} <= baseline {baseline_seam:.6f}"
              f"  (PSNR {quality:.2f} vs {baseline_psnr:.2f})")
    print("ok: optimized methods reduce the seam metric on a smooth gradient")


def main() -> int:
    tests = [
        test_seam_zero_for_identical,
        test_seam_detects_added_step,
        test_seam_ignores_original_gradient,
        test_methods_produce_valid_blocks,
        test_methods_are_deterministic,
        test_methods_reduce_seam_on_smooth_gradient,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
