"""Tests for image I/O helpers (load_image, load_texture, save_image).

Usage:  .venv/bin/python tests/test_io.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ntc import load_image, load_texture, save_image  # noqa: E402


def _write(path, height, width):
    arr = (np.random.default_rng(0).random((height, width, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def test_load_image_shape_and_range():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "square.png"
        _write(path, 32, 32)
        image = load_image(path)
        assert image.shape == (32, 32, 3)
        assert image.min() >= 0.0 and image.max() <= 1.0
    print("ok: load_image returns (H, W, 3) in [0, 1]")


def test_load_texture_crops_and_rounds():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rect.jpg"
        _write(path, 40, 70)  # rectangular and not a multiple of 4
        texture = load_texture(path, size=None)
        height, width = texture.shape[:2]
        assert height == width, (height, width)  # square crop
        assert height % 4 == 0 and width % 4 == 0
    print("ok: load_texture center-crops to a square multiple of 4")


def test_load_texture_resizes():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.png"
        _write(path, 300, 200)
        texture = load_texture(path, size=128)
        assert texture.shape == (128, 128, 3)
    print("ok: load_texture resizes to the requested square size")


def test_save_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        image = torch.rand(16, 24, 3)
        path = Path(tmp) / "out.png"
        save_image(image, path)
        reloaded = load_image(path)
        assert reloaded.shape == image.shape
        assert (reloaded - image).abs().max() <= 1.0 / 255.0 + 1e-6
    print("ok: save_image / load_image round-trip within 8-bit precision")


def main() -> int:
    tests = [
        test_load_image_shape_and_range,
        test_load_texture_crops_and_rounds,
        test_load_texture_resizes,
        test_save_roundtrip,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
