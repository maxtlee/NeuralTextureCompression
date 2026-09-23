# NeuralTextureCompression

Implementing a neural solution for texture compression and lookup (i.e. training memory-efficient mappings from `(u, v)` coordinate in → `(r, g, b)` color out)

- One neural network will be generated per texture
- Instead of storing every texel, fit each texture with a small multi-resolution feature grid plus a shallow decoder MLP.
- Results (size and PSNR) will be compared against classic S3TC/DXT1 block compression.

Created as part of 15-474/674 (Neural Graphics) Assignment 1

## Setup

Needs Python 3.10+ (developed on 3.14) and pip.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_env.py
```

- If PowerShell blocks the activate script: run `Set-ExecutionPolicy -Scope Process Bypass` first, or activate in `cmd` with `.venv\Scripts\activate.bat`.

**GPU notes**

- macOS: PyTorch uses the Apple GPU (MPS) automatically, nothing to configure.
- Windows / Linux with an NVIDIA GPU: the default PyPI PyTorch wheel targets a recent CUDA and works on most cards.
- My GPU is an old GTX 1050 Ti (Pascal, sm_61). The default CUDA 13 wheel dropped Pascal and fails with "no kernel image", so I use the CUDA 12.6 build:
  ```bash
  pip install -r requirements-cu126.txt
  ```
- CPU-only: `pip install numpy pillow matplotlib tqdm`, then `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
- The code selects CUDA, then Apple MPS, then CPU on its own.

## Assets

- Provided 512x512 textures in `assets/provided/`:
  - `gradient.png`: smooth, low-frequency
  - `bricks.png`: structured, sharp edges
  - `clouds.png`: fractal, multi-scale detail
- The CC0 textures I sourced from ambientCG live in `assets/own/` (credited in `assets/own/CREDITS.md`). Re-fetch them with `python scripts/fetch_own_textures.py --execute`.
- Any image works, see below.

## Run on your own image

- One command fits S3TC plus the three neural sizes and saves the comparison images:
  ```bash
  python scripts/run_custom_texture.py --image path/to/your.png
  ```
  It center-crops to a square, resizes to 512 (`--size-px`), and writes to `experiments/custom/`:
  - `<name>_comparison.png`: original, S3TC, neural small/medium/large side by side
  - `<name>_zoom.png`: 4x center crop
  - `<name>_training.png`: PSNR over training
  - `<name>_metrics.json`: PSNR, size, and compression per representation
  - Options: `--sizes small large`, `--steps 2000`, `--seeds 3`, `--no-s3tc`, `--out DIR`, `--name STEM`.
- Just the S3TC baseline: `python scripts/run_s3tc.py --image path/to/your.png`.
- Just one neural size: `python scripts/run_train.py --image path/to/your.png --size medium`.

## Project Layout

- `ntc/` (library): sampler, S3TC (plus seam-aware variants), feature grid, MLP, model, train, quantize, metrics
- `scripts/`: runnable entry points + experiment drivers
- `experiments/`: generated reconstructions, plots, metrics (not committed)
- `writeup/`: progression notes and the assignment writeup (`progression.html`/`.pdf`, `submission.html`/`.pdf`)

## Reproducing results

- Environment check: `python scripts/check_env.py`
- Bilinear sampler: `python scripts/run_sampler.py` (round-trip PSNR = inf, exact)
- S3TC baseline: `python scripts/run_s3tc.py` (fixed 6:1)
  - bricks 41.4 dB, clouds 42.6 dB, gradient 42.9 dB (128 KB each vs 768 KB raw)
- Seam metric + seam-aware S3TC: `python scripts/run_s3tc_optimizations.py`
- Hyperparameter sweeps: `python scripts/sweep_s3tc_optimizations.py`
- Feature grid: `python scripts/run_feature_grid.py`
- Decoder MLP + combined sizes: `python scripts/run_color_mlp.py`
- Train a neural texture: `python scripts/run_train.py --texture gradient --size small`
- Full size-vs-quality comparison (includes post-training 8-bit quantization): `python scripts/run_results.py` (`--plot-only` to rebuild the figures without retraining)
- Fetch the sourced (CC0) textures: `python scripts/fetch_own_textures.py --execute`
- Sourced-texture comparison: `python scripts/run_own_textures.py`
- Writeup figures: `python scripts/make_figures.py`
- Custom image: `python scripts/run_custom_texture.py --image YOUR_IMAGE`
- Tests: `python tests/test_io.py`, `python tests/test_sampler_s3tc.py`, `python tests/test_s3tc_optimizations.py`, `python tests/test_feature_grid.py`, `python tests/test_color_mlp.py`, `python tests/test_train.py`, `python tests/test_quantize.py`

## Outputs

- Every script writes images, plots, and metrics under `experiments/` (not committed): `experiments/custom/`, `experiments/results/`, `experiments/s3tc/`, `experiments/train/`, `experiments/own/`.

## Writeups

- Progression notes: [HTML](writeup/progression.html) · [PDF](writeup/progression.pdf)
- Assignment writeup: [PDF](writeup/submission.pdf) ([source](writeup/submission.html))
- Rebuild the PDFs: `pip install -r requirements-docs.txt && python scripts/build_writeup.py`
