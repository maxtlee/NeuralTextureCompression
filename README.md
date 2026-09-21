# NeuralTextureCompression

Implementing a neural solution for texture compression and lookup (i.e. training memory-efficient mappings from `(u, v)` coordinate in → `(r, g, b)` color out)

- One neural network will be generated per texture
- Instead of storing every texel, fit each texture with a small multi-resolution feature grid plus a shallow decoder MLP.
- Results (size and PSNR) will be compared against classic S3TC/DXT1 block compression.

Created as part of 15-474/674 (Neural Graphics) Assignment 1

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

- Note: the GPU I'm using is an old ass GTX 1050 Ti (Pascal, sm_61), so I use the CUDA 12.6 PyTorch build in `requirements.txt` (newer CUDA 13 builds dropped Pascal support). On different hardware, install whatever PyTorch build works on your own GPU. 

## Assets

- Provided 512x512 textures in `assets/provided/`:
  - `gradient.png`: smooth, low-frequency
  - `bricks.png`: structured, sharp edges
  - `clouds.png`: fractal, multi-scale detail
- Other textures will go in `assets/own/`.

## Project Layout

- `ntc/` (library): sampler, S3TC (plus seam-aware variants), feature grid, MLP, model, train, quantize, metrics
- `scripts/`: runnable entry points + experiment drivers
- `experiments/`: generated reconstructions, plots, metrics (not committed)
- `writeup/`: running progression notes ([progression.html](writeup/progression.html), [progression.pdf](writeup/progression.pdf)) + final report source

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
- Full size-vs-quality comparison: `python scripts/run_results.py` (add `--plot-only` to rebuild the figures without retraining)
- Tests: `python tests/test_sampler_s3tc.py`, `python tests/test_s3tc_optimizations.py`, `python tests/test_feature_grid.py`, `python tests/test_color_mlp.py`, and `python tests/test_train.py`
- Figures (writeup): `python scripts/make_figures.py`
- Progression notes: [HTML](writeup/progression.html) · [PDF](writeup/progression.pdf)
- Rebuild the PDF: `pip install -r requirements-docs.txt && python scripts/build_writeup.py`
- Outputs in `experiments/` (not committed).
