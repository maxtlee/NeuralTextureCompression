# NeuralTextureCompression

15-474/674 (Neural Graphics), Assignment 1.

Implementing a neural solution for texture compression and lookup (i.e. training memory-efficient mappings from `(u, v)` coordinate in → `(r, g, b)` color out)

- One neural network will be generated per texture
- Instead of storing every texel, fit each texture with a small multi-resolution feature grid plus a shallow decoder MLP.
- Results (size and PSNR) will be compared against classic S3TC/DXT1 block compression.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

- My computer's GPU is a GTX 1050 Ti (Pascal, sm_61), so I use the CUDA 12.6 PyTorch build in `requirements.txt` (newer CUDA 13 builds dropped Pascal support). On different hardware, install whatever PyTorch build works on your own GPU. 

## Assets

- Provided 512x512 textures in `assets/provided/`:
  - `gradient.png`: smooth, low-frequency
  - `bricks.png`: structured, sharp edges
  - `clouds.png`: fractal, multi-scale detail
- Other textures will go in `assets/own/`.

## Project Layout

- `ntc/` (library): sampler, S3TC, feature grid, MLP, model, train, quantize, metrics
- `scripts/`: entry points for each problem + experiment drivers
- `experiments/`: generated reconstructions, plots, metrics (not committed)
- `writeup/`: report source + PDF

## Reproducing results

- I'll list exact run commands and outputs here as I implement the project.
