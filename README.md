# Explore Image Analysis

Minimal playground for experimenting with computer vision models, starting with an MNIST CNN baseline.

## Setup
- Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).
- Install dependencies into a virtual env: `uv sync`

## Run
- Train/test the MNIST CNN: `uv run python mnist_basic_cnn.py`
- Visualize a few predictions: `uv run python mnist_basic_cnn.py --visualize`

## Docker (GPU)
- Build: `docker build -t explore-image-analysis:latest .`
- Run (NVIDIA): `docker run --rm --gpus all -v "$PWD/.cache:/app/.cache" explore-image-analysis:latest`

Notes:
- Uses GPU if available (CUDA ➜ MPS ➜ CPU fallback).
- MNIST data is cached under `./.cache/mnist`.
