# Minecraft Terrain Repair

Minecraft terrain export, visualization, and deterministic U-Net repair for masked terrain infill.

## Setup

```bash
make sync
```

## Commands

```bash
make test
make export WORLD=/path/to/World
make visualize
make train EPOCHS=10
make prepare-infer
make infer
make repair
```

`make export` accepts either a Minecraft save root or a direct overworld directory.

`make train` trains the deterministic repair U-Net. `make prepare-infer` creates scratch repair inputs, and `make infer` repairs those inputs.

Shared validation cases live in `repair_cases/`. `make repair` runs every case, and `make repair-case CASE=Name` runs a single case.

Training supports CUDA, AMP, channels-last tensors, gradient accumulation, and TensorBoard logging.

## References

- [InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation](https://arxiv.org/abs/2512.08309)
- [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)
- [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)
- [Resolution-Robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
