# Minecraft Terrain Diffusion

Minecraft terrain export and diffusion for Vanilla-style infill.

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
make train-repair
make repair
```

`make export` accepts either a Minecraft save root or a direct overworld directory.

References: [Terrain Diffusion / InfiniteDiffusion](https://arxiv.org/abs/2512.08309), [MultiDiffusion](https://arxiv.org/abs/2302.08113)
