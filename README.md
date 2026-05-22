# Minecraft Terrain Repair

This repo exports surface-aligned Minecraft terrain tiles, renders them for inspection, and trains a deterministic U-Net to repair masked regions so they settle back into vanilla-like terrain.

The practical use case is terrain cleanup after large edits. If you clear out space for a base, castle, megaproject, rail line, or other large build, the surrounding land often ends up with harsh cut lines that no longer look like naturally generated Minecraft. The repair model is meant to fill those damaged regions back in with terrain that blends into the nearby world instead of looking hand-patched.

Structure:

- `packages/exporter`: reads Minecraft regions and writes `surface_*.npy` and `chunk_*.npy`
- `packages/unet`: dataset assembly, training, inference, and validation
- `packages/render`: lightweight 3D viewer for saved repair outputs
- `scripts/`: thin CLI wrappers around the workspace packages

## Requirements

- Python 3.11 to 3.14
- [`uv`](https://docs.astral.sh/uv/)
- A local Minecraft world if you want to export your own data

If you are generating your own worlds for training or evaluation, disable structures so villages, temples, and other generated builds do not get mixed into the terrain data.

## Setup

Install the workspace and all package extras:

```bash
make sync
```

Run the test suite once to make sure the environment is healthy:

```bash
make test
```

## What It Is For

This is a terrain infill tool for local reconstruction. The main use case is clearing space for a large build, then repairing the surrounding terrain so it blends back into the world in a vanilla-like way.

## Typical Workflow

### 1. Export terrain

Point `WORLD` at either a save root or the overworld directory itself:

```bash
make export WORLD=/path/to/World
```

```bash
make export WORLD=/path/to/World OUT=./data/chunks LIMIT=512 WORKERS=4
```

The exporter writes chunk-aligned `.npy` arrays into `OUT`:

- `surface_x_z.npy`: 16x16 surface heights
- `chunk_x_z.npy`: 16x16x40 surface-anchored material slabs

Surface keep the terrain surface as a compact representation reduced to our vocabulary of allowed surface block classe. Blocks outside that list are represented as air.

The `chunk_x_z.npy` file stores a shallow vertical strip around that surface anchor for each `(x, z)` chunk. We use that depth information mainly to derive surface material and a simple support signal.

### 2. Visualize the export

Before training, it is worth checking whether the export actually looks sane:

```bash
make visualize OUT=./data/chunks
```

This generates stitched maps and sample previews under the render output directory.

### 3. Train the repair U-Net

The main training path uses PyTorch Lightning:

```bash
make train OUT=./data/chunks EPOCHS=10 BATCH_SIZE=2
```

Common knobs:

```bash
make train \
  OUT=./data/chunks \
  EPOCHS=25 \
  BATCH_SIZE=4 \
  LIGHTNING_DEVICES=1 \
  AMP=auto \
  MODEL_BASE_CHANNELS=64 \
  MODEL_DEPTH=4 \
  TRAIN_TILE_SIZE=128
```

Our training was done on exported sections from multiple worlds and multiple parts of those worlds to expose the model to a broad mix of terrain shapes and materials: coastlines, rivers, hillsides, beaches, snow and ice transitions, flatter plains, and rougher elevation changes.

Artifacts land in:

- `artifacts/repair.pt`: latest compatible checkpoint
- `artifacts/repair_latest.pt`: rolling latest snapshot
- `artifacts/repair_best.pt`: best validation checkpoint
- `artifacts/lightning/`: Lightning logs and checkpoint files

Validation runs against the shared saved cases in `repair_cases/` during training. If you need the older plain PyTorch loop, `make train-legacy` is still available.

## Inference And Evaluation

Prepare a scratch inference window from exported terrain:

```bash
make prepare-infer OUT=./data/chunks
make infer
```

Run the shared regression-style repair cases:

```bash
make repair
make repair-case CASE=Beach
```

Open the local selector GUI for choosing a repair region interactively:

```bash
make infer-gui OUT=./data/chunks
```

View saved repair outputs in the 3D viewer:

```bash
make view-repair
```

## Common Commands

```bash
make help
make test
make analyze-variance OUT=./data/chunks
make train-lightning
make train-legacy
make repair
```

## References

- [InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation](https://arxiv.org/abs/2512.08309)
- [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)
- [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)
- [Resolution-Robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
