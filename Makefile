PYTEST ?= uv run pytest
WORLD ?= /path/to/World
OUT ?= ./data/chunks
RENDERS ?= ./data/chunks/renders
CHECKPOINT ?= ./artifacts/diffusion.pt
RESUME ?=
SAVE_EVERY ?= 1
INPUTS ?= ./inputs
KNOWN_HEIGHT ?= $(INPUTS)/known_height.npy
KNOWN_MATERIAL ?= $(INPUTS)/known_material.npy
MASK ?= $(INPUTS)/mask.npy
OUTPUTS ?= ./outputs
SAMPLE_COUNT ?= 5
SEED ?= 7
LIMIT ?=
WORKERS ?=
EXPORT_SEED ?=
EPOCHS ?= 1
BATCH_SIZE ?= 2
LEARNING_RATE ?= 1e-4
TRAIN_TILE_SIZE ?= 128
STRIDE_CHUNKS ?= 1
INFER_TILE_SIZE ?= 128
OVERLAP ?= 32
NUM_STEPS ?=
ORIGIN_CHUNK_X ?=
ORIGIN_CHUNK_Z ?=
MASK_TOP ?= 48
MASK_LEFT ?= 48
MASK_HEIGHT ?= 32
MASK_WIDTH ?= 32

.PHONY: help sync test export visualize train prepare-infer infer infer-gui

help:
	@printf "Targets:\n"
	@printf "  make sync                                      Install workspace dependencies\n"
	@printf "  make test                                      Run exporter and diffusion tests\n"
	@printf "  make export WORLD=... [OUT=...] [LIMIT=N] [WORKERS=N] [EXPORT_SEED=N]\n"
	@printf "  make visualize [OUT=...]                       Render export validation images\n"
	@printf "  make train [EPOCHS=...] [BATCH_SIZE=...] [RESUME=...] [SAVE_EVERY=N]\n"
	@printf "  make prepare-infer [ORIGIN_CHUNK_X=...]        Build known_height/material/mask from exported chunks\n"
	@printf "  make infer [CHECKPOINT=...] [INPUTS=...]       Run tiled diffusion inference\n"
	@printf "  make infer-gui [CHECKPOINT=...]                Pick a chunk region in a local GUI and run regeneration\n"

sync:
	uv sync --all-packages --all-extras

test:
	$(PYTEST) packages/exporter/tests packages/diffusion/tests

export:
	uv run --package mc-terrain-exporter python scripts/run_export.py --world "$(WORLD)" --out "$(OUT)" $(if $(LIMIT),--limit $(LIMIT),) $(if $(EXPORT_SEED),--seed $(EXPORT_SEED),) $(if $(WORKERS),--workers $(WORKERS),)

visualize:
	uv run --package mc-terrain-exporter python scripts/visualize_export.py --export-dir "$(OUT)" --out-dir "$(RENDERS)" --sample-count $(SAMPLE_COUNT) --seed $(SEED)

train:
	uv run --package mc-terrain-diffusion python -m diffusion.training --export-dir "$(OUT)" --checkpoint "$(CHECKPOINT)" --epochs $(EPOCHS) --save-every $(SAVE_EVERY) --batch-size $(BATCH_SIZE) --learning-rate $(LEARNING_RATE) --tile-size $(TRAIN_TILE_SIZE) --stride-chunks $(STRIDE_CHUNKS) $(if $(RESUME),--resume "$(RESUME)",)

prepare-infer:
	uv run --package mc-terrain-diffusion python scripts/prepare_infer_inputs.py --export-dir "$(OUT)" --out-dir "$(INPUTS)" --checkpoint "$(CHECKPOINT)" --tile-size $(INFER_TILE_SIZE) $(if $(ORIGIN_CHUNK_X),--origin-chunk-x $(ORIGIN_CHUNK_X),) $(if $(ORIGIN_CHUNK_Z),--origin-chunk-z $(ORIGIN_CHUNK_Z),) --mask-top $(MASK_TOP) --mask-left $(MASK_LEFT) --mask-height $(MASK_HEIGHT) --mask-width $(MASK_WIDTH)

infer:
	uv run --package mc-terrain-diffusion python -m diffusion.inference --checkpoint "$(CHECKPOINT)" --known-height "$(KNOWN_HEIGHT)" --known-material "$(KNOWN_MATERIAL)" --mask "$(MASK)" --out-dir "$(OUTPUTS)" --tile-size $(INFER_TILE_SIZE) --overlap $(OVERLAP) $(if $(NUM_STEPS),--num-steps $(NUM_STEPS),)

infer-gui:
	uv run --package mc-terrain-diffusion python scripts/infer_gui.py --export-dir "$(OUT)" --checkpoint "$(CHECKPOINT)" --inputs-dir "$(INPUTS)" --out-dir "$(OUTPUTS)" --tile-size $(INFER_TILE_SIZE) --overlap $(OVERLAP) $(if $(NUM_STEPS),--num-steps $(NUM_STEPS),)
