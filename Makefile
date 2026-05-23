PYTEST ?= uv run pytest
WORLD ?= /path/to/World
OUT ?= ./data/chunks
RENDERS ?= ./data/chunks/renders
REPAIR_CHECKPOINT ?= ./artifacts/repair.pt
REPAIR_LATEST_CHECKPOINT ?= ./artifacts/repair_latest.pt
REPAIR_BEST_CHECKPOINT ?= ./artifacts/repair_best.pt
RESUME ?=
SAVE_EVERY ?= 1
INPUTS ?= ./inputs
KNOWN_HEIGHT ?= $(INPUTS)/known_height.npy
KNOWN_MATERIAL ?= $(INPUTS)/known_material.npy
MASK ?= $(INPUTS)/mask.npy
OUTPUTS ?= ./outputs
SAVED_CASES ?= $(OUTPUTS)/saved_cases
FIGURES ?= $(OUTPUTS)/figures
REPAIR_CASES ?= ./repair_cases
CASE ?=
SAMPLE_COUNT ?= 5
SEED ?= 7
VARIANCE_OUT ?= ./artifacts/variance_analysis.json
VARIANCE_CSV ?= ./artifacts/variance_windows.csv
VARIANCE_LIMIT ?=
LIMIT ?=
WORKERS ?=
EXPORT_SEED ?=
EPOCHS ?= 1
BATCH_SIZE ?= 2
LEARNING_RATE ?= 1e-4
LR_SCHEDULER ?= none
MODEL_BASE_CHANNELS ?= 64
MODEL_DEPTH ?= 4
MODEL_BOTTLENECK_DILATIONS ?= 1,2,4,2
TRAIN_TILE_SIZE ?= 128
STRIDE_CHUNKS ?= 1
REPAIR_MASK_MODE ?= selection_mixed
AMP ?= auto
DEVICE ?= auto
NUM_WORKERS ?= 0
GRAD_CLIP_NORM ?= 1.0
GRAD_ACCUM_STEPS ?= 1
VALIDATE_EVERY ?= 1
TENSORBOARD_DIR ?= ./runs/repair
TENSORBOARD ?=
COMPILE ?=
CHANNELS_LAST ?=
TF32 ?= auto
LIGHTNING_ACCELERATOR ?= auto
LIGHTNING_DEVICES ?= 1
LIGHTNING_STRATEGY ?= auto
LIGHTNING_PRECISION ?=
LIGHTNING_ROOT ?= ./artifacts/lightning
LIGHTNING_CKPT ?=
LIGHTNING_NUM_WORKERS ?= -1
LIGHTNING_PREFETCH_FACTOR ?= 4
PREFILL_ITERATIONS ?= 64
LITLOGGER_NAME ?=
LITLOGGER_TEAMSPACE ?=
LITLOGGER_ROOT ?=
LITLOGGER_METADATA ?=
LITLOGGER_LOG_MODEL ?=
LITLOGGER_NO_SAVE_LOGS ?=
INFER_TILE_SIZE ?= 128
ORIGIN_CHUNK_X ?=
ORIGIN_CHUNK_Z ?=
MASK_TOP ?= 48
MASK_LEFT ?= 48
MASK_HEIGHT ?= 32
MASK_WIDTH ?= 32

.PHONY: help sync test export visualize analyze-variance train train-lightning train-legacy prepare-infer infer repair repair-case infer-gui view-repair generate-figures

help:
	@printf "Targets:\n"
	@printf "  make sync                                      Install workspace dependencies\n"
	@printf "  make test                                      Run exporter and repair tests\n"
	@printf "  make export WORLD=... [OUT=...] [LIMIT=N] [WORKERS=N] [EXPORT_SEED=N]\n"
	@printf "  make visualize [OUT=./data/exports]            Render export validation images\n"
	@printf "  make analyze-variance [OUT=...]                Report height/material variance balance\n"
	@printf "  make train [OUT=...] [EPOCHS=...] [BATCH_SIZE=...] [LIGHTNING_DEVICES=1] [AMP=auto|off|fp16|bf16]\n"
	@printf "            PyTorch Lightning + LitLogger; optional: [LITLOGGER_NAME=...] [LITLOGGER_TEAMSPACE=user/ts]\n"
	@printf "            [LITLOGGER_ROOT=path] [LITLOGGER_METADATA='k=v'] [LITLOGGER_LOG_MODEL=1]\n"
	@printf "  make train-legacy …                            Plain PyTorch loop + optional TensorBoard\n"
	@printf "  make prepare-infer [ORIGIN_CHUNK_X=...]        Build known_height/material/mask from exported chunks\n"
	@printf "  make infer [REPAIR_CHECKPOINT=...] [INPUTS=...] Run U-Net repair on prepared scratch inputs\n"
	@printf "  make repair [REPAIR_CHECKPOINT=...] [REPAIR_CASES=...] Run shared deterministic repair cases\n"
	@printf "  make repair-case CASE=name                     Run deterministic repair on REPAIR_CASES/name\n"
	@printf "  make infer-gui [REPAIR_CHECKPOINT=...]         Pick a chunk region in a local GUI and run U-Net repair\n"
	@printf "  make view-repair [SAVED_CASES=...]             Open 3D repair output viewer (pygame)\n"
	@printf "  make generate-figures [SAVED_CASES=...]        Export isometric PNG figures\n"

sync:
	uv sync --all-packages --all-extras

test:
	$(PYTEST) packages/exporter/tests packages/unet/tests

export:
	uv run --package mc-terrain-exporter python scripts/run_export.py --world "$(WORLD)" --out "$(OUT)" $(if $(LIMIT),--limit $(LIMIT),) $(if $(EXPORT_SEED),--seed $(EXPORT_SEED),) $(if $(WORKERS),--workers $(WORKERS),)

visualize:
	uv run --package mc-terrain-exporter python scripts/visualize_export.py --export-dir "$(OUT)" --out-dir "$(RENDERS)" --sample-count $(SAMPLE_COUNT) --seed $(SEED)

analyze-variance:
	uv run --package mc-terrain-unet python scripts/analyze_variance.py \
		--export-dir "$(OUT)" \
		--tile-size $(TRAIN_TILE_SIZE) \
		--stride-chunks $(STRIDE_CHUNKS) \
		--seed $(SEED) \
		--out "$(VARIANCE_OUT)" \
		--csv-out "$(VARIANCE_CSV)" \
		$(if $(VARIANCE_LIMIT),--limit-windows $(VARIANCE_LIMIT),)

train:
	uv run --package mc-terrain-unet train-terrain-repair \
		--export-dir "$(OUT)" \
		--checkpoint "$(REPAIR_CHECKPOINT)" \
		--latest-checkpoint "$(REPAIR_LATEST_CHECKPOINT)" \
		--best-checkpoint "$(REPAIR_BEST_CHECKPOINT)" \
		--epochs $(EPOCHS) \
		--save-every $(SAVE_EVERY) \
		--batch-size $(BATCH_SIZE) \
		--grad-accum-steps $(GRAD_ACCUM_STEPS) \
		--learning-rate $(LEARNING_RATE) \
		--lr-scheduler "$(LR_SCHEDULER)" \
		--model-base-channels $(MODEL_BASE_CHANNELS) \
		--model-depth $(MODEL_DEPTH) \
		--model-bottleneck-dilations "$(MODEL_BOTTLENECK_DILATIONS)" \
		--tile-size $(TRAIN_TILE_SIZE) \
		--stride-chunks $(STRIDE_CHUNKS) \
		--mask-mode "$(REPAIR_MASK_MODE)" \
		--amp "$(AMP)" \
		--accelerator "$(LIGHTNING_ACCELERATOR)" \
		--devices "$(LIGHTNING_DEVICES)" \
		--strategy "$(LIGHTNING_STRATEGY)" \
		--num-workers $(LIGHTNING_NUM_WORKERS) \
		--prefetch-factor $(LIGHTNING_PREFETCH_FACTOR) \
		--prefill-iterations $(PREFILL_ITERATIONS) \
		--grad-clip-norm $(GRAD_CLIP_NORM) \
		--validation-cases-dir "$(REPAIR_CASES)" \
		--validate-every $(VALIDATE_EVERY) \
		--tf32 "$(TF32)" \
		--lightning-root-dir "$(LIGHTNING_ROOT)" \
		$(if $(LIGHTNING_PRECISION),--precision "$(LIGHTNING_PRECISION)",) \
		$(if $(LITLOGGER_NAME),--litlogger-name "$(LITLOGGER_NAME)",) \
		$(if $(LITLOGGER_TEAMSPACE),--litlogger-teamspace "$(LITLOGGER_TEAMSPACE)",) \
		$(if $(LITLOGGER_ROOT),--litlogger-root-dir "$(LITLOGGER_ROOT)",) \
		$(foreach kv,$(LITLOGGER_METADATA),--litlogger-metadata "$(kv)") \
		$(if $(LITLOGGER_LOG_MODEL),--litlogger-log-model,) \
		$(if $(LITLOGGER_NO_SAVE_LOGS),--no-litlogger-save-logs,) \
		$(if $(COMPILE),--compile,) \
		$(if $(CHANNELS_LAST),--channels-last,) \
		$(if $(RESUME),--resume "$(RESUME)",) \
		$(if $(LIGHTNING_CKPT),--ckpt-path "$(LIGHTNING_CKPT)",)

train-lightning: train

train-legacy:
	uv run --package mc-terrain-unet python -m unet.repair_training --export-dir "$(OUT)" --checkpoint "$(REPAIR_CHECKPOINT)" --latest-checkpoint "$(REPAIR_LATEST_CHECKPOINT)" --best-checkpoint "$(REPAIR_BEST_CHECKPOINT)" --epochs $(EPOCHS) --save-every $(SAVE_EVERY) --batch-size $(BATCH_SIZE) --grad-accum-steps $(GRAD_ACCUM_STEPS) --learning-rate $(LEARNING_RATE) --model-base-channels $(MODEL_BASE_CHANNELS) --model-depth $(MODEL_DEPTH) --model-bottleneck-dilations "$(MODEL_BOTTLENECK_DILATIONS)" --tile-size $(TRAIN_TILE_SIZE) --stride-chunks $(STRIDE_CHUNKS) --mask-mode "$(REPAIR_MASK_MODE)" --device "$(DEVICE)" --amp "$(AMP)" --num-workers $(NUM_WORKERS) --grad-clip-norm $(GRAD_CLIP_NORM) --validation-cases-dir "$(REPAIR_CASES)" --validate-every $(VALIDATE_EVERY) --tf32 "$(TF32)" $(if $(TENSORBOARD),--tensorboard-dir "$(TENSORBOARD_DIR)",) $(if $(COMPILE),--compile,) $(if $(CHANNELS_LAST),--channels-last,) $(if $(RESUME),--resume "$(RESUME)",)

prepare-infer:
	uv run --package mc-terrain-unet python scripts/prepare_infer_inputs.py --export-dir "$(OUT)" --out-dir "$(INPUTS)" --checkpoint "$(REPAIR_CHECKPOINT)" --tile-size $(INFER_TILE_SIZE) $(if $(ORIGIN_CHUNK_X),--origin-chunk-x $(ORIGIN_CHUNK_X),) $(if $(ORIGIN_CHUNK_Z),--origin-chunk-z $(ORIGIN_CHUNK_Z),) --mask-top $(MASK_TOP) --mask-left $(MASK_LEFT) --mask-height $(MASK_HEIGHT) --mask-width $(MASK_WIDTH)

infer:
	uv run --package mc-terrain-unet python -m unet.repair_inference --checkpoint "$(REPAIR_CHECKPOINT)" --known-height "$(KNOWN_HEIGHT)" --known-material "$(KNOWN_MATERIAL)" --mask "$(MASK)" --out-dir "$(OUTPUTS)" --known-support "$(INPUTS)/known_support.npy"

repair:
	uv run --package mc-terrain-unet python -m unet.repair_inference --checkpoint "$(REPAIR_CHECKPOINT)" --skip-current --saved-cases-dir "$(REPAIR_CASES)" --saved-cases-out-dir "$(OUTPUTS)/saved_cases"

repair-case:
	uv run --package mc-terrain-unet python -m unet.repair_inference --checkpoint "$(REPAIR_CHECKPOINT)" --known-height "$(REPAIR_CASES)/$(CASE)/known_height.npy" --known-material "$(REPAIR_CASES)/$(CASE)/known_material.npy" --mask "$(REPAIR_CASES)/$(CASE)/mask.npy" --out-dir "$(OUTPUTS)/saved_cases/$(CASE)" --known-support "$(REPAIR_CASES)/$(CASE)/known_support.npy"

infer-gui:
	uv run --package mc-terrain-unet python scripts/infer_gui.py --export-dir "$(OUT)" --checkpoint "$(REPAIR_CHECKPOINT)" --inputs-dir "$(INPUTS)" --repair-cases-dir "$(REPAIR_CASES)" --out-dir "$(OUTPUTS)" --tile-size $(INFER_TILE_SIZE)

view-repair:
	uv run --package mc-terrain-render repair-3d --cases-dir "$(SAVED_CASES)" --repair-cases-dir "$(REPAIR_CASES)"

generate-figures:
	uv run --package mc-terrain-render generate-figures --cases-dir "$(SAVED_CASES)" --repair-cases-dir "$(REPAIR_CASES)" --out-dir "$(FIGURES)" $(if $(LIMIT),--limit $(LIMIT),)
