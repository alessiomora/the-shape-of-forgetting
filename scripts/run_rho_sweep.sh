#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: scripts/run_rho_sweep.sh CONFIG EPOCHS LR [WD] [SEED] [DATA_DIR] [RUN_DIR] [RHO_GRID]}
EPOCHS=${2:?missing epochs}
LR=${3:?missing lr}
WD=${4:-0}
SEED=${5:-0}
DATA_DIR=${6:-./data}
RUN_DIR=${7:-./runs}
RHO_GRID=${8:-0:1:0.05}

shape-forgetting rho-sweep \
  --config "$CONFIG" \
  --seed "$SEED" \
  --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --rho-grid "$RHO_GRID" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --wd "$WD" \
  --no-save-checkpoints

