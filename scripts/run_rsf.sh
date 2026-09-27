#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: scripts/run_rsf.sh CONFIG RHO EPOCHS LR [WD] [SEED] [DATA_DIR] [RUN_DIR]}
RHO=${2:?missing rho}
EPOCHS=${3:?missing epochs}
LR=${4:?missing lr}
WD=${5:-0}
SEED=${6:-0}
DATA_DIR=${7:-./data}
RUN_DIR=${8:-./runs}

shape-forgetting rsf \
  --config "$CONFIG" \
  --seed "$SEED" \
  --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --rho "$RHO" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --wd "$WD"

