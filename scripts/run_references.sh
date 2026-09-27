#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: scripts/run_references.sh CONFIG [SEED] [DATA_DIR] [RUN_DIR]}
SEED=${2:-0}
DATA_DIR=${3:-./data}
RUN_DIR=${4:-./runs}

shape-forgetting make-split --config "$CONFIG" --seed "$SEED" --data-dir "$DATA_DIR" --run-dir "$RUN_DIR"
shape-forgetting train --config "$CONFIG" --seed "$SEED" --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role original
shape-forgetting train --config "$CONFIG" --seed "$SEED" --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --role retrained

