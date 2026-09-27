#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: scripts/run_umia.sh CONFIG ROLE [SEED] [DATA_DIR] [RUN_DIR]}
ROLE=${2:?missing role}
SEED=${3:-0}
DATA_DIR=${4:-./data}
RUN_DIR=${5:-./runs}

shape-forgetting u-mia \
  --config "$CONFIG" \
  --seed "$SEED" \
  --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --role "$ROLE"

