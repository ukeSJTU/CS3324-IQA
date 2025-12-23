#!/bin/bash
# Train HyperIQA model on KonIQ dataset

# Parse command line arguments with defaults
LR="${1:-2e-5}"
BATCH_SIZE="${2:-96}"
EPOCHS="${3:-16}"
VAL_SPLIT="${4:-0.1}"
SEED="${5:-42}"

# Run training from hyperIQA directory
cd "$(dirname "$0")/.."

python train.py \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --val_split "$VAL_SPLIT" \
    --seed "$SEED"
