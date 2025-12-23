#!/bin/bash
# Evaluate HyperIQA model on test datasets

# Parse arguments
# Usage: ./evaluate.sh [CHECKPOINT_FOLDER or MODEL_PATH] [DATASETS] [PATCH_NUM]

INPUT_PATH="${1:-../checkpoints/latest/best_model.pkl}"
DATASETS="${2:-all}"
PATCH_NUM="${3:-25}"

# Run evaluation from hyperIQA directory
cd "$(dirname "$0")/.."

# Check if input is a folder or file
if [ -d "$INPUT_PATH" ]; then
    # It's a folder, use checkpoint_folder argument
    python evaluate.py \
        --checkpoint_folder "$INPUT_PATH" \
        --datasets $DATASETS \
        --patch_num "$PATCH_NUM"
else
    # It's a file path, use model_path argument
    python evaluate.py \
        --model_path "$INPUT_PATH" \
        --datasets $DATASETS \
        --patch_num "$PATCH_NUM"
fi
