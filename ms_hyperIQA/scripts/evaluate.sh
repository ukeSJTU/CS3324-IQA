#!/bin/bash

# MS-HyperIQA Evaluation Script
# Usage: ./scripts/evaluate.sh [CHECKPOINT_FOLDER or MODEL_PATH] [DATASETS] [PATCH_NUM]

# Default values
CHECKPOINT=${1:-"../checkpoints/latest/"}
DATASETS=${2:-"all"}
PATCH_NUM=${3:-25}

echo "========================================="
echo "Evaluating MS-HyperIQA"
echo "========================================="
echo "Checkpoint/Model: $CHECKPOINT"
echo "Datasets: $DATASETS"
echo "Patch Number: $PATCH_NUM"
echo "========================================="

# Check if input is a directory (checkpoint folder) or file (model path)
if [ -d "$CHECKPOINT" ]; then
    # It's a directory, use checkpoint_folder argument
    python evaluate.py \
        --checkpoint_folder "$CHECKPOINT" \
        --dataset_root ../datasets/ \
        --datasets $DATASETS \
        --patch_num $PATCH_NUM
elif [ -f "$CHECKPOINT" ]; then
    # It's a file, use model_path argument
    python evaluate.py \
        --model_path "$CHECKPOINT" \
        --dataset_root ../datasets/ \
        --datasets $DATASETS \
        --patch_num $PATCH_NUM
else
    echo "Error: Checkpoint path '$CHECKPOINT' not found!"
    exit 1
fi

echo "========================================="
echo "Evaluation completed!"
echo "Check eval_results.json in checkpoint folder"
echo "========================================="
