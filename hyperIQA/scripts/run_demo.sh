#!/bin/bash
# Run HyperIQA demo for single image quality assessment

# Default values
IMAGE_PATH="${1:-./data/D_01.jpg}"
MODEL_PATH="${2:-./pretrained/koniq_pretrained.pkl}"
NUM_PATCHES="${3:-10}"

# Run demo
python demo.py \
    --image_path "$IMAGE_PATH" \
    --model_path "$MODEL_PATH" \
    --num_patches "$NUM_PATCHES"