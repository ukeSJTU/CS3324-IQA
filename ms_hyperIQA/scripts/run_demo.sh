#!/bin/bash

# MS-HyperIQA Demo Script
# Usage: ./scripts/run_demo.sh [IMAGE_PATH] [MODEL_PATH] [NUM_PATCHES]

# Default values
IMAGE_PATH=${1:-"../data/D_01.jpg"}
MODEL_PATH=${2:-"../pretrained/ms_hyperIQA_koniq.pkl"}
NUM_PATCHES=${3:-10}

echo "========================================="
echo "MS-HyperIQA Demo - Single Image Quality Assessment"
echo "========================================="
echo "Image: $IMAGE_PATH"
echo "Model: $MODEL_PATH"
echo "Patches: $NUM_PATCHES"
echo "========================================="

python demo.py \
    --image_path "$IMAGE_PATH" \
    --model_path "$MODEL_PATH" \
    --num_patches $NUM_PATCHES

echo "========================================="
