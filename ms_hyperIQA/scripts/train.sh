#!/bin/bash

# MS-HyperIQA Training Script
# Usage: ./scripts/train.sh [LR] [BATCH_SIZE] [EPOCHS] [VAL_SPLIT] [SEED]

# Default values
LR=${1:-2e-5}
BATCH_SIZE=${2:-96}
EPOCHS=${3:-30}
VAL_SPLIT=${4:-0.1}
SEED=${5:-42}

echo "========================================="
echo "Training MS-HyperIQA"
echo "========================================="
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Validation Split: $VAL_SPLIT"
echo "Random Seed: $SEED"
echo "========================================="

python train.py \
    --dataset_root ../datasets/ \
    --train_json ../datasets/metas/koniq_train.json \
    --lr $LR \
    --lr_ratio 10 \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --val_split $VAL_SPLIT \
    --seed $SEED \
    --loss_type combined \
    --rank_loss_weight 0.3 \
    --lr_schedule cosine \
    --save_all_epochs

echo "========================================="
echo "Training completed!"
echo "Check ../checkpoints/ for results"
echo "========================================="
