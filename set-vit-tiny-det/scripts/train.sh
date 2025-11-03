#!/bin/bash

# Training Script for Tiny Object Detection
# Usage: bash scripts/train.sh

echo "=========================================="
echo "Starting Training Pipeline"
echo "=========================================="

# Configuration
CONFIG="configs/default.yaml"
TRAIN_DATA="data/train"
TRAIN_ANNOT="data/train_annotations.txt"
VAL_DATA="data/val"
VAL_ANNOT="data/val_annotations.txt"
EPOCHS=50

echo "Config: $CONFIG"
echo "Training data: $TRAIN_DATA"
echo "Epochs: $EPOCHS"
echo ""

# Run training
python train.py \
    --config "$CONFIG" \
    --train-data "$TRAIN_DATA" \
    --train-annot "$TRAIN_ANNOT" \
    --val-data "$VAL_DATA" \
    --val-annot "$VAL_ANNOT" \
    --epochs "$EPOCHS"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="