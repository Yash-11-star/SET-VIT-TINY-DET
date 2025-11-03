#!/bin/bash

# Evaluation Script for Tiny Object Detection
# Usage: bash scripts/eval.sh

echo "=========================================="
echo "Starting Evaluation Pipeline"
echo "=========================================="

# Configuration
CONFIG="configs/default.yaml"
CHECKPOINT="checkpoints/best_model.pt"
TEST_DATA="data/test"
TEST_ANNOT="data/test_annotations.txt"
OUTPUT_DIR="eval_results"

echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Test data: $TEST_DATA"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run evaluation
python eval.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --test-data "$TEST_DATA" \
    --test-annot "$TEST_ANNOT" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="