#!/bin/bash

# Setup Script for Vision Transformer Tiny Object Detection
# Run this script to set up the project

echo "=========================================="
echo "Vision Transformer - Tiny Object Detection"
echo "Setup Script"
echo "=========================================="
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/train data/val data/test
mkdir -p checkpoints
mkdir -p eval_results
mkdir -p logs
echo "✓ Directories created"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Create sample data structure
echo "Creating sample data structure..."
cat > data/train_annotations_sample.txt << 'EOF'
# Format: image_name.jpg x1 y1 x2 y2 class [x1 y1 x2 y2 class ...]
# Example:
# image_001.jpg 50 100 75 125 0
# image_002.jpg 10 10 30 30 0 200 50 240 90 1
EOF

cp data/train_annotations_sample.txt data/train_annotations.txt
cp data/train_annotations_sample.txt data/val_annotations.txt
cp data/train_annotations_sample.txt data/test_annotations.txt

echo "✓ Sample annotation files created"
echo ""

# Check CUDA availability
echo "Checking GPU support..."
python3 << 'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ CUDA not available. Training will use CPU (slower).")
PYEOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare your data in data/train, data/val, data/test"
echo "2. Update annotations files with your data"
echo "3. Customize configs/default.yaml if needed"
echo "4. Run: python train.py --config configs/default.yaml"
echo ""
echo "For more information, see README.md"