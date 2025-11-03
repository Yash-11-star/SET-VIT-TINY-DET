# Vision Transformer for Tiny Object Detection

A complete implementation of Vision Transformer (ViT) optimized for detecting small objects in noisy images.

## 🎯 Project Overview

**Problem**: Standard computer vision models struggle to detect tiny objects because:
- Background noise dominates the image
- Tiny objects occupy few pixels
- Standard models' attention gets diluted

**Solution**: Multi-scale Vision Transformer with specialized augmentation, focal loss, and per-patch predictions.

## 📁 Project Structure

```
set-vit-tiny-det/
├── README.md                 # This file
├── LICENSE
├── .gitignore
├── requirements.txt          # Python dependencies
├── Makefile
├── configs/
│   └── default.yaml         # Default configuration
├── data/
│   └── README_DATA.md       # Data format documentation
├── scripts/
│   ├── train.sh            # Training script
│   └── eval.sh             # Evaluation script
├── src/
│   ├── datasets/           # Data loading & augmentation
│   │   ├── __init__.py
│   │   ├── coco.py        # Dataset class
│   │   └── transforms.py  # Augmentation pipeline
│   ├── models/             # Model architecture
│   │   ├── __init__.py
│   │   ├── deformable_detr_backbone.py  # Main ViT model
│   │   ├── heads.py        # Detection heads
│   │   ├── loss.py         # Focal + Smooth L1 losses
│   │   ├── neck.py         # Feature pyramid
│   │   └── set_modules/    # Advanced techniques
│   │       ├── __init__.py
│   │       ├── hbs.py      # Hierarchical Background Smoothing
│   │       └── api.py      # Adversarial Perturbation Injection
│   └── utils/              # Utilities
│       ├── __init__.py
│       ├── masks.py        # Mask generation
│       ├── dist.py         # Distributed training
│       ├── meter.py        # Metric tracking
│       └── viz.py          # Visualization
├── train.py               # Main training script
└── eval.py                # Main evaluation script
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd set-vit-tiny-det

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Create your dataset structure:
```
data/
├── train/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── train_annotations.txt
├── val/
│   └── ...
└── val_annotations.txt
```

Annotation format (`train_annotations.txt`):
```
image001.jpg 50 100 75 125 0 200 50 240 90 1
image002.jpg 10 10 30 30 0
image003.jpg
```
Format: `image_name.jpg x1 y1 x2 y2 class [x1 y1 x2 y2 class ...]`

### 3. Train Model

```bash
# Using shell script
bash scripts/train.sh

# Or directly
python train.py --config configs/default.yaml --epochs 50
```

### 4. Evaluate Model

```bash
# Using shell script
bash scripts/eval.sh

# Or directly
python eval.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --test-data data/test
```

### 5. Single Image Inference

```bash
python eval.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/image.jpg
```

## 🏗️ Architecture Overview

### Vision Transformer (ViT) Backbone

1. **Patch Embedding** (384×384 image → 16×16 patches)
   - Converts image into 576 patch embeddings
   - Each patch: 768-dimensional vector

2. **Positional Encoding**
   - Learnable positional embeddings
   - Tells model spatial location of each patch
   - Critical for tiny object localization

3. **Transformer Blocks** (12 layers)
   - Multi-head self-attention (12 heads)
   - Feed-forward network (MLP)
   - Each head specializes in different features

4. **Detection Heads** (Per-patch predictions)
   - Bounding box head: predicts (x1, y1, x2, y2)
   - Classification head: predicts class probabilities

### Loss Functions

**Focal Loss** (Classification)
- Addresses class imbalance (95% background, 5% objects)
- Down-weights easy examples, focuses on hard ones
- Formula: FL(pt) = -α * (1 - pt)^γ * log(pt)

**Smooth L1 Loss** (Bounding Box)
- Quadratic for small errors (precision)
- Linear for large errors (robustness)
- Important for pixel-level accuracy in tiny objects

## 📊 Configuration

Edit `configs/default.yaml` to customize:

```yaml
MODEL:
  IMAGE_SIZE: 384         # Input image size
  PATCH_SIZE: 16          # Patch size
  NUM_CLASSES: 10         # Number of classes
  NUM_HEADS: 12           # Attention heads
  NUM_LAYERS: 12          # Transformer blocks
  DROPOUT: 0.1

TRAIN:
  EPOCHS: 50
  BATCH_SIZE: 32
  LEARNING_RATE: 1.0e-4
  FOCAL_ALPHA: 0.25       # Focal loss alpha
  FOCAL_GAMMA: 2.0        # Focal loss gamma
  BBOX_LOSS_WEIGHT: 5.0
  CLS_LOSS_WEIGHT: 1.0
```

## 📈 Expected Performance

### On Different Object Sizes
- **Tiny (8-32px)**: mAP ~55-65%
- **Small (32-64px)**: mAP ~70-80%
- **Medium (64-128px)**: mAP ~85-90%

### Training Timeline
- **Epoch 1-5**: Fast loss decrease
- **Epoch 5-20**: Gradual improvement
- **Epoch 20-50**: Plateauing

## 🔧 Advanced Features

### Data Augmentation for Tiny Objects
- Contrast enhancement (makes objects visible)
- Gaussian blur (robust edge learning)
- Random crops (multi-scale training)
- Noise injection (real-world robustness)

### SET Modules (Optional)

1. **Hierarchical Background Smoothing (HBS)**
   - Reduces background noise
   - Multi-scale smoothing
   - Enhances tiny object visibility

2. **Adversarial Perturbation Injection (API)**
   - Improves robustness
   - Handles small object variations
   - Prevents overfitting

## 🐛 Troubleshooting

### Issue: "Model predicts everything as background"
→ Increase `cls_weight` or `focal_alpha`

### Issue: "Loss not decreasing"
→ Reduce learning rate or check augmentation intensity

### Issue: "Out of memory"
→ Reduce `BATCH_SIZE` or `IMAGE_SIZE`

### Issue: "Bounding boxes inaccurate"
→ Increase `bbox_weight` or train longer

## 📚 Key Concepts

### Why Tiny Objects Need Special Treatment

1. **Resolution Issue**
   - 32×32 object in 384×384 image = only 4 patches
   - Background dilutes attention

2. **Class Imbalance**
   - 95% background pixels
   - Standard training ignores objects

3. **Noise Dominance**
   - Small objects = few signal pixels
   - Noise overwhelms signal

### Solution: Multi-Scale ViT

1. **Keep reasonable resolution** (384×384 with 16×16 patches)
2. **Focal Loss** (focus on hard examples)
3. **Smart Augmentation** (enhance visibility)
4. **Per-Patch Predictions** (dense object detection)

## 🎓 Understanding the Approach

### Model Decision Rationale

| Component | Choice | Why |
|-----------|--------|-----|
| Image Size | 384×384 | Balance between detail and speed |
| Patch Size | 16×16 | 32×32 object spans 4 patches |
| Num Layers | 12 | ViT standard, proven performance |
| Num Heads | 12 | Multiple feature specialization |
| Loss | Focal + Smooth L1 | Class imbalance + precision |
| Optimizer | AdamW | Adaptive learning + weight decay |

## 📖 References

- Vision Transformer (ViT): https://arxiv.org/abs/2010.11929
- Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
- DETR: https://arxiv.org/abs/2005.12139

## 📝 Citation

If you use this project, please cite:

```bibtex
@software{tiny_vit_detector,
  title={Vision Transformer for Tiny Object Detection},
  author={Your Name},
  year={2024}
}
```

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy Training!** 🚀