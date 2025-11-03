# Project Structure & File Organization

This document describes the complete file structure of the Vision Transformer Tiny Object Detection project.

## 📁 Complete Directory Tree

```
set-vit-tiny-det/
│
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # License file
├── 📄 .gitignore                   # Git ignore rules
├── 📄 requirements.txt             # Python dependencies
├── 📄 Makefile                     # Build/development commands
│
├── 📁 configs/                     # Configuration files
│   ├── default.yaml               # Default model & training config
│   └── coco_small.yaml            # COCO small config (template)
│
├── 📁 data/                        # Data directory
│   ├── README_DATA.md             # Data format and preparation guide
│   ├── train/                     # Training images (create this)
│   ├── train_annotations.txt      # Training annotations
│   ├── val/                       # Validation images (create this)
│   ├── val_annotations.txt        # Validation annotations
│   ├── test/                      # Test images (create this)
│   └── test_annotations.txt       # Test annotations
│
├── 📁 scripts/                     # Shell scripts
│   ├── setup_coco.sh              # Project setup script
│   ├── train.sh                   # Training script
│   └── eval.sh                    # Evaluation script
│
├── 📁 src/                         # Source code
│   │
│   ├── 📁 datasets/               # Dataset module
│   │   ├── __init__.py            # Module initialization
│   │   ├── coco.py               # TinyObjectDataset class
│   │   └── transforms.py         # Augmentation pipelines
│   │
│   ├── 📁 models/                 # Model architecture
│   │   ├── __init__.py            # Module initialization
│   │   ├── deformable_detr_backbone.py  # Main ViT model
│   │   ├── heads.py              # BBoxHead & ClassHead
│   │   ├── loss.py               # Loss functions (Focal + Smooth L1)
│   │   ├── neck.py               # FPN (Feature Pyramid Network)
│   │   │
│   │   └── 📁 set_modules/       # Advanced techniques
│   │       ├── __init__.py        # Module initialization
│   │       ├── hbs.py            # Hierarchical Background Smoothing
│   │       └── api.py            # Adversarial Perturbation Injection
│   │
│   └── 📁 utils/                  # Utility functions
│       ├── __init__.py            # Module initialization
│       ├── masks.py              # Binary mask generation
│       ├── dist.py               # Distributed training utilities
│       ├── meter.py              # Metric tracking (AverageMeter, ProgressMeter)
│       └── viz.py                # Visualization functions
│
├── 📄 train.py                    # Main training script
└── 📄 eval.py                     # Main evaluation/inference script
```

## 🗂️ File Descriptions

### Root Level Files

| File | Purpose |
|------|---------|
| `README.md` | Complete project documentation |
| `requirements.txt` | Python package dependencies |
| `Makefile` | Common development tasks (make train, make eval, etc.) |
| `.gitignore` | Git ignore patterns |
| `LICENSE` | Project license |

### Configuration Files (`configs/`)

| File | Purpose |
|------|---------|
| `default.yaml` | Default training configuration (model, data, optimization settings) |
| `coco_small.yaml` | Template for small COCO dataset configuration |

### Data Directory (`data/`)

| Item | Purpose |
|------|---------|
| `README_DATA.md` | Data format guide and examples |
| `train/` | Training image files |
| `train_annotations.txt` | Training ground truth annotations |
| `val/` | Validation image files |
| `val_annotations.txt` | Validation annotations |
| `test/` | Test image files |
| `test_annotations.txt` | Test annotations |

### Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `setup_coco.sh` | Initial project setup (creates directories, installs deps) |
| `train.sh` | Wrapper script to run training with default settings |
| `eval.sh` | Wrapper script to run evaluation with default settings |

### Source Code (`src/`)

#### Datasets Module (`src/datasets/`)

| File | Class/Function | Purpose |
|------|---|---------|
| `__init__.py` | - | Module exports |
| `transforms.py` | `TinyObjectAugmentation` | Data augmentation pipelines for training/validation/test |
| `coco.py` | `TinyObjectDataset` | Custom dataset class for loading images and annotations |

#### Models Module (`src/models/`)

| File | Class/Function | Purpose |
|------|---|---------|
| `__init__.py` | - | Module exports |
| `deformable_detr_backbone.py` | `TinyObjectViT` | Main Vision Transformer architecture |
| `heads.py` | `BBoxHead`, `ClassHead` | Prediction heads for detection |
| `loss.py` | `TinyObjectLoss` | Combined Focal Loss + Smooth L1 Loss |
| `neck.py` | `FPN` | Feature Pyramid Network (template) |

#### SET Modules (`src/models/set_modules/`)

| File | Class/Function | Purpose |
|------|---|---------|
| `__init__.py` | - | Module exports |
| `hbs.py` | `HierarchicalBackgroundSmoothing` | Background noise reduction technique |
| `api.py` | `AdversarialPerturbationInjection` | Adversarial training for robustness |

#### Utilities Module (`src/utils/`)

| File | Class/Function | Purpose |
|------|---|---------|
| `__init__.py` | - | Module exports |
| `masks.py` | `MaskGenerator` | Generate binary masks for detection |
| `dist.py` | `synchronize()`, `get_rank()`, `get_world_size()` | Distributed training helpers |
| `meter.py` | `AverageMeter`, `ProgressMeter` | Track metrics and training progress |
| `viz.py` | `visualize_detections()`, `plot_training_curves()` | Visualization utilities |

### Main Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Main training pipeline with Trainer class |
| `eval.py` | Evaluation and inference script with Evaluator class |

## 🔄 Data Flow

### Training Flow
```
data/ → TinyObjectDataset → DataLoader
                ↓
        TinyObjectAugmentation
                ↓
        train.py (Trainer)
                ↓
        TinyObjectViT (model)
                ↓
        TinyObjectLoss (criterion)
                ↓
        Optimizer (AdamW)
                ↓
        checkpoints/best_model.pt
```

### Inference Flow
```
data/ → Image Loading → Augmentation (test transforms)
                ↓
        TinyObjectViT (model)
                ↓
        Detection heads (bbox + class predictions)
                ↓
        Post-processing (confidence filtering)
                ↓
        Detections (boxes, classes, scores)
                ↓
        visualize_detections()
                ↓
        eval_results/
```

## 📝 Key Configuration Files

### `configs/default.yaml`

Controls all training parameters:

```yaml
MODEL:           # Model architecture settings
TRAIN:           # Training hyperparameters
DATA:            # Data loading settings
OPTIMIZER:       # Optimizer configuration
SCHEDULER:       # Learning rate scheduler
CHECKPOINT:      # Model saving
LOGGING:         # Logging configuration
INFERENCE:       # Inference settings
```

### `data/train_annotations.txt` Format

```
image_001.jpg 50 100 75 125 0
image_002.jpg 10 10 30 30 0 200 50 240 90 1
image_003.jpg
```

Format: `filename x1 y1 x2 y2 class [x1 y1 x2 y2 class ...]`

## 🚀 Usage Examples

### 1. Setup Project
```bash
bash scripts/setup_coco.sh
```

### 2. Train Model
```bash
python train.py --config configs/default.yaml --epochs 50
```

### 3. Evaluate Model
```bash
python eval.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt
```

### 4. Single Image Inference
```bash
python eval.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --image test.jpg
```

### 5. Using Makefile
```bash
make install          # Install dependencies
make train            # Train the model
make eval             # Evaluate the model
make clean            # Clean cache
make gpu-info         # Check GPU
```

## 📦 Dependencies

See `requirements.txt`:
- torch
- torchvision
- numpy
- pillow
- albumentations
- pyyaml
- opencv-python
- matplotlib
- tqdm

## 🔍 File Locations Reference

| What | Where |
|------|-------|
| Trained models | `checkpoints/` |
| Evaluation results | `eval_results/` |
| Training logs | `logs/` |
| Configuration | `configs/*.yaml` |
| Training code | `train.py` |
| Evaluation code | `eval.py` |
| Model architecture | `src/models/deformable_detr_backbone.py` |
| Dataset loading | `src/datasets/coco.py` |
| Loss functions | `src/models/loss.py` |
| Augmentation | `src/datasets/transforms.py` |

## 🎯 Next Steps

1. **Setup**: Run `bash scripts/setup_coco.sh`
2. **Prepare Data**: Put images in `data/train`, `data/val`, `data/test`
3. **Create Annotations**: Write annotation files
4. **Configure**: Edit `configs/default.yaml` as needed
5. **Train**: Run `python train.py --config configs/default.yaml`
6. **Evaluate**: Run `python eval.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt`

---

For more details, see `README.md` and individual module documentation.