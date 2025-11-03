# Data Format and Preparation

## Dataset Structure

Your dataset should be organized as follows:

```
data/
├── train/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── image_003.png
│   └── ...
├── train_annotations.txt
│
├── val/
│   ├── image_val_001.jpg
│   ├── image_val_002.jpg
│   └── ...
├── val_annotations.txt
│
├── test/
│   ├── image_test_001.jpg
│   └── ...
└── test_annotations.txt (optional)
```

## Annotation Format

Each `.txt` file should contain one line per image:

### Format:
```
image_name.jpg x1 y1 x2 y2 class [x1 y1 x2 y2 class ...]
```

### Explanation:
- `image_name.jpg`: Image filename
- `x1 y1 x2 y2`: Bounding box coordinates (top-left to bottom-right)
- `class`: Integer class ID (0-indexed)

### Examples:

**Single object:**
```
car_001.jpg 100 50 200 150 0
```

**Multiple objects:**
```
street_scene.jpg 50 100 150 200 0 300 150 400 250 1 500 50 550 100 2
```

**No objects:**
```
empty_street.jpg
```

**Mixed:**
```
scene_001.jpg 10 20 50 60 0
scene_002.jpg
scene_003.jpg 100 100 200 200 1 300 50 400 150 0 450 200 500 300 2
```

## Coordinate System

- Origin (0,0) is at **top-left** corner
- X increases to the **right**
- Y increases **downward**
- All coordinates are in **pixel units**

```
(0,0) ─────────── x ──────────►
  │
  │
  y
  │
  ▼
```

## Class IDs

Assign integer IDs starting from 0:
- 0: Class A
- 1: Class B
- 2: Class C
- ...
- N: Class Z

## Data Validation

Before training, validate your data:

```python
from src.utils import DataValidator

validator = DataValidator()
validator.check_annotations_format('data/train_annotations.txt')
```

This will:
- Check filename extensions
- Verify coordinate ranges
- Validate class IDs
- Detect malformed lines

## Best Practices

1. **Consistent image formats**: Use JPG or PNG
2. **Reasonable image sizes**: 512×512 to 2048×2048 is good
3. **Accurate annotations**: Tight bounding boxes around objects
4. **Balance classes**: Try to have similar number of each class
5. **Include background**: Vary scenes and viewpoints
6. **Handle edge cases**: Partially visible objects, occlusion

## For Tiny Objects

When annotating tiny objects:
- Use coordinates precise to pixel level
- Ensure boxes don't overlap unnecessarily
- Include full tiny object (even if small)
- Provide enough context around object

Example tiny object:
```
small_bird.jpg 150 200 155 205 2
# This is a 5×5 pixel bird at position (150,200) to (155,205)
```

## Dataset Size Recommendations

### Minimum:
- 100 images for quick testing
- At least 10 examples per class

### Good:
- 500-1000 training images
- 100-200 validation images
- 50-100 test images

### Ideal:
- 1000+ training images
- 200+ validation images
- 200+ test images
- Diverse scenes and conditions

## Augmentation

The code includes automatic augmentation:
- Contrast enhancement
- Random rotations
- Noise injection
- Random crops
- Brightness/contrast variations

No need to manually augment your dataset!

## Data Statistics

After loading a dataset, check statistics:

```python
from src.datasets import TinyObjectDataset

dataset = TinyObjectDataset('data/train', 'data/train_annotations.txt')
print(f"Total images: {len(dataset)}")

# Count objects per class
class_counts = {}
for i in range(len(dataset)):
    sample = dataset[i]
    labels = sample['labels']
    for label in labels:
        label_id = label.item()
        class_counts[label_id] = class_counts.get(label_id, 0) + 1

print("Objects per class:")
for cls_id, count in sorted(class_counts.items()):
    print(f"  Class {cls_id}: {count}")
```

## Troubleshooting

### Issue: "FileNotFoundError: Image not found"
→ Check image path in annotations file matches actual filename

### Issue: "Invalid box coordinates"
→ Ensure x1 < x2 and y1 < y2

### Issue: "Class ID out of range"
→ Class IDs should be 0 to NUM_CLASSES-1

### Issue: "Mismatch between image and annotation"
→ Verify image exists and annotation format is correct