"""
Dataset Module
Custom dataset for tiny object detection
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class TinyObjectDataset(Dataset):
    """
    Custom dataset for tiny objects
    
    Expected structure:
    - images/: folder with image files (jpg, png)
    - annotations.txt: format "image_name.jpg x1 y1 x2 y2 class [x1 y1 x2 y2 class ...]"
    
    Example annotation:
        image001.jpg 50 100 75 125 0 200 50 240 90 1
        image002.jpg 10 10 30 30 0
        image003.jpg
    """
    
    def __init__(self, image_dir, annotations_file, transforms=None):
        """
        Initialize dataset
        
        Args:
            image_dir: Directory containing images
            annotations_file: Path to annotations file
            transforms: Augmentation pipeline
        """
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.images = []
        self.annotations = []
        
        # Parse annotations
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0].startswith('#'):
                    continue
                    
                image_name = parts[0]
                boxes = []
                labels = []
                
                # Parse bounding boxes (x1, y1, x2, y2, class)
                for i in range(1, len(parts), 5):
                    if i + 4 <= len(parts):
                        x1, y1, x2, y2, cls = map(int, parts[i:i+5])
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls)
                
                self.images.append(image_name)
                self.annotations.append({
                    'boxes': boxes,
                    'labels': labels
                })
    
    def __len__(self):
        """Number of samples"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get single sample
        
        Returns:
            dict with 'image', 'boxes', 'labels'
        """
        # Load image
        image_path = self.image_dir / self.images[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Get annotations
        boxes = self.annotations[idx]['boxes']
        labels = self.annotations[idx]['labels']
        
        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['class_labels']
        
        # Convert to tensors
        if not boxes:
            boxes = [[0, 0, 1, 1]]  # Dummy box
            labels = [0]  # Background class
            is_valid = [False]  # Mark as invalid
        else:
            is_valid = [True] * len(boxes)
            
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'is_valid': torch.tensor(is_valid, dtype=torch.bool)
        }