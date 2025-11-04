"""
Custom collate functions for handling batches with variable-sized data
"""

import torch
from typing import List, Dict


def collate_variable_boxes(batch: List[Dict]) -> Dict:
    """
    Collate function for batches with variable number of bounding boxes
    
    Args:
        batch: List of dicts with 'image', 'boxes', 'labels', 'is_valid'
        
    Returns:
        Collated batch with separate lists for boxes and labels
    """
    # Stack images
    images = torch.stack([b['image'] for b in batch], 0)
    
    # Keep boxes and labels as lists (variable length)
    boxes = [b['boxes'] for b in batch]
    labels = [b['labels'] for b in batch]
    is_valid = [b['is_valid'] for b in batch]
    
    return {
        'images': images,
        'boxes': boxes,  # List of tensors [(N1, 4), (N2, 4), ...]
        'labels': labels,  # List of tensors [(N1,), (N2,), ...]
        'is_valid': is_valid,  # List of tensors [(N1,), (N2,), ...]
    }