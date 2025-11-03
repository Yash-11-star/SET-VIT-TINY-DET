"""
Loss Functions
Specialized losses for tiny object detection

Key components:
1. Focal Loss - Handles class imbalance (background vs. objects)
2. Smooth L1 Loss - Precise bounding box regression
3. Weighted combination - Tunable importance of each component
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyObjectLoss(nn.Module):
    """
    Combined loss function for tiny object detection
    
    Combines:
    - Focal Loss: Handles class imbalance by focusing on hard examples
    - Smooth L1 Loss: Bounding box regression with robustness to outliers
    
    Why focal loss?
    Standard cross-entropy treats all pixels equally. In tiny object detection:
    - 95% of image is background (easy)
    - 5% contains objects (hard)
    
    Without focal loss, model learns to ignore objects (95% accuracy by default).
    Focal loss down-weights easy negatives and focuses on hard positives.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, bbox_weight=5.0, cls_weight=1.0):
        """
        Initialize loss function
        
        Args:
            alpha: Focal loss alpha parameter
                - Higher = more focus on positives
            gamma: Focal loss gamma (focusing parameter)
                - Higher = more focus on hard examples
            bbox_weight: Weight for bounding box loss
            cls_weight: Weight for classification loss
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight
    
    def focal_loss(self, predictions, targets):
        """
        Focal Loss for classification
        
        Formula:
        FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
        
        Where:
        - pt: probability of true class
        - alpha: balancing factor
        - gamma: focusing parameter
        
        Why:
        - (1 - pt)^gamma = 0 for easy examples (high confidence)
        - (1 - pt)^gamma = 1 for hard examples (low confidence)
        
        This focuses training on misclassified tiny objects.
        
        Args:
            predictions: Model predictions (B*num_patches, num_classes)
            targets: Ground truth labels (B*num_patches,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def smooth_l1_loss(self, predictions, targets):
        """
        Smooth L1 Loss for bounding box regression
        
        Formula:
        - If |x| < 1: loss = 0.5 * x^2 (quadratic)
        - Otherwise: loss = |x| - 0.5 (linear)
        
        Why:
        - Small errors: Quadratic penalty encourages precision
        - Large errors: Linear prevents explosion (robust to outliers)
        
        For tiny objects where each pixel matters, this precision is crucial.
        
        Args:
            predictions: Predicted bboxes (B*num_patches, 4)
            targets: Ground truth bboxes (B*num_patches, 4)
            
        Returns:
            Smooth L1 loss value
        """
        return F.smooth_l1_loss(predictions, targets, reduction='mean', beta=1.0)
    
    def forward(self, outputs, targets):
        """
        Compute total loss
        
        Args:
            outputs: Model outputs dict
                - 'bbox_preds': (B, num_patches, 4)
                - 'cls_preds': (B, num_patches, num_classes)
            targets: Ground truth dict
                - 'boxes': (B, num_patches, 4) or variable length
                - 'labels': (B, num_patches) or variable length
                
        Returns:
            Total loss value
        """
        # Flatten predictions and targets
        bbox_preds = outputs['bbox_preds'].view(-1, 4)
        cls_preds = outputs['cls_preds'].view(-1, outputs['cls_preds'].shape[-1])
        
        # Handle targets
        if isinstance(targets['boxes'], list):
            # Variable length targets - pad them
            max_boxes = max(len(b) for b in targets['boxes'])
            boxes = torch.zeros(len(targets['boxes']), max_boxes, 4)
            labels = torch.zeros(len(targets['boxes']), max_boxes, dtype=torch.long)
            
            for i, (b, l) in enumerate(zip(targets['boxes'], targets['labels'])):
                boxes[i, :len(b)] = b
                labels[i, :len(l)] = l
            
            boxes = boxes.view(-1, 4)
            labels = labels.view(-1)
        else:
            boxes = targets['boxes'].view(-1, 4)
            labels = targets['labels'].view(-1)
        
        # Compute individual losses
        bbox_loss = self.smooth_l1_loss(bbox_preds, boxes)
        cls_loss = self.focal_loss(cls_preds, labels)
        
        # Weighted combination
        total_loss = self.bbox_weight * bbox_loss + self.cls_weight * cls_loss
        
        return {
            'total_loss': total_loss,
            'bbox_loss': bbox_loss,
            'cls_loss': cls_loss
        }