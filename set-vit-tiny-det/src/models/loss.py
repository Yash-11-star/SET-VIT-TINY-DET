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


def compute_patch_boxes(image_size, patch_size):
    """
    Compute bounding boxes for each patch in the image
    
    Args:
        image_size: Size of the input image (assumed square)
        patch_size: Size of each patch (assumed square)
        
    Returns:
        Tensor of patch boxes (N_patches, 4) in format (x1, y1, x2, y2)
    """
    num_patches = (image_size // patch_size)
    patch_boxes = []
    
    for i in range(num_patches):
        for j in range(num_patches):
            x1 = j * patch_size
            y1 = i * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size
            patch_boxes.append([x1, y1, x2, y2])
            
    return torch.tensor(patch_boxes, dtype=torch.float32)


def box_iou(boxes1, boxes2):
    """
    Compute IoU between all pairs of boxes
    
    Args:
        boxes1: (N, 4) tensor (x1, y1, x2, y2)
        boxes2: (M, 4) tensor (x1, y1, x2, y2)
        
    Returns:
        IoU matrix (N, M)
    """
    # Get coordinates
    x11, y11, x12, y12 = boxes1.unbind(-1)
    x21, y21, x22, y22 = boxes2.unbind(-1)
    
    # Compute areas
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    
    # Compute intersection
    xA = torch.max(x11[:, None], x21)
    yA = torch.max(y11[:, None], y21)
    xB = torch.min(x12[:, None], x22)
    yB = torch.min(y12[:, None], y22)
    
    intersection = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    union = area1[:, None] + area2 - intersection
    
    return intersection / (union + 1e-6)


class TinyObjectLoss(nn.Module):
    """
    Combined loss function for tiny object detection
    
    Key components:
    1. Patch-Box Matching:
       - Each patch is matched to ground truth boxes based on IoU
       - Positive patches: IoU > threshold
       - Negative patches: IoU < threshold
    
    2. Focal Loss:
       - Handles class imbalance (many background patches)
       - Down-weights easy negatives
       - Focuses training on hard positives
    
    3. Smooth L1 Loss:
       - Bounding box regression for positive patches
       - More robust to outliers than L2
       - Linear behavior for large errors
    """
    
    def __init__(self, image_size=512, patch_size=16,
                 alpha=0.25, gamma=2.0, bbox_weight=5.0, cls_weight=1.0,
                 iou_threshold=0.5):
        """
        Initialize loss function
        
        Args:
            image_size: Input image size (assumed square)
            patch_size: Patch size for ViT (assumed square)
            alpha: Focal loss alpha parameter (focus on positives)
            gamma: Focal loss gamma parameter (focus on hard examples)
            bbox_weight: Weight for bounding box regression loss
            cls_weight: Weight for classification loss
            iou_threshold: IoU threshold for positive patches
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bbox_weight = bbox_weight
        self.cls_weight = cls_weight
        self.iou_threshold = iou_threshold
        
        # Pre-compute patch boxes
        self.patch_boxes = compute_patch_boxes(image_size, patch_size)
        self.num_patches = len(self.patch_boxes)
    
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
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_lossbatch_size = predictions.size(0)
            total_focal_loss = 0
        
            for b in range(batch_size):
                pred = predictions[b]  # (num_patches, num_classes)
                tgt = targets[b]       # (num_patches,)
            
                ce_loss = F.cross_entropy(pred, tgt, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                total_focal_loss += focal_loss.mean()
    
            return total_focal_loss / batch_size
        
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
            
            batch_size = predictions.size(0)
            total_l1_loss = 0
        
            for b in range(batch_size):
                pred = predictions[b]  # (num_patches, 4)
                tgt = targets[b]      # (num_patches, 4)
            
                l1_loss = F.smooth_l1_loss(pred, tgt, reduction='mean', beta=1.0)
                total_l1_loss += l1_loss
        
            return total_l1_loss / batch_size
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
                - 'boxes': List of tensors [(N1, 4), (N2, 4), ...]
                - 'labels': List of tensors [(N1,), (N2,), ...]
                
        Returns:
            Loss values dict
        """
        device = outputs['bbox_preds'].device
        batch_size = outputs['bbox_preds'].shape[0]
        num_classes = outputs['cls_preds'].shape[-1]
        
        # Move patch boxes to device
        patch_boxes = self.patch_boxes.to(device)
        
        # Initialize total loss
        total_bbox_loss = 0
        total_cls_loss = 0
        num_positives = 0
        
        # Process each image in the batch
        for b in range(batch_size):
            # Get predictions for this image
            bbox_preds_b = outputs['bbox_preds'][b]  # (num_patches, 4)
            cls_preds_b = outputs['cls_preds'][b]    # (num_patches, num_classes)
            
            # Get ground truth boxes and labels
            gt_boxes = targets['boxes'][b]     # (N, 4)
            gt_labels = targets['labels'][b]   # (N,)
            
            # Skip if no ground truth boxes
            if len(gt_boxes) == 0:
                # For background-only images, all patches should predict background
                cls_target = torch.zeros(self.num_patches, dtype=torch.long, device=device)
                cls_loss = self.focal_loss(cls_preds_b, cls_target)
                total_cls_loss += cls_loss
                continue
            
            # Compute IoU between patches and ground truth boxes
            iou_matrix = box_iou(patch_boxes, gt_boxes)  # (num_patches, N)
            max_iou, gt_idx = iou_matrix.max(dim=1)      # (num_patches,)
            
            # Create labels for all patches
            cls_target = torch.zeros(self.num_patches, dtype=torch.long, device=device)
            pos_mask = max_iou > self.iou_threshold
            cls_target[pos_mask] = gt_labels[gt_idx[pos_mask]]
            
            # Classification loss
            cls_loss = self.focal_loss(
                cls_preds_b,  # Shape: (num_patches, num_classes)
                cls_target    # Shape: (num_patches,)
            )
            total_cls_loss += cls_loss
            
            # Regression loss (only for positive patches)
            if pos_mask.any():
                # Get ground truth boxes for positive patches
                pos_gt_boxes = gt_boxes[gt_idx[pos_mask]]
                # Get predictions for positive patches
                pos_bbox_preds = bbox_preds_b[pos_mask]
                # Compute regression loss
                bbox_loss = self.smooth_l1_loss(pos_bbox_preds, pos_gt_boxes)
                total_bbox_loss += bbox_loss
                num_positives += pos_mask.sum()
        
        # Average losses
        avg_cls_loss = total_cls_loss / batch_size
        avg_bbox_loss = total_bbox_loss / max(num_positives, 1)
        
        # Combine losses
        total_loss = self.cls_weight * avg_cls_loss + self.bbox_weight * avg_bbox_loss
        
        return {
            'total_loss': total_loss,
            'bbox_loss': avg_bbox_loss,
            'cls_loss': avg_cls_loss,
            'num_positives': num_positives
        }