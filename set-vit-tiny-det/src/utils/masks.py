"""
Mask Utilities
GT-binary mask generation at feature scales
"""

import torch


class MaskGenerator:
    """
    Generate binary masks for ground truth at feature scales
    """
    
    @staticmethod
    def generate_mask(boxes, labels, image_size=384, patch_size=16):
        """
        Generate binary mask for tiny objects
        
        Args:
            boxes: Bounding boxes
            labels: Class labels
            image_size: Image size
            patch_size: Patch size
            
        Returns:
            Binary mask at patch level
        """
        num_patches = (image_size // patch_size) ** 2
        mask = torch.zeros(num_patches, dtype=torch.long)
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            
            # Convert to patch coordinates
            patch_x1 = int(x1 // patch_size)
            patch_y1 = int(y1 // patch_size)
            patch_x2 = int((x2 + patch_size - 1) // patch_size)
            patch_y2 = int((y2 + patch_size - 1) // patch_size)
            
            # Mark patches containing objects
            grid_size = image_size // patch_size
            for py in range(patch_y1, min(patch_y2, grid_size)):
                for px in range(patch_x1, min(patch_x2, grid_size)):
                    patch_idx = py * grid_size + px
                    mask[patch_idx] = label + 1
        
        return mask