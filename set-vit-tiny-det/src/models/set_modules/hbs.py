"""
Hierarchical Background Smoothing (HBS)
Technique to reduce background noise and make tiny objects more visible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalBackgroundSmoothing(nn.Module):
    """
    Hierarchical Background Smoothing
    
    Purpose: Reduce background clutter to make tiny objects more visible
    
    Approach:
    - Extract background at multiple scales
    - Smooth background
    - Subtract from original image to enhance objects
    """
    
    def __init__(self, kernel_sizes=[3, 5, 7]):
        """
        Initialize HBS
        
        Args:
            kernel_sizes: List of kernel sizes for smoothing
        """
        super().__init__()
        self.kernel_sizes = kernel_sizes
    
    def forward(self, x, image=None):
        """
        Apply hierarchical background smoothing
        
        Args:
            x: Input features
            image: Optional input image for visualization
            
        Returns:
            Enhanced features with background smoothed
        """
        # Multi-scale background smoothing
        smoothed_features = x.clone()
        
        for kernel_size in self.kernel_sizes:
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Depthwise convolution for smoothing
            padding = kernel_size // 2
            kernel = torch.ones(1, 1, kernel_size, kernel_size,
                              device=x.device, dtype=x.dtype) / (kernel_size ** 2)
            
            smoothed = F.conv2d(x, kernel, padding=padding)
            smoothed_features = smoothed_features - 0.1 * smoothed
        
        return smoothed_features