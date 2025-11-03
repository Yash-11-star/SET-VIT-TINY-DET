"""
Neck Module
Feature pyramid construction for multi-scale detection
"""

import torch.nn as nn


class FPN(nn.Module):
    """
    Feature Pyramid Network
    
    For Vision Transformer, this is minimal since we already have
    multi-scale features. This serves as a template for potential
    future enhancements.
    """
    
    def __init__(self, in_channels=768, out_channels=256):
        """
        Initialize FPN
        
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
        """
        super().__init__()
        self.lateral_conv = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
    
    def forward(self, features):
        """
        Forward pass
        
        Args:
            features: Input features
            
        Returns:
            Output features
        """
        return self.lateral_conv(features)