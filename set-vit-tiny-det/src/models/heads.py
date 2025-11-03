"""
Detection Heads
Prediction heads for bounding box and classification
"""

import torch.nn as nn


class BBoxHead(nn.Module):
    """
    Bounding Box Prediction Head
    
    Predicts: x1, y1, x2, y2 for each patch
    """
    
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=4, dropout=0.1):
        """
        Initialize BBox head
        
        Args:
            in_dim: Input feature dimension (768 from ViT)
            hidden_dim: Hidden layer dimension
            out_dim: Output dimension (4 for bbox)
            dropout: Dropout rate
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Features (B, num_patches, in_dim)
            
        Returns:
            Bounding box predictions (B, num_patches, 4)
        """
        return self.head(x)


class ClassHead(nn.Module):
    """
    Classification Head
    
    Predicts: class probabilities for each patch
    """
    
    def __init__(self, in_dim=768, hidden_dim=256, num_classes=10, dropout=0.1):
        """
        Initialize Classification head
        
        Args:
            in_dim: Input feature dimension (768 from ViT)
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes
            dropout: Dropout rate
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Features (B, num_patches, in_dim)
            
        Returns:
            Class predictions (B, num_patches, num_classes)
        """
        return self.head(x)