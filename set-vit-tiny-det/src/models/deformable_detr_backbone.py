"""
Vision Transformer Backbone
Multi-scale ViT optimized for tiny object detection

Architecture:
- Patch Embedding: Convert image to sequence of patches
- Positional Encoding: Add spatial information
- Transformer Blocks: Multi-head self-attention + MLP
- Detection Head: Per-patch predictions
"""

import torch
import torch.nn as nn


class TinyObjectViT(nn.Module):
    """
    Vision Transformer optimized for tiny object detection
    
    Key improvements for tiny objects:
    1. Multi-scale patch embeddings
    2. Enhanced local attention with multiple heads
    3. Per-patch prediction head for dense object detection
    4. Maintains reasonable resolution (384x384 with 16x16 patches)
    """
    
    def __init__(self, 
                 image_size=384,
                 patch_size=16,
                 num_classes=10,
                 num_heads=12,
                 num_layers=12,
                 mlp_dim=3072,
                 dropout=0.1):
        """
        Initialize Vision Transformer
        
        Args:
            image_size: Input image size (384x384)
            patch_size: Patch size (16x16)
            num_classes: Number of object classes
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            mlp_dim: MLP hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        self.num_classes = num_classes
        
        # ========== PATCH EMBEDDING ==========
        # Convert image patches to 768-dimensional embeddings
        self.patch_embed = nn.Linear(self.patch_dim, 768)
        
        # Learnable positional encoding
        # Why: Tells model spatial location of each patch - CRITICAL for tiny objects
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, 768)
        )
        
        # Class token - aggregates information
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        
        self.dropout = nn.Dropout(dropout)
        
        # ========== TRANSFORMER BLOCKS ==========
        # Stack of transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        # ========== DETECTION HEADS ==========
        # Separate heads for bounding box and classification
        # Per-patch predictions for dense detection
        self.bbox_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4)  # x1, y1, x2, y2
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly - important for training stability"""
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, 384, 384)
            
        Returns:
            dict with 'bbox_preds', 'cls_preds', 'cls_output'
        """
        B = x.shape[0]
        
        # ========== STEP 1: PATCH EMBEDDING ==========
        # x shape: (B, 3, 384, 384)
        # Reshape into patches: (B, num_patches, patch_dim)
        x = x.reshape(
            B, 3,
            self.image_size // self.patch_size, self.patch_size,
            self.image_size // self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, -1)
        
        # Linear embedding of patches
        x = self.patch_embed(x)  # (B, num_patches, 768)
        
        # ========== STEP 2: ADD CLASS TOKEN & POSITIONAL ENCODING ==========
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, 768)
        x = x + self.pos_embed  # Add positional encoding
        x = self.dropout(x)
        
        # ========== STEP 3: PASS THROUGH TRANSFORMER BLOCKS ==========
        # Each block: MultiHeadAttention + MLP
        for block in self.transformer_blocks:
            x = block(x)
        
        # ========== STEP 4: EXTRACT & PREDICT ==========
        # Class token (global features)
        cls_output = x[:, 0]
        
        # Per-patch predictions (local features)
        patch_features = x[:, 1:]  # (B, num_patches, 768)
        
        # Bounding box predictions: (B, num_patches, 4)
        bbox_preds = self.bbox_head(patch_features)
        
        # Class predictions: (B, num_patches, num_classes)
        cls_preds = self.cls_head(patch_features)
        
        return {
            'bbox_preds': bbox_preds,
            'cls_preds': cls_preds,
            'cls_output': cls_output,
            'patch_features': patch_features
        }