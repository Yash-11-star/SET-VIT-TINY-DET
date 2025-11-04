"""
Augmentation Module
Specialized augmentation for tiny object detection
Why: Tiny objects need special treatment to be visible during training
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


class TinyObjectAugmentation:
    """
    Custom augmentation strategy optimized for tiny objects
    
    Key principle: Enhance tiny object visibility while maintaining realism
    """
    
    @staticmethod
    def get_train_transforms(image_size=384):
        """
        Training augmentations - AGGRESSIVE but smart
        
        Why each augmentation for tiny objects:
        - Contrast: Makes tiny objects visible against noise
        - Blur: Forces edge-based learning
        - Rotations: Rotation invariance
        - Random crops: Multi-scale training
        - Noise: Real-world robustness
        
        Args:
            image_size: Size of output image
            
        Returns:
            Augmentation pipeline
        """
        return A.Compose([
            # Enhance visibility of tiny objects
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.3,
                p=0.7
            ),
            
            # Forces robust feature learning
            A.Blur(blur_limit=3, p=0.3),
            
            # Geometric invariance
            A.Rotate(limit=15, p=0.5),
            
            # Multi-scale training - critical for tiny objects
            A.RandomCrop(width=image_size, height=image_size, p=0.7),
            
            # Spatial augmentation
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            
            # Simulate real-world noisy conditions
            A.GaussNoise(p=0.5),
            
            # Normalization with ImageNet stats
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.1  # Keep boxes if 10% visible
        ))
    
    @staticmethod
    def get_val_transforms(image_size=384):
        """
        Validation augmentations - MINIMAL, just normalization
        
        No randomness: Want consistent evaluation
        
        Args:
            image_size: Size of output image
            
        Returns:
            Validation pipeline
        """
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_test_transforms(image_size=384):
        """
        Test augmentations - MINIMAL, for inference
        
        Args:
            image_size: Size of output image
            
        Returns:
            Test pipeline
        """
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])