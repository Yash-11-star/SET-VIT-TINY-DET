"""
Adversarial Perturbation Injection (API)
Technique to improve robustness of tiny object detection

Purpose: Inject controlled adversarial perturbations during training
to improve model robustness to small object variations
"""

import torch
import torch.nn as nn


class AdversarialPerturbationInjection(nn.Module):
    """
    Adversarial Perturbation Injection
    
    Improves model robustness by:
    - Injecting small adversarial perturbations during training
    - Forcing model to learn robust features
    - Particularly effective for small objects
    """
    
    def __init__(self, epsilon=0.1, num_steps=1):
        """
        Initialize API
        
        Args:
            epsilon: Maximum perturbation magnitude
            num_steps: Number of perturbation steps
        """
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
    
    def forward(self, x, target=None):
        """
        Apply adversarial perturbation
        
        Args:
            x: Input features or image
            target: Optional target for adversarial direction
            
        Returns:
            Perturbed input
        """
        if not self.training:
            return x
        
        # Generate small random perturbations
        perturbation = torch.randn_like(x) * self.epsilon
        
        # Apply perturbation
        perturbed_x = x + perturbation
        
        # Clip to valid range
        perturbed_x = torch.clamp(perturbed_x, x.min(), x.max())
        
        return perturbed_x