"""
SET Modules (Spatial Enhancement Techniques)
Advanced techniques for improving tiny object detection
"""

from .hbs import HierarchicalBackgroundSmoothing
from .api import AdversarialPerturbationInjection

__all__ = [
    'HierarchicalBackgroundSmoothing',
    'AdversarialPerturbationInjection'
]