"""
Visualization Utilities
Visualize detection results and training progress
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_detections(image, detections, class_names=None, save_path=None):
    """
    Visualize bounding box detections on image
    
    Args:
        image: Input image (numpy array or tensor)
        detections: Dict with 'boxes', 'classes', 'confidences'
        class_names: List of class names
        save_path: Path to save visualization
        
    Returns:
        Visualized image
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)
    
    # Denormalize if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(image)
    
    # Default class names
    if class_names is None:
        max_class = max(detections['classes']) if len(detections['classes']) > 0 else 10
        class_names = [f"Class {i}" for i in range(max_class + 1)]
    
    # Color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    # Draw detections
    for box, class_id, conf in zip(
        detections['boxes'],
        detections['classes'],
        detections['confidences']
    ):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=colors[class_id],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        label = f"{class_names[class_id]}\n{conf:.2f}"
        ax.text(x1, y1 - 5, label, color=colors[class_id],
               fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_title(f"Detections ({len(detections['boxes'])} objects)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves
    
    Args:
        history: Training history dict
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].semilogy(epochs, history['learning_rate'], 'g-')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig