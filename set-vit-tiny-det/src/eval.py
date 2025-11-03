"""
Evaluation Script
Evaluate model on test dataset and perform inference

Usage:
    python eval.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --test-data data/test
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from src.datasets import TinyObjectDataset, TinyObjectAugmentation
from src.models import TinyObjectViT
from src.utils import visualize_detections


class Evaluator:
    """Model evaluator and inference"""
    
    def __init__(self, model_path, config, device):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            config: Configuration dict
            device: Device to evaluate on
        """
        self.config = config
        self.device = device
        
        # Load model
        model_cfg = config['MODEL']
        self.model = TinyObjectViT(
            image_size=model_cfg['IMAGE_SIZE'],
            patch_size=model_cfg['PATCH_SIZE'],
            num_classes=model_cfg['NUM_CLASSES'],
            num_heads=model_cfg['NUM_HEADS'],
            num_layers=model_cfg['NUM_LAYERS'],
            mlp_dim=model_cfg['MLP_DIM'],
            dropout=model_cfg['DROPOUT']
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def predict_single_image(self, image_path):
        """
        Make predictions on single image
        
        Args:
            image_path: Path to image
            
        Returns:
            detections dict
        """
        from PIL import Image
        import numpy as np
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = TinyObjectAugmentation.get_test_transforms(
            image_size=self.config['MODEL']['IMAGE_SIZE']
        )
        image_tensor = transform(image=np.array(image))['image'].unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Post-process
        bbox_preds = outputs['bbox_preds'][0]  # (num_patches, 4)
        cls_preds = outputs['cls_preds'][0]     # (num_patches, num_classes)
        
        confidences = F.softmax(cls_preds, dim=-1)
        max_confs, class_ids = torch.max(confidences, dim=1)
        
        # Filter by confidence threshold
        threshold = self.config['INFERENCE']['CONFIDENCE_THRESHOLD']
        valid_idx = max_confs > threshold
        
        detections = {
            'boxes': bbox_preds[valid_idx].cpu().numpy(),
            'classes': class_ids[valid_idx].cpu().numpy(),
            'confidences': max_confs[valid_idx].cpu().numpy()
        }
        
        return detections
    
    def evaluate_dataset(self, dataset, output_dir=None):
        """
        Evaluate on full dataset
        
        Args:
            dataset: Dataset to evaluate
            output_dir: Directory to save visualizations
            
        Returns:
            Statistics dict
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_images': len(dataset),
            'total_detections': 0,
            'avg_detections': 0
        }
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataset, desc='Evaluating')):
                image = batch['image'].unsqueeze(0).to(self.device)
                
                outputs = self.model(image)
                
                bbox_preds = outputs['bbox_preds'][0]
                cls_preds = outputs['cls_preds'][0]
                
                confidences = F.softmax(cls_preds, dim=-1)
                max_confs, class_ids = torch.max(confidences, dim=1)
                
                threshold = self.config['INFERENCE']['CONFIDENCE_THRESHOLD']
                valid_idx = max_confs > threshold
                
                detections = {
                    'boxes': bbox_preds[valid_idx].cpu().numpy(),
                    'classes': class_ids[valid_idx].cpu().numpy(),
                    'confidences': max_confs[valid_idx].cpu().numpy()
                }
                
                stats['total_detections'] += len(detections['boxes'])
                
                # Visualize if output directory provided
                if output_dir:
                    visualize_detections(
                        batch['image'],
                        detections,
                        save_path=output_dir / f'detection_{idx:04d}.png'
                    )
        
        stats['avg_detections'] = stats['total_detections'] / stats['total_images']
        
        return stats


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description='Evaluate tiny object detection model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, default='data/test',
                       help='Test data directory')
    parser.add_argument('--test-annot', type=str, default='data/test_annotations.txt',
                       help='Test annotations file')
    parser.add_argument('--image', type=str,
                       help='Path to single image for inference')
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize evaluator
    evaluator = Evaluator(args.checkpoint, config, device)
    
    # Single image inference
    if args.image:
        print(f"\nInferring on image: {args.image}")
        detections = evaluator.predict_single_image(args.image)
        print(f"Found {len(detections['boxes'])} objects")
        print(f"Boxes: {detections['boxes']}")
        print(f"Classes: {detections['classes']}")
        print(f"Confidences: {detections['confidences']}")
        
        # Visualize
        visualize_detections(
            torch.zeros(3, config['MODEL']['IMAGE_SIZE'], config['MODEL']['IMAGE_SIZE']),
            detections,
            save_path='./inference_result.png'
        )
        print(f"Visualization saved to ./inference_result.png")
    
    # Full dataset evaluation
    else:
        print(f"\nEvaluating on dataset: {args.test_data}")
        
        dataset = TinyObjectDataset(
            image_dir=args.test_data,
            annotations_file=args.test_annot,
            transforms=TinyObjectAugmentation.get_test_transforms(
                image_size=config['MODEL']['IMAGE_SIZE']
            )
        )
        
        stats = evaluator.evaluate_dataset(dataset, output_dir=args.output_dir)
        
        print(f"\n{'='*70}")
        print(f"Evaluation Results:")
        print(f"{'='*70}")
        print(f"Total images: {stats['total_images']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average detections per image: {stats['avg_detections']:.2f}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()