"""
Training Script
Complete training pipeline for Vision Transformer tiny object detection

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --epochs 100
"""

import argparse
import yaml
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

from src.datasets import TinyObjectDataset, TinyObjectAugmentation
from src.models import TinyObjectViT, TinyObjectLoss
from src.utils import AverageMeter, ProgressMeter, is_main_process


class Trainer:
    """Training orchestrator"""
    
    def __init__(self, config, device):
        """
        Initialize trainer
        
        Args:
            config: Configuration dict
            device: Device to train on
        """
        self.config = config
        self.device = device
        self.best_val_loss = float('inf')
        
        # Initialize model
        model_cfg = config['MODEL']
        self.model = TinyObjectViT(
            image_size=model_cfg['IMAGE_SIZE'],
            patch_size=model_cfg['PATCH_SIZE'],
            num_classes=model_cfg['NUM_CLASSES'],
            num_heads=model_cfg['NUM_HEADS'],
            num_layers=model_cfg['NUM_LAYERS'],
            mlp_dim=model_cfg['MLP_DIM'],
            dropout=model_cfg['DROPOUT']
        ).to(device)
        
        # Loss function
        train_cfg = config['TRAIN']
        self.criterion = TinyObjectLoss(
            alpha=train_cfg['FOCAL_ALPHA'],
            gamma=train_cfg['FOCAL_GAMMA'],
            bbox_weight=train_cfg['BBOX_LOSS_WEIGHT'],
            cls_weight=train_cfg['CLS_LOSS_WEIGHT']
        )
        
        # Optimizer
        opt_cfg = config['OPTIMIZER']
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg['LEARNING_RATE'],
            betas=opt_cfg['BETAS'],
            weight_decay=train_cfg['WEIGHT_DECAY']
        )
        
        # Learning rate scheduler
        sched_cfg = config['SCHEDULER']
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=sched_cfg['MODE'],
            factor=sched_cfg['FACTOR'],
            patience=sched_cfg['PATIENCE'],
            verbose=sched_cfg['VERBOSE']
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['CHECKPOINT']['SAVE_DIR'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_main_process():
            print(f"\n{'='*70}")
            print(f"Model: {model_cfg['NAME']}")
            print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Device: {device}")
            print(f"{'='*70}\n")
    
    def train_epoch(self, train_loader, epoch):
        """
        Single training epoch
        
        Args:
            train_loader: Training data loader
            epoch: Epoch number
            
        Returns:
            Average loss
        """
        self.model.train()
        loss_meter = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(len(train_loader), [loss_meter],
                                prefix=f'Epoch [{epoch}]')
        
        train_cfg = self.config['TRAIN']
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            boxes = batch['boxes'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Prepare targets
            targets = {
                'boxes': boxes.view(-1, 4),
                'labels': labels.view(-1)
            }
            
            # Flatten outputs
            loss_outputs = {
                'bbox_preds': outputs['bbox_preds'].view(-1, 4),
                'cls_preds': outputs['cls_preds'].view(-1, self.config['MODEL']['NUM_CLASSES'])
            }
            
            # Compute loss
            loss_dict = self.criterion(loss_outputs, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                train_cfg['GRADIENT_CLIP']
            )
            self.optimizer.step()
            
            # Update meter
            loss_meter.update(loss.item(), n=images.size(0))
            
            # Print progress
            if (batch_idx + 1) % train_cfg['PRINT_INTERVAL'] == 0 and is_main_process():
                progress.display(batch_idx + 1)
        
        return loss_meter.avg
    
    def evaluate(self, val_loader):
        """
        Validation/Evaluation
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss
        """
        self.model.eval()
        loss_meter = AverageMeter('Loss', ':.4f')
        
        model_cfg = self.config['MODEL']
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Prepare targets
                targets = {
                    'boxes': boxes.view(-1, 4),
                    'labels': labels.view(-1)
                }
                
                # Flatten outputs
                loss_outputs = {
                    'bbox_preds': outputs['bbox_preds'].view(-1, 4),
                    'cls_preds': outputs['cls_preds'].view(-1, model_cfg['NUM_CLASSES'])
                }
                
                # Compute loss
                loss_dict = self.criterion(loss_outputs, targets)
                loss = loss_dict['total_loss']
                
                loss_meter.update(loss.item(), n=images.size(0))
        
        return loss_meter.avg
    
    def train(self, train_loader, val_loader, num_epochs):
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
        """
        if is_main_process():
            print(f"\nStarting training for {num_epochs} epochs...\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            if is_main_process():
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_path = self.checkpoint_dir / 'best_model.pt'
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✓ Saved best model!")
                
                # Save periodic checkpoint
                if epoch % self.config['CHECKPOINT']['SAVE_INTERVAL'] == 0:
                    save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✓ Saved checkpoint!")
        
        if is_main_process():
            print(f"\n✅ Training complete!")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description='Train tiny object detection model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--train-data', type=str, default='data/train',
                       help='Training data directory')
    parser.add_argument('--train-annot', type=str, default='data/train_annotations.txt',
                       help='Training annotations file')
    parser.add_argument('--val-data', type=str, default='data/val',
                       help='Validation data directory')
    parser.add_argument('--val-annot', type=str, default='data/val_annotations.txt',
                       help='Validation annotations file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs:
        config['TRAIN']['EPOCHS'] = args.epochs
    
    # Setup device
    device = torch.device(config['DEVICE'] if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = TinyObjectDataset(
        image_dir=args.train_data,
        annotations_file=args.train_annot,
        transforms=TinyObjectAugmentation.get_train_transforms(
            image_size=config['MODEL']['IMAGE_SIZE']
        )
    )
    
    val_dataset = TinyObjectDataset(
        image_dir=args.val_data,
        annotations_file=args.val_annot,
        transforms=TinyObjectAugmentation.get_val_transforms(
            image_size=config['MODEL']['IMAGE_SIZE']
        )
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['TRAIN']['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['DATA']['NUM_WORKERS'],
        pin_memory=config['DATA']['PIN_MEMORY']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['TRAIN']['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['DATA']['NUM_WORKERS'],
        pin_memory=config['DATA']['PIN_MEMORY']
    )
    
    # Initialize trainer
    trainer = Trainer(config, device)
    
    # Train
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['TRAIN']['EPOCHS']
    )


if __name__ == '__main__':
    main()