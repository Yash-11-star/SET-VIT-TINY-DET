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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import TinyObjectDataset, TinyObjectAugmentation
from datasets.collate import collate_variable_boxes
from models import TinyObjectViT, TinyObjectLoss
from utils import AverageMeter, ProgressMeter, is_main_process


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
            # Get images and move to device
            images = batch['images'].to(self.device)
            
            # Handle annotations (already as lists)
            boxes = [b.to(self.device) for b in batch['boxes']]
            labels = [l.to(self.device) for l in batch['labels']]
            is_valid = [v.to(self.device) for v in batch['is_valid']]
            
            # Forward pass
            outputs = self.model(images)
            
            # Prepare targets - apply is_valid per image
            valid_boxes = []
            valid_labels = []
            for b, l, v in zip(boxes, labels, is_valid):
                if v.any():
                    valid_boxes.append(b[v])
                    valid_labels.append(l[v])
            
            targets = {
                'boxes': valid_boxes if valid_boxes else [torch.zeros(0, 4).to(self.device)],
                'labels': valid_labels if valid_labels else [torch.zeros(0, dtype=torch.long).to(self.device)]
            }
            
            loss_outputs = outputs  # Already in the right format
            
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
                images = batch['images'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                labels = batch['labels'].to(self.device)
                is_valid = batch['is_valid'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Prepare targets - only consider valid boxes
                valid_boxes = []
                valid_labels = []
                for i in range(len(boxes)):
                    valid_boxes.append(boxes[i][is_valid[i]])
                    valid_labels.append(labels[i][is_valid[i]])
                
                targets = {
                    'boxes': torch.cat(valid_boxes) if valid_boxes else torch.zeros(0, 4),
                    'labels': torch.cat(valid_labels) if valid_labels else torch.zeros(0, dtype=torch.long)
                }
                
                # Get predictions corresponding to each patch and convert to boxes/classes
                bbox_preds = outputs['bbox_preds'].view(-1, 4)
                cls_preds = outputs['cls_preds'].view(-1, model_cfg['NUM_CLASSES'])
                
                # Prepare loss inputs
                loss_outputs = {
                    'bbox_preds': bbox_preds,
                    'cls_preds': cls_preds
                }
                
                # Compute loss
                loss_dict = self.criterion(loss_outputs, targets)
                loss = loss_dict['total_loss']
                
                loss_meter.update(loss.item(), n=images.size(0))
        
        return loss_meter.avg
    
    def train(self, train_loader, val_loader, num_epochs):
            self.model.train()
            meter = AverageMeter('Loss', ':.4f')
            prog = ProgressMeter(len(train_loader), [meter], prefix=f'Epoch [{epoch}]')
        
            for batch_idx, batch in enumerate(train_loader):
            # Update learning rate
            if is_main_process():
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                
                # Save best model
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


                loss_dict = self.criterion(outputs, targets)


    """Main training entry point"""
    parser = argparse.ArgumentParser(description='Train tiny object detection model')
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['TRAIN']['GRADIENT_CLIP'])
    parser.add_argument('--train-annot', type=str, default='data/train_annotations.txt',
                       help='Training annotations file')
                meter.update(loss.item(), n=images.size(0))
                if (batch_idx + 1) % self.config['TRAIN']['PRINT_INTERVAL'] == 0 and is_main_process():
                    prog.display(batch_idx + 1)
                       help='Number of epochs (overrides config)')
            return meter.avg
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
        pin_memory=config['DATA']['PIN_MEMORY'],
        collate_fn=collate_variable_boxes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['TRAIN']['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['DATA']['NUM_WORKERS'],
        pin_memory=config['DATA']['PIN_MEMORY'],
        collate_fn=collate_variable_boxes
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