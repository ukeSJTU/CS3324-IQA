"""
Training script for HyperIQA model.

This script trains the HyperIQA model on the KonIQ dataset with automatic train/validation split.
It provides comprehensive logging including loss curves, SRCC/PLCC metrics, and model checkpointing.

Usage:
    python train.py --train_json ../datasets/metas/koniq_train.json \\
                    --dataset_root ../datasets \\
                    --val_split 0.2 \\
                    --epochs 16 \\
                    --batch_size 96 \\
                    --lr 2e-5 \\
                    --save_dir ./checkpoints
"""

import argparse
import json
import os
import random
from datetime import datetime

# Import HyperIQA components
import models
import numpy as np
import torch
from json_dataset import JSONImageDataset, get_transforms
from scipy import stats


class HyperIQATrainer:
    """Trainer class for HyperIQA model with train/validation split."""
    
    def __init__(self, config):
        self.config = config
        self.epochs = config.epochs
        self.train_patch_num = config.train_patch_num
        self.val_patch_num = config.val_patch_num
        self.save_dir = config.save_dir
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize model
        self.model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model.train(True)
        
        # Loss function
        self.l1_loss = torch.nn.L1Loss().cuda()
        
        # Setup optimizer with different learning rates for backbone and hypernet
        backbone_params = list(map(id, self.model.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, 
                                     self.model.parameters())
        
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        
        paras = [
            {'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
            {'params': self.model.res.parameters(), 'lr': self.lr}
        ]
        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_srcc': [],
            'val_srcc': [],
            'val_plcc': [],
            'epoch': []
        }
        
        self.best_val_srcc = 0.0
        self.best_val_plcc = 0.0
        self.best_epoch = 0
    
    def _setup_data_loaders(self):
        """Setup train and validation data loaders with automatic splitting."""
        config = self.config
        
        # Load full training JSON
        with open(config.train_json, 'r') as f:
            all_data = json.load(f)
        
        # Shuffle and split into train/val
        random.seed(config.seed)
        random.shuffle(all_data)
        
        val_size = int(len(all_data) * config.val_split)
        val_data = all_data[:val_size]
        train_data = all_data[val_size:]
        
        print(f"Dataset split: {len(train_data)} train, {len(val_data)} validation")
        
        # Create temporary JSON files for splits
        train_json_path = os.path.join(self.save_dir, 'train_split.json')
        val_json_path = os.path.join(self.save_dir, 'val_split.json')
        
        with open(train_json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(val_json_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        # Create datasets
        train_transform = get_transforms(patch_size=config.patch_size, is_train=True)
        val_transform = get_transforms(patch_size=config.patch_size, is_train=False)
        
        train_dataset = JSONImageDataset(
            json_path=train_json_path,
            root_dir=config.dataset_root,
            transform=train_transform,
            patch_num=self.train_patch_num
        )
        
        val_dataset = JSONImageDataset(
            json_path=val_json_path,
            root_dir=config.dataset_root,
            transform=val_transform,
            patch_num=self.val_patch_num
        )
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train(True)
        epoch_loss = []
        pred_scores = []
        gt_scores = []
        
        for batch_idx, (img, label) in enumerate(self.train_loader):
            img = img.cuda()
            label = label.cuda()
            
            self.optimizer.zero_grad()
            
            # Generate weights for target network
            paras = self.model(img)
            
            # Build target network
            model_target = models.TargetNet(paras).cuda()
            for param in model_target.parameters():
                param.requires_grad = False
            
            # Quality prediction
            pred = model_target(paras['target_in_vec'])
            pred_scores.extend(pred.cpu().detach().tolist())
            gt_scores.extend(label.cpu().tolist())
            
            # Compute loss and backpropagate
            loss = self.l1_loss(pred.squeeze(), label.float().detach())
            epoch_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        
        return avg_loss, train_srcc
    
    def validate(self):
        """Validate the model."""
        self.model.train(False)
        pred_scores = []
        gt_scores = []
        
        with torch.no_grad():
            for img, label in self.val_loader:
                img = img.cuda()
                label = label.cuda()
                
                # Generate weights and predict
                paras = self.model(img)
                model_target = models.TargetNet(paras).cuda()
                model_target.train(False)
                pred = model_target(paras['target_in_vec'])
                
                pred_scores.append(float(pred.item()))
                gt_scores.extend(label.cpu().tolist())
        
        # Average predictions over patches
        pred_scores = np.mean(np.reshape(np.array(pred_scores), 
                                        (-1, self.val_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), 
                                      (-1, self.val_patch_num)), axis=1)
        
        val_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        val_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        
        self.model.train(True)
        return val_srcc, val_plcc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_srcc': self.best_val_srcc,
            'best_val_plcc': self.best_val_plcc,
            'history': self.history,
            'config': vars(self.config)
        }
        
        # Save latest checkpoint
        save_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, save_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint (SRCC: {self.best_val_srcc:.4f}, PLCC: {self.best_val_plcc:.4f})")
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def update_learning_rate(self, epoch):
        """Update learning rate based on epoch."""
        lr = self.lr / pow(10, (epoch // 6))
        if epoch > 8:
            self.lrratio = 1
        
        paras = [
            {'params': self.hypernet_params, 'lr': lr * self.lrratio},
            {'params': self.model.res.parameters(), 'lr': self.lr}
        ]
        self.optimizer = torch.optim.Adam(paras, weight_decay=self.weight_decay)
    
    def train(self):
        """Main training loop."""
        print("="*80)
        print("Starting HyperIQA Training")
        print("="*80)
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.lr}")
        print(f"Save directory: {self.save_dir}")
        print("="*80)
        print()
        
        print("Epoch\tTrain_Loss\tTrain_SRCC\tVal_SRCC\tVal_PLCC")
        print("-"*80)
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Train for one epoch
            train_loss, train_srcc = self.train_epoch(epoch)
            
            # Validate
            val_srcc, val_plcc = self.validate()
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_srcc'].append(train_srcc)
            self.history['val_srcc'].append(val_srcc)
            self.history['val_plcc'].append(val_plcc)
            
            # Print metrics
            print(f"{epoch + 1}\t{train_loss:.4f}\t\t{train_srcc:.4f}\t\t{val_srcc:.4f}\t\t{val_plcc:.4f}")
            
            # Save best model
            is_best = False
            if val_srcc > self.best_val_srcc:
                self.best_val_srcc = val_srcc
                self.best_val_plcc = val_plcc
                self.best_epoch = epoch + 1
                is_best = True
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Update learning rate
            self.update_learning_rate(epoch)
            
            print()
        
        # Save training history
        self.save_training_history()
        
        print("="*80)
        print("Training Completed!")
        print(f"Best Validation SRCC: {self.best_val_srcc:.4f} at epoch {self.best_epoch}")
        print(f"Best Validation PLCC: {self.best_val_plcc:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Train HyperIQA model on KonIQ dataset')
    
    # Dataset arguments
    parser.add_argument('--train_json', type=str, 
                       default='../datasets/metas/koniq_train.json',
                       help='Path to training JSON file')
    parser.add_argument('--dataset_root', type=str, 
                       default='../datasets',
                       help='Root directory for dataset images')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=16,
                       help='Number of training epochs (default: 16)')
    parser.add_argument('--batch_size', type=int, default=96,
                       help='Batch size for training (default: 96)')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--lr_ratio', type=int, default=10,
                       help='Learning rate ratio for hyper network (default: 10)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay (default: 5e-4)')
    
    # Data arguments
    parser.add_argument('--patch_size', type=int, default=224,
                       help='Patch size for cropping (default: 224)')
    parser.add_argument('--train_patch_num', type=int, default=25,
                       help='Number of patches per training image (default: 25)')
    parser.add_argument('--val_patch_num', type=int, default=25,
                       help='Number of patches per validation image (default: 25)')
    
    # Other arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    config = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # Create trainer and start training
    trainer = HyperIQATrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
