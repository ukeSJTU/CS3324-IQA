"""
Training script for MS-HyperIQA (Multi-Scale HyperIQA with Feature Pyramid and Attention)

Enhanced training with:
- Improved data augmentation
- Combined loss function (L1 + Rank loss)
- Cosine annealing learning rate schedule
- Better regularization
"""

import os
import json
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import torch.utils.data
from scipy import stats
import torch.nn.functional as F

import models
from dataset import IQADataset
from logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train MS-HyperIQA on KonIQ dataset')

    # Data parameters
    parser.add_argument('--dataset_root', type=str, default='../datasets/',
                        help='Root directory containing image folders')
    parser.add_argument('--train_json', type=str, default='../datasets/metas/koniq_train.json',
                        help='Path to training JSON metadata')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1 = 10%%)')

    # Training parameters
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Initial learning rate for backbone')
    parser.add_argument('--lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hypernet (hypernet_lr = lr * lr_ratio)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (increased from 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Loss function parameters
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['l1', 'combined'],
                        help='Loss function type: l1 or combined (L1 + Rank)')
    parser.add_argument('--rank_loss_weight', type=float, default=0.3,
                        help='Weight for rank loss in combined loss')

    # Learning rate schedule
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['step', 'cosine'],
                        help='Learning rate schedule: step or cosine')

    # Patch parameters
    parser.add_argument('--patch_size', type=int, default=224,
                        help='Size of image patches')
    parser.add_argument('--train_patch_num', type=int, default=25,
                        help='Number of patches to sample per training image')
    parser.add_argument('--val_patch_num', type=int, default=25,
                        help='Number of patches to sample per validation image')

    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Custom checkpoint directory (auto-generated if not specified)')
    parser.add_argument('--save_all_epochs', action='store_true',
                        help='Save checkpoint for every epoch (default: only save best)')

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_checkpoint_dir(args):
    """Create checkpoint directory with auto-generated name"""
    if args.checkpoint_dir is None:
        # Auto-generate folder name with key hyperparameters + timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"ms_hyper_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_{args.loss_type}_{timestamp}"
        checkpoint_dir = os.path.join('../checkpoints', folder_name)
    else:
        checkpoint_dir = args.checkpoint_dir

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save all arguments to JSON for reference
    args_dict = vars(args)
    args_dict['checkpoint_dir'] = checkpoint_dir
    with open(os.path.join(checkpoint_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    return checkpoint_dir


def get_train_val_loaders(args, logger):
    """Create train and validation data loaders with enhanced augmentation"""
    # Load JSON metadata
    with open(args.train_json, 'r') as f:
        data = json.load(f)

    # Split at IMAGE level
    num_images = len(data)
    val_num_images = int(num_images * args.val_split)
    train_num_images = num_images - val_num_images

    # Shuffle images with seed for reproducibility
    rng = random.Random(args.seed)
    rng.shuffle(data)

    train_data = data[:train_num_images]
    val_data = data[train_num_images:]

    logger.info(f"Dataset split: {train_num_images} train images, {val_num_images} val images")

    # Save split data
    train_split_json = os.path.join(os.path.dirname(args.train_json), '.tmp_train_split_ms.json')
    val_split_json = os.path.join(os.path.dirname(args.train_json), '.tmp_val_split_ms.json')

    with open(train_split_json, 'w') as f:
        json.dump(train_data, f)
    with open(val_split_json, 'w') as f:
        json.dump(val_data, f)

    # Create datasets (enhanced augmentation is now in dataset.py)
    train_dataset = IQADataset(
        dataset_root=args.dataset_root,
        json_path=train_split_json,
        patch_size=args.patch_size,
        patch_num=args.train_patch_num,
        is_train=True,
        resize_size=(512, 384)
    )

    val_dataset = IQADataset(
        dataset_root=args.dataset_root,
        json_path=val_split_json,
        patch_size=args.patch_size,
        patch_num=args.val_patch_num,
        is_train=False,
        resize_size=(512, 384)
    )

    logger.info(f"Total samples: {len(train_dataset)} train, {len(val_dataset)} val")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def rank_loss(pred, labels, margin=0.5):
    """
    Pairwise ranking loss to preserve relative quality ordering

    Args:
        pred: Predicted quality scores (batch_size,)
        labels: Ground truth quality scores (batch_size,)
        margin: Margin for ranking loss
    """
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0).cuda()

    # Expand predictions and labels to compute pairwise differences
    pred_i = pred.unsqueeze(1).expand(n, n)
    pred_j = pred.unsqueeze(0).expand(n, n)
    label_i = labels.unsqueeze(1).expand(n, n)
    label_j = labels.unsqueeze(0).expand(n, n)

    # Compute signs: +1 if i should rank higher than j, -1 otherwise
    sign = torch.sign(label_i - label_j)

    # Ranking loss: max(0, -sign * (pred_i - pred_j) + margin)
    loss = F.relu(-sign * (pred_i - pred_j) + margin)

    # Only consider pairs where labels differ significantly
    mask = torch.abs(label_i - label_j) > 0.1
    loss = loss * mask.float()

    # Average over valid pairs
    num_pairs = mask.float().sum()
    if num_pairs > 0:
        loss = loss.sum() / num_pairs
    else:
        loss = torch.tensor(0.0).cuda()

    return loss


def combined_loss(pred, labels, rank_weight=0.3):
    """
    Combined loss: L1 + weighted rank loss

    Args:
        pred: Predicted scores
        labels: Ground truth scores
        rank_weight: Weight for rank loss component
    """
    l1 = F.l1_loss(pred.squeeze(), labels.float())
    rank = rank_loss(pred.squeeze(), labels.float())
    return l1 + rank_weight * rank, l1, rank


def train_one_epoch(model_hyper, train_loader, optimizer, criterion_type, rank_weight, epoch, logger):
    """Train for one epoch"""
    model_hyper.train()
    epoch_loss = []
    epoch_l1_loss = []
    epoch_rank_loss = []
    pred_scores = []
    gt_scores = []

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        # Generate weights for target network
        paras = model_hyper(imgs)

        # Build target network
        model_target = models.TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        pred = model_target(paras['target_in_vec'])
        pred_scores.extend(pred.cpu().tolist())
        gt_scores.extend(labels.cpu().tolist())

        # Compute loss
        if criterion_type == 'combined':
            loss, l1, rank = combined_loss(pred, labels, rank_weight)
            epoch_l1_loss.append(l1.item())
            epoch_rank_loss.append(rank.item())
        else:  # l1 only
            loss = F.l1_loss(pred.squeeze(), labels.float())
            epoch_l1_loss.append(loss.item())
            epoch_rank_loss.append(0.0)

        epoch_loss.append(loss.item())

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            if criterion_type == 'combined':
                logger.debug(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f} (L1: {l1.item():.4f}, Rank: {rank.item():.4f})")
            else:
                logger.debug(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # Calculate metrics
    train_loss = np.mean(epoch_loss)
    train_l1 = np.mean(epoch_l1_loss)
    train_rank = np.mean(epoch_rank_loss)
    train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

    return train_loss, train_l1, train_rank, train_srcc


def validate(model_hyper, val_loader, val_patch_num, logger):
    """Validate the model"""
    model_hyper.eval()
    pred_scores = []
    gt_scores = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()

            # Generate weights for target network
            paras = model_hyper(imgs)

            # Build target network
            model_target = models.TargetNet(paras).cuda()
            model_target.eval()

            # Quality prediction
            pred = model_target(paras['target_in_vec'])
            pred_scores.append(float(pred.item()))
            gt_scores.extend(labels.cpu().tolist())

    # Average predictions over patches
    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, val_patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, val_patch_num)), axis=1)

    # Calculate metrics
    val_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    val_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    return val_srcc, val_plcc


def save_checkpoint(model_hyper, checkpoint_dir, epoch, is_best):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pkl')
    torch.save(model_hyper.state_dict(), checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pkl')
        torch.save(model_hyper.state_dict(), best_path)


def save_metrics(checkpoint_dir, metrics_list):
    """Save training metrics to JSON"""
    metrics_dict = {"epochs": metrics_list}
    metrics_path = os.path.join(checkpoint_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(args)

    # Setup logger
    log_file = os.path.join(checkpoint_dir, 'train.log')
    logger = setup_logger(log_file)

    logger.info("="*70)
    logger.info("Training MS-HyperIQA (Multi-Scale with FPN and Attention)")
    logger.info("="*70)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Arguments: {vars(args)}")

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader = get_train_val_loaders(args, logger)

    # Initialize model
    logger.info("Initializing MS-HyperIQA model...")
    model_hyper = models.MSHyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train()

    # Count parameters
    total_params = sum(p.numel() for p in model_hyper.parameters())
    trainable_params = sum(p.numel() for p in model_hyper.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Setup optimizer
    backbone_params = list(map(id, model_hyper.res.parameters()))
    hypernet_params = filter(lambda p: id(p) not in backbone_params, model_hyper.parameters())

    paras = [
        {'params': hypernet_params, 'lr': args.lr * args.lr_ratio},
        {'params': model_hyper.res.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(paras, weight_decay=args.weight_decay)

    # Setup learning rate scheduler
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/100)
        logger.info(f"Using Cosine Annealing LR schedule")
    else:
        scheduler = None
        logger.info(f"Using step LR schedule (original)")

    # Training loop
    logger.info("Starting training...")
    if args.loss_type == 'combined':
        logger.info(f"{'Epoch':<8}{'Train_Loss':<15}{'L1_Loss':<15}{'Rank_Loss':<15}{'Train_SRCC':<15}{'Val_SRCC':<15}{'Val_PLCC':<15}")
    else:
        logger.info(f"{'Epoch':<8}{'Train_Loss':<15}{'Train_SRCC':<15}{'Val_SRCC':<15}{'Val_PLCC':<15}")
    logger.info("-" * 100)

    best_val_srcc = 0.0
    metrics_list = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_l1, train_rank, train_srcc = train_one_epoch(
            model_hyper, train_loader, optimizer, args.loss_type, args.rank_loss_weight, epoch, logger
        )

        # Validate
        val_srcc, val_plcc = validate(model_hyper, val_loader, args.val_patch_num, logger)

        # Log metrics
        if args.loss_type == 'combined':
            logger.info(f"{epoch:<8}{train_loss:<15.4f}{train_l1:<15.4f}{train_rank:<15.4f}"
                       f"{train_srcc:<15.4f}{val_srcc:<15.4f}{val_plcc:<15.4f}")
        else:
            logger.info(f"{epoch:<8}{train_loss:<15.4f}{train_srcc:<15.4f}{val_srcc:<15.4f}{val_plcc:<15.4f}")

        # Save metrics
        metrics_entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_l1_loss": float(train_l1),
            "train_rank_loss": float(train_rank),
            "train_srcc": float(train_srcc),
            "val_srcc": float(val_srcc),
            "val_plcc": float(val_plcc)
        }
        metrics_list.append(metrics_entry)
        save_metrics(checkpoint_dir, metrics_list)

        # Save checkpoint
        is_best = val_srcc > best_val_srcc
        if is_best:
            best_val_srcc = val_srcc
            logger.success(f"New best model! Val SRCC: {val_srcc:.4f}, Val PLCC: {val_plcc:.4f}")

        if args.save_all_epochs or is_best:
            save_checkpoint(model_hyper, checkpoint_dir, epoch, is_best)

        # Update learning rate
        if args.lr_schedule == 'cosine':
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"Learning rate updated to: {current_lr:.6f}")
        else:
            # Original step schedule
            lr = args.lr / pow(10, (epoch // 6))
            lr_ratio = 1 if epoch > 8 else args.lr_ratio
            paras = [
                {'params': hypernet_params, 'lr': lr * lr_ratio},
                {'params': model_hyper.res.parameters(), 'lr': lr}
            ]
            optimizer = torch.optim.Adam(paras, weight_decay=args.weight_decay)

    logger.info("="*70)
    logger.success(f"Training completed! Best Val SRCC: {best_val_srcc:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
