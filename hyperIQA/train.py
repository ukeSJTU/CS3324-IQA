"""
Training script for HyperIQA on KonIQ dataset.
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.utils.data
from scipy import stats

import models
from dataset import IQADataset
from logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train HyperIQA on KonIQ dataset')

    # Data parameters
    parser.add_argument('--dataset_root', type=str, default='../datasets/',
                        help='Root directory containing image folders')
    parser.add_argument('--train_json', type=str, default='../datasets/metas/koniq_train.json',
                        help='Path to training JSON metadata')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1 = 10%%)')

    # Training parameters
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate for backbone')
    parser.add_argument('--lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hypernet (hypernet_lr = lr * lr_ratio)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=16,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

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
        # Auto-generate folder name with key hyperparameters
        folder_name = f"lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_val{args.val_split}_seed{args.seed}"
        checkpoint_dir = os.path.join('../checkpoints', folder_name)
    else:
        checkpoint_dir = args.checkpoint_dir

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save all arguments to JSON for reference
    args_dict = vars(args)
    args_dict['checkpoint_dir'] = checkpoint_dir  # Update with actual path
    with open(os.path.join(checkpoint_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=2)

    return checkpoint_dir


def get_train_val_loaders(args, logger):
    """Create train and validation data loaders"""
    # Load full training dataset
    full_dataset = IQADataset(
        dataset_root=args.dataset_root,
        json_path=args.train_json,
        patch_size=args.patch_size,
        patch_num=args.train_patch_num,
        is_train=True,
        resize_size=(512, 384)
    )

    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Dataset split: {train_size} train, {val_size} val (ratio: {1-args.val_split:.2f}/{args.val_split:.2f})")

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


def train_one_epoch(model_hyper, train_loader, optimizer, criterion, epoch, logger):
    """Train for one epoch"""
    model_hyper.train()
    epoch_loss = []
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
        loss = criterion(pred.squeeze(), labels.float())
        epoch_loss.append(loss.item())

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # Calculate metrics
    train_loss = np.mean(epoch_loss)
    train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

    return train_loss, train_srcc


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

    logger.info("="*60)
    logger.info("Training HyperIQA")
    logger.info("="*60)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Arguments: {vars(args)}")

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader = get_train_val_loaders(args, logger)

    # Initialize model
    logger.info("Initializing model...")
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.train()

    # Setup optimizer
    criterion = torch.nn.L1Loss().cuda()

    backbone_params = list(map(id, model_hyper.res.parameters()))
    hypernet_params = filter(lambda p: id(p) not in backbone_params, model_hyper.parameters())

    paras = [
        {'params': hypernet_params, 'lr': args.lr * args.lr_ratio},
        {'params': model_hyper.res.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(paras, weight_decay=args.weight_decay)

    # Training loop
    logger.info("Starting training...")
    logger.info(f"{'Epoch':<8}{'Train_Loss':<15}{'Train_SRCC':<15}{'Val_SRCC':<15}{'Val_PLCC':<15}")
    logger.info("-" * 67)

    best_val_srcc = 0.0
    metrics_list = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_srcc = train_one_epoch(
            model_hyper, train_loader, optimizer, criterion, epoch, logger
        )

        # Validate
        val_srcc, val_plcc = validate(model_hyper, val_loader, args.val_patch_num, logger)

        # Log metrics
        logger.info(f"{epoch:<8}{train_loss:<15.4f}{train_srcc:<15.4f}{val_srcc:<15.4f}{val_plcc:<15.4f}")

        # Save metrics
        metrics_list.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_srcc": float(train_srcc),
            "val_srcc": float(val_srcc),
            "val_plcc": float(val_plcc)
        })
        save_metrics(checkpoint_dir, metrics_list)

        # Save checkpoint
        is_best = val_srcc > best_val_srcc
        if is_best:
            best_val_srcc = val_srcc
            logger.success(f"New best model! Val SRCC: {val_srcc:.4f}")

        if args.save_all_epochs or is_best:
            save_checkpoint(model_hyper, checkpoint_dir, epoch, is_best)

        # Update learning rate (same schedule as original)
        lr = args.lr / pow(10, (epoch // 6))
        lr_ratio = 1 if epoch > 8 else args.lr_ratio

        paras = [
            {'params': hypernet_params, 'lr': lr * lr_ratio},
            {'params': model_hyper.res.parameters(), 'lr': args.lr}
        ]
        optimizer = torch.optim.Adam(paras, weight_decay=args.weight_decay)

    logger.info("="*60)
    logger.success(f"Training completed! Best Val SRCC: {best_val_srcc:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
