"""
Solver for training and testing HyperIQA model.

This module provides the training and evaluation logic for the
HyperNetwork-based Image Quality Assessment model.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.tensorboard import SummaryWriter

import data_loader
import models

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# Learning rate schedule parameters
LR_DECAY_INTERVAL = 6
LR_RATIO_CHANGE_EPOCH = 8


def setup_logger(
    name: str,
    log_dir: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files. If None, only console output.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class HyperIQASolver:
    """
    Solver for training and testing HyperIQA.

    Handles model initialization, training loop, validation,
    learning rate scheduling, and experiment tracking.
    """

    def __init__(
        self,
        config: Any,
        path: str,
        train_idx: list[int],
        test_idx: list[int],
        exp_name: str | None = None,
        output_dir: str = "./experiments",
    ) -> None:
        """
        Initialize the solver.

        Args:
            config: Configuration object with training hyperparameters.
            path: Path to the dataset.
            train_idx: Indices for training split.
            test_idx: Indices for test split.
            exp_name: Experiment name for logging. Auto-generated if None.
            output_dir: Base directory for experiment outputs.
        """
        self.config = config
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.lr = config.lr
        self.lr_ratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        # Set up experiment directory
        if exp_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{config.dataset}_{timestamp}"

        self.exp_dir = os.path.join(output_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Set up logging
        self.logger = setup_logger("HyperIQA", self.exp_dir)
        self.logger.info(f"Experiment directory: {self.exp_dir}")

        # Set up TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))

        # Save experiment config
        self._save_config()

        # Initialize model
        # HyperNet architecture: lda_out=16, hyper_in=112, target_in=224, fc_sizes=[112,56,28,14], feat=7
        self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_hyper.train(True)

        self.l1_loss = nn.L1Loss().cuda()

        # Set up optimizer with different learning rates for backbone and hypernet
        self._setup_optimizer()

        # Initialize data loaders
        train_loader = data_loader.DataLoader(
            config.dataset,
            path,
            train_idx,
            config.patch_size,
            config.train_patch_num,
            batch_size=config.batch_size,
            is_train=True,
        )
        test_loader = data_loader.DataLoader(
            config.dataset,
            path,
            test_idx,
            config.patch_size,
            config.test_patch_num,
            is_train=False,
        )
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        self.logger.info(f"Training samples: {len(train_loader.data)}")
        self.logger.info(f"Testing samples: {len(test_loader.data)}")

        # Track best results
        self.best_srcc = 0.0
        self.best_plcc = 0.0
        self.best_epoch = 0
        self.global_step = 0

    def _save_config(self) -> None:
        """Save experiment configuration to JSON file."""
        config_dict = {
            "dataset": self.config.dataset,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lr": self.config.lr,
            "lr_ratio": self.config.lr_ratio,
            "weight_decay": self.config.weight_decay,
            "patch_size": self.config.patch_size,
            "train_patch_num": self.config.train_patch_num,
            "test_patch_num": self.config.test_patch_num,
        }
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        self.logger.info(f"Config saved to {config_path}")

    def _setup_optimizer(self, lr: float | None = None) -> None:
        """
        Set up the Adam optimizer with parameter groups.

        Args:
            lr: Learning rate override. Uses self.lr if None.
        """
        if lr is None:
            lr = self.lr

        backbone_param_ids = set(map(id, self.model_hyper.backbone.parameters()))
        hypernet_params = [
            p for p in self.model_hyper.parameters()
            if id(p) not in backbone_param_ids
        ]

        param_groups = [
            {"params": hypernet_params, "lr": lr * self.lr_ratio},
            {"params": self.model_hyper.backbone.parameters(), "lr": lr},
        ]
        self.optimizer = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """
        Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, training_srcc).
        """
        epoch_losses: list[float] = []
        pred_scores: list[float] = []
        gt_scores: list[float] = []

        for batch_idx, (img, label) in enumerate(self.train_data):
            img = img.cuda()
            label = label.cuda()

            self.optimizer.zero_grad()

            # Generate weights for target network
            params = self.model_hyper(img)

            # Build target network with generated weights
            model_target = models.TargetNet(params).cuda()
            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            pred = model_target(params["target_in_vec"])
            pred_scores.extend(pred.cpu().tolist())
            gt_scores.extend(label.cpu().tolist())

            loss = self.l1_loss(pred.squeeze(), label.float().detach())
            epoch_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            # Log batch loss to TensorBoard
            self.writer.add_scalar("Loss/batch", loss.item(), self.global_step)
            self.global_step += 1

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

        return avg_loss, float(train_srcc)

    def _update_learning_rate(self, epoch: int) -> None:
        """
        Update learning rate based on epoch.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        lr = self.lr / pow(10, epoch // LR_DECAY_INTERVAL)

        if epoch > LR_RATIO_CHANGE_EPOCH:
            self.lr_ratio = 1

        self._setup_optimizer(lr)

        # Log learning rate
        self.writer.add_scalar("LR/backbone", lr, epoch)
        self.writer.add_scalar("LR/hypernet", lr * self.lr_ratio, epoch)

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model_hyper.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_srcc": self.best_srcc,
            "best_plcc": self.best_plcc,
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.exp_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.exp_dir, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved at epoch {epoch + 1}")

    def train(self) -> tuple[float, float]:
        """
        Run the full training loop.

        Returns:
            Tuple of (best_srcc, best_plcc) on test set.
        """
        self.logger.info("Starting training...")
        self.logger.info("Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC")

        for epoch in range(self.epochs):
            avg_loss, train_srcc = self._train_epoch(epoch)
            test_srcc, test_plcc = self.test(self.test_data)

            is_best = test_srcc > self.best_srcc
            if is_best:
                self.best_srcc = test_srcc
                self.best_plcc = test_plcc
                self.best_epoch = epoch

            # Log to console and file
            self.logger.info(
                f"{epoch + 1}\t{avg_loss:.3f}\t\t{train_srcc:.4f}\t\t"
                f"{test_srcc:.4f}\t\t{test_plcc:.4f}"
            )

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
            self.writer.add_scalar("SRCC/train", train_srcc, epoch)
            self.writer.add_scalar("SRCC/test", test_srcc, epoch)
            self.writer.add_scalar("PLCC/test", test_plcc, epoch)

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Update learning rate
            self._update_learning_rate(epoch)

        self.logger.info(
            f"Training completed. Best SRCC: {self.best_srcc:.6f}, "
            f"PLCC: {self.best_plcc:.6f} at epoch {self.best_epoch + 1}"
        )

        # Save final results
        results = {
            "best_srcc": float(self.best_srcc),
            "best_plcc": float(self.best_plcc),
            "best_epoch": self.best_epoch + 1,
        }
        results_path = os.path.join(self.exp_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.writer.close()

        return self.best_srcc, self.best_plcc

    def test(self, data: DataLoader) -> tuple[float, float]:
        """
        Evaluate the model on a dataset.

        Args:
            data: DataLoader for evaluation.

        Returns:
            Tuple of (srcc, plcc) correlation coefficients.
        """
        self.model_hyper.train(False)
        pred_scores: list[float] = []
        gt_scores: list[float] = []

        with torch.no_grad():
            for img, label in data:
                img = img.cuda()
                label = label.cuda()

                params = self.model_hyper(img)
                model_target = models.TargetNet(params).cuda()
                model_target.train(False)
                pred = model_target(params["target_in_vec"])

                pred_scores.append(float(pred.item()))
                gt_scores.extend(label.cpu().tolist())

        # Average predictions across patches
        pred_array = np.mean(
            np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1
        )
        gt_array = np.mean(
            np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1
        )

        test_srcc, _ = stats.spearmanr(pred_array, gt_array)
        test_plcc, _ = stats.pearsonr(pred_array, gt_array)

        self.model_hyper.train(True)

        return float(test_srcc), float(test_plcc)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model_hyper.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_srcc = checkpoint.get("best_srcc", 0.0)
        self.best_plcc = checkpoint.get("best_plcc", 0.0)
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
