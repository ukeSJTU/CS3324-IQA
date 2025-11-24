"""
Solver for training and testing HyperIQA model.

This module provides the training and evaluation logic for the
HyperNetwork-based Image Quality Assessment model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

import data_loader
import models

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# HyperNet architecture parameters
HYPERNET_LDA_OUT_CHANNELS = 16
HYPERNET_HYPER_IN_CHANNELS = 112
HYPERNET_TARGET_IN_SIZE = 224
HYPERNET_TARGET_FC1_SIZE = 112
HYPERNET_TARGET_FC2_SIZE = 56
HYPERNET_TARGET_FC3_SIZE = 28
HYPERNET_TARGET_FC4_SIZE = 14
HYPERNET_FEATURE_SIZE = 7

# Learning rate schedule parameters
LR_DECAY_INTERVAL = 6
LR_RATIO_CHANGE_EPOCH = 8


class HyperIQASolver:
    """
    Solver for training and testing HyperIQA.

    Handles model initialization, training loop, validation,
    and learning rate scheduling.
    """

    def __init__(
        self,
        config: Any,
        path: str,
        train_idx: list[int],
        test_idx: list[int],
    ) -> None:
        """
        Initialize the solver.

        Args:
            config: Configuration object with training hyperparameters.
            path: Path to the dataset.
            train_idx: Indices for training split.
            test_idx: Indices for test split.
        """
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.lr = config.lr
        self.lr_ratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        # Initialize model
        self.model_hyper = models.HyperNet(
            HYPERNET_LDA_OUT_CHANNELS,
            HYPERNET_HYPER_IN_CHANNELS,
            HYPERNET_TARGET_IN_SIZE,
            HYPERNET_TARGET_FC1_SIZE,
            HYPERNET_TARGET_FC2_SIZE,
            HYPERNET_TARGET_FC3_SIZE,
            HYPERNET_TARGET_FC4_SIZE,
            HYPERNET_FEATURE_SIZE,
        ).cuda()
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

    def _train_epoch(self) -> tuple[float, float]:
        """
        Run one training epoch.

        Returns:
            Tuple of (average_loss, training_srcc).
        """
        epoch_losses: list[float] = []
        pred_scores: list[float] = []
        gt_scores: list[float] = []

        for img, label in self.train_data:
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

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

        return avg_loss, train_srcc

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

    def train(self) -> tuple[float, float]:
        """
        Run the full training loop.

        Returns:
            Tuple of (best_srcc, best_plcc) on test set.
        """
        best_srcc = 0.0
        best_plcc = 0.0

        print("Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC")

        for epoch in range(self.epochs):
            avg_loss, train_srcc = self._train_epoch()
            test_srcc, test_plcc = self.test(self.test_data)

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc

            print(
                f"{epoch + 1}\t{avg_loss:.3f}\t\t{train_srcc:.4f}\t\t"
                f"{test_srcc:.4f}\t\t{test_plcc:.4f}"
            )

            self._update_learning_rate(epoch)

        print(f"Best test SRCC {best_srcc:.6f}, PLCC {best_plcc:.6f}")

        return best_srcc, best_plcc

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

        return test_srcc, test_plcc
