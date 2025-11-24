"""
Main training and testing script for HyperIQA.

This script handles command-line argument parsing, dataset configuration,
and runs the training/testing loop for multiple random train-test splits.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

import numpy as np

from HyerIQASolver import HyperIQASolver

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Train/test split ratio
TRAIN_RATIO = 0.8

# Default dataset paths (can be overridden via environment variables or config)
DEFAULT_DATASET_PATHS = {
    "live": "/home/ssl/Database/databaserelease2/",
    "csiq": "/home/ssl/Database/CSIQ/",
    "tid2013": "/home/ssl/Database/TID2013/",
    "livec": "/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/",
    "koniq-10k": "/home/ssl/Database/koniq-10k/",
    "bid": "/home/ssl/Database/BID/",
}

# Number of reference/source images in each dataset
DATASET_IMAGE_COUNTS = {
    "live": 29,
    "csiq": 30,
    "tid2013": 25,
    "livec": 1162,
    "koniq-10k": 10073,
    "bid": 586,
}


def get_train_test_split(
    image_indices: list[int],
    train_ratio: float = TRAIN_RATIO,
) -> tuple[list[int], list[int]]:
    """
    Randomly split image indices into train and test sets.

    Args:
        image_indices: List of image indices to split.
        train_ratio: Fraction of images for training.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    indices = image_indices.copy()
    random.shuffle(indices)

    split_point = int(round(train_ratio * len(indices)))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    return train_indices, test_indices


def main(config: Any) -> None:
    """
    Main training and testing loop.

    Args:
        config: Configuration namespace from argparse.
    """
    dataset_path = DEFAULT_DATASET_PATHS.get(config.dataset)
    if dataset_path is None:
        raise ValueError(
            f"Unknown dataset: {config.dataset}. "
            f"Supported: {list(DEFAULT_DATASET_PATHS.keys())}"
        )

    image_count = DATASET_IMAGE_COUNTS[config.dataset]
    image_indices = list(range(image_count))

    srcc_results = np.zeros(config.train_test_num, dtype=np.float64)
    plcc_results = np.zeros(config.train_test_num, dtype=np.float64)

    print(
        f"Training and testing on {config.dataset} dataset "
        f"for {config.train_test_num} rounds..."
    )

    for round_idx in range(config.train_test_num):
        print(f"Round {round_idx + 1}")

        train_indices, test_indices = get_train_test_split(image_indices)

        solver = HyperIQASolver(
            config,
            dataset_path,
            train_indices,
            test_indices,
        )
        srcc_results[round_idx], plcc_results[round_idx] = solver.train()

    srcc_median = np.median(srcc_results)
    plcc_median = np.median(plcc_results)

    print(f"Testing median SRCC {srcc_median:.4f},\tmedian PLCC {plcc_median:.4f}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train and test HyperIQA on various IQA datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="livec",
        choices=list(DEFAULT_DATASET_PATHS.keys()),
        help="Dataset to use for training and testing",
    )
    parser.add_argument(
        "--train_patch_num",
        type=int,
        default=25,
        help="Number of patches to sample from each training image",
    )
    parser.add_argument(
        "--test_patch_num",
        type=int,
        default=25,
        help="Number of patches to sample from each testing image",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Base learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--lr_ratio",
        type=int,
        default=10,
        help="Learning rate multiplier for hyper network layers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=16,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Size of image patches for training and testing",
    )
    parser.add_argument(
        "--train_test_num",
        type=int,
        default=10,
        help="Number of train-test rounds with different random splits",
    )

    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    main(config)
