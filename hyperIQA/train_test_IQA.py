"""
Main training and testing script for HyperIQA.

This script handles command-line argument parsing, dataset configuration,
and runs the training/testing loop for multiple random train-test splits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Any

import numpy as np

from HyerIQASolver import HyperIQASolver

# GPU configuration (can be overridden via command line)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Train/test split ratio
TRAIN_RATIO = 0.8

# Default dataset paths
DEFAULT_DATASET_PATHS = {
    "live": "/home/ssl/Database/databaserelease2/",
    "csiq": "/home/ssl/Database/CSIQ/",
    "tid2013": "/home/ssl/Database/TID2013/",
    "livec": "/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/",
    "koniq-10k": "/home/ssl/Database/koniq-10k/",
    "bid": "/home/ssl/Database/BID/",
    # Cross-dataset testing
    "spaq": "/home/ssl/Database/SPAQ/",
    "kadid-10k": "/home/ssl/Database/KADID-10K/",
    "agiqa-3k": "/home/ssl/Database/AGIQA-3K/",
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


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def main(config: argparse.Namespace) -> None:
    """
    Main training and testing loop.

    Args:
        config: Configuration namespace from argparse.
    """
    # Set random seed if provided
    if config.seed is not None:
        set_random_seed(config.seed)

    # Get dataset path
    if config.data_path:
        dataset_path = config.data_path
    else:
        dataset_path = DEFAULT_DATASET_PATHS.get(config.dataset)
        if dataset_path is None:
            raise ValueError(
                f"Unknown dataset: {config.dataset}. "
                f"Please provide --data_path or use supported dataset."
            )

    image_count = DATASET_IMAGE_COUNTS.get(config.dataset)
    if image_count is None:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    image_indices = list(range(image_count))

    srcc_results = np.zeros(config.train_test_num, dtype=np.float64)
    plcc_results = np.zeros(config.train_test_num, dtype=np.float64)

    print(
        f"Training and testing on {config.dataset} dataset "
        f"for {config.train_test_num} rounds..."
    )
    print(f"Output directory: {config.output_dir}")

    # Create experiment base name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_base_name = config.exp_name or f"{config.dataset}_{timestamp}"

    for round_idx in range(config.train_test_num):
        print(f"\n{'='*50}")
        print(f"Round {round_idx + 1}/{config.train_test_num}")
        print(f"{'='*50}")

        train_indices, test_indices = get_train_test_split(image_indices)

        # Create experiment name for this round
        if config.train_test_num > 1:
            exp_name = f"{exp_base_name}_round{round_idx + 1}"
        else:
            exp_name = exp_base_name

        solver = HyperIQASolver(
            config,
            dataset_path,
            train_indices,
            test_indices,
            exp_name=exp_name,
            output_dir=config.output_dir,
        )
        srcc_results[round_idx], plcc_results[round_idx] = solver.train()

    # Compute statistics
    srcc_median = np.median(srcc_results)
    plcc_median = np.median(plcc_results)
    srcc_mean = np.mean(srcc_results)
    plcc_mean = np.mean(plcc_results)
    srcc_std = np.std(srcc_results)
    plcc_std = np.std(plcc_results)

    print(f"\n{'='*50}")
    print("Final Results")
    print(f"{'='*50}")
    print(f"SRCC: {srcc_mean:.4f} ± {srcc_std:.4f} (median: {srcc_median:.4f})")
    print(f"PLCC: {plcc_mean:.4f} ± {plcc_std:.4f} (median: {plcc_median:.4f})")

    # Save summary results
    summary = {
        "dataset": config.dataset,
        "train_test_num": config.train_test_num,
        "srcc_results": srcc_results.tolist(),
        "plcc_results": plcc_results.tolist(),
        "srcc_mean": float(srcc_mean),
        "srcc_std": float(srcc_std),
        "srcc_median": float(srcc_median),
        "plcc_mean": float(plcc_mean),
        "plcc_std": float(plcc_std),
        "plcc_median": float(plcc_median),
    }

    summary_path = os.path.join(config.output_dir, f"{exp_base_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train and test HyperIQA on various IQA datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--dataset",
        type=str,
        default="koniq-10k",
        choices=list(DATASET_IMAGE_COUNTS.keys()),
        help="Dataset name",
    )
    data_group.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset (overrides default paths)",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=16,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Training batch size",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Base learning rate",
    )
    train_group.add_argument(
        "--lr_ratio",
        type=int,
        default=10,
        help="Learning rate multiplier for hyper network layers",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for optimizer",
    )

    # Patch arguments
    patch_group = parser.add_argument_group("Patches")
    patch_group.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Size of image patches",
    )
    patch_group.add_argument(
        "--train_patch_num",
        type=int,
        default=25,
        help="Number of patches per training image",
    )
    patch_group.add_argument(
        "--test_patch_num",
        type=int,
        default=25,
        help="Number of patches per testing image",
    )

    # Experiment arguments
    exp_group = parser.add_argument_group("Experiment")
    exp_group.add_argument(
        "--train_test_num",
        type=int,
        default=10,
        help="Number of train-test rounds",
    )
    exp_group.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )
    exp_group.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
        help="Output directory for experiments",
    )
    exp_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    exp_group.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID",
    )

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args


if __name__ == "__main__":
    config = parse_args()
    main(config)
