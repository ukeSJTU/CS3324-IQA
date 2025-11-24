"""
Cross-dataset testing script for HyperIQA.

Tests a trained model on different IQA datasets to evaluate generalization.
Required by the course assignment for SPAQ, KADID-10K, and AGIQA-3K.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from scipy import stats
from tqdm import tqdm

import models
from folders import pil_loader

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_koniq_test(data_path: str) -> list[tuple[str, float]]:
    """Load KonIQ-10k test split."""
    csv_path = os.path.join(data_path, "koniq10k_scores_and_distributions.csv")
    samples = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Use last 20% as test (or use official split if available)
            if i >= 8058:  # Last ~2015 images
                img_path = os.path.join(data_path, "1024x768", row["image_name"])
                mos = float(row["MOS_zscore"])
                samples.append((img_path, mos))

    return samples


def load_spaq(data_path: str) -> list[tuple[str, float]]:
    """Load SPAQ dataset."""
    # SPAQ uses Excel file for annotations
    try:
        import pandas as pd
        anno_path = os.path.join(data_path, "Annotations", "MOS and Image attribute scores.xlsx")
        df = pd.read_excel(anno_path)

        samples = []
        for _, row in df.iterrows():
            img_name = row["Image name"]
            img_path = os.path.join(data_path, "TestImage", img_name)
            if os.path.exists(img_path):
                mos = float(row["MOS"])
                samples.append((img_path, mos))

        return samples
    except ImportError:
        print("Warning: pandas not installed. Install with: pip install pandas openpyxl")
        return []


def load_kadid(data_path: str) -> list[tuple[str, float]]:
    """Load KADID-10K dataset."""
    csv_path = os.path.join(data_path, "dmos.csv")
    samples = []

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(data_path, "images", row["dist_img"])
                dmos = float(row["dmos"])
                samples.append((img_path, dmos))

    return samples


def load_agiqa(data_path: str) -> list[tuple[str, float]]:
    """Load AGIQA-3K dataset."""
    # AGIQA-3K uses CSV for annotations
    csv_path = os.path.join(data_path, "data.csv")
    samples = []

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(data_path, "images", row["name"])
                mos = float(row["mos_quality"])
                samples.append((img_path, mos))

    return samples


DATASET_LOADERS = {
    "koniq-10k": load_koniq_test,
    "spaq": load_spaq,
    "kadid-10k": load_kadid,
    "agiqa-3k": load_agiqa,
}


def create_transform(resize: tuple[int, int] = (512, 384), crop_size: int = 224) -> T.Compose:
    """Create inference transform."""
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def evaluate_dataset(
    model: models.HyperNet,
    samples: list[tuple[str, float]],
    num_patches: int = 10,
) -> dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: HyperNet model.
        samples: List of (image_path, label) tuples.
        num_patches: Number of random patches per image.

    Returns:
        Dictionary with SRCC and PLCC metrics.
    """
    model.eval()
    transform = create_transform()

    pred_scores = []
    gt_scores = []

    with torch.no_grad():
        for img_path, label in tqdm(samples, desc="Evaluating"):
            if not os.path.exists(img_path):
                continue

            # Multiple crops for robust prediction
            img = pil_loader(img_path)
            preds = []

            for _ in range(num_patches):
                # Random crop transform
                transform_with_crop = T.Compose([
                    T.Resize((512, 384)),
                    T.RandomCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
                img_tensor = transform_with_crop(img).unsqueeze(0).cuda()

                params = model(img_tensor)
                model_target = models.TargetNet(params).cuda()
                model_target.eval()
                pred = model_target(params["target_in_vec"])
                preds.append(float(pred.item()))

            pred_scores.append(np.mean(preds))
            gt_scores.append(label)

    # Compute metrics
    pred_array = np.array(pred_scores)
    gt_array = np.array(gt_scores)

    srcc, _ = stats.spearmanr(pred_array, gt_array)
    plcc, _ = stats.pearsonr(pred_array, gt_array)

    return {
        "srcc": float(srcc),
        "plcc": float(plcc),
        "num_samples": len(pred_scores),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset testing for HyperIQA."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["spaq", "kadid-10k", "agiqa-3k"],
        choices=list(DATASET_LOADERS.keys()),
        help="Datasets to test on",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/ssl/Database",
        help="Root directory containing datasets",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=10,
        help="Number of patches per image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cross_dataset_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()

    checkpoint = torch.load(args.model_path)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Model loaded from: {args.model_path}")

    # Evaluate on each dataset
    results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"Testing on {dataset_name}")
        print(f"{'='*50}")

        # Get dataset path
        dataset_paths = {
            "koniq-10k": os.path.join(args.data_root, "koniq-10k"),
            "spaq": os.path.join(args.data_root, "SPAQ"),
            "kadid-10k": os.path.join(args.data_root, "KADID-10K"),
            "agiqa-3k": os.path.join(args.data_root, "AGIQA-3K"),
        }

        data_path = dataset_paths.get(dataset_name)
        if not os.path.exists(data_path):
            print(f"Dataset not found at: {data_path}")
            continue

        # Load samples
        loader = DATASET_LOADERS[dataset_name]
        samples = loader(data_path)

        if not samples:
            print(f"No samples loaded for {dataset_name}")
            continue

        print(f"Loaded {len(samples)} samples")

        # Evaluate
        metrics = evaluate_dataset(model, samples, args.num_patches)
        results[dataset_name] = metrics

        print(f"SRCC: {metrics['srcc']:.4f}")
        print(f"PLCC: {metrics['plcc']:.4f}")

    # Print summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"{'Dataset':<15} {'SRCC':<10} {'PLCC':<10} {'Samples':<10}")
    print("-" * 45)
    for dataset_name, metrics in results.items():
        print(
            f"{dataset_name:<15} {metrics['srcc']:<10.4f} "
            f"{metrics['plcc']:<10.4f} {metrics['num_samples']:<10}"
        )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
