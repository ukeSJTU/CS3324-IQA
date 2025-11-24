"""
Demo script for HyperIQA inference.

This script demonstrates how to use a pre-trained HyperIQA model
to predict the quality score of an input image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

import models
from folders import pil_loader

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default inference settings
DEFAULT_IMAGE_PATH = "./data/D_01.jpg"
DEFAULT_MODEL_PATH = "./pretrained/koniq_pretrained.pkl"
DEFAULT_NUM_PATCHES = 10
DEFAULT_RESIZE = (512, 384)
DEFAULT_CROP_SIZE = 224

# HyperNet architecture parameters (must match training)
HYPERNET_LDA_OUT_CHANNELS = 16
HYPERNET_HYPER_IN_CHANNELS = 112
HYPERNET_TARGET_IN_SIZE = 224
HYPERNET_TARGET_FC1_SIZE = 112
HYPERNET_TARGET_FC2_SIZE = 56
HYPERNET_TARGET_FC3_SIZE = 28
HYPERNET_TARGET_FC4_SIZE = 14
HYPERNET_FEATURE_SIZE = 7


def create_inference_transform(
    resize: tuple[int, int] = DEFAULT_RESIZE,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> T.Compose:
    """
    Create image transform pipeline for inference.

    Args:
        resize: Target size for resizing (height, width).
        crop_size: Size of random crop.

    Returns:
        Composed transform pipeline.
    """
    return T.Compose([
        T.Resize(resize),
        T.RandomCrop(size=crop_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(model_path: str) -> models.HyperNet:
    """
    Load a pre-trained HyperIQA model.

    Args:
        model_path: Path to the pre-trained model weights.

    Returns:
        Loaded HyperNet model in evaluation mode.
    """
    model = models.HyperNet(
        HYPERNET_LDA_OUT_CHANNELS,
        HYPERNET_HYPER_IN_CHANNELS,
        HYPERNET_TARGET_IN_SIZE,
        HYPERNET_TARGET_FC1_SIZE,
        HYPERNET_TARGET_FC2_SIZE,
        HYPERNET_TARGET_FC3_SIZE,
        HYPERNET_TARGET_FC4_SIZE,
        HYPERNET_FEATURE_SIZE,
    ).cuda()

    model.load_state_dict(torch.load(model_path))
    model.train(False)

    return model


def predict_quality(
    model: models.HyperNet,
    image_path: str,
    num_patches: int = DEFAULT_NUM_PATCHES,
) -> float:
    """
    Predict quality score for an image.

    Uses random cropping to generate multiple patches and averages
    their quality predictions for robustness.

    Args:
        model: Pre-trained HyperNet model.
        image_path: Path to the input image.
        num_patches: Number of random patches to average.

    Returns:
        Predicted quality score (range 0-100, higher is better).
    """
    transforms = create_inference_transform()
    pred_scores: list[float] = []

    for _ in range(num_patches):
        img = pil_loader(image_path)
        img_tensor = transforms(img).cuda().unsqueeze(0)

        # Generate target network parameters
        params = model(img_tensor)

        # Build and run target network
        model_target = models.TargetNet(params).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        pred = model_target(params["target_in_vec"])
        pred_scores.append(float(pred.item()))

    return float(np.mean(pred_scores))


def main() -> None:
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Predict image quality using HyperIQA."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the pre-trained model weights",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=DEFAULT_NUM_PATCHES,
        help="Number of random patches for averaging",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    # Load model and predict
    model = load_model(args.model)
    score = predict_quality(model, args.image, args.num_patches)

    # Quality score ranges from 0-100, higher indicates better quality
    print(f"Predicted quality score: {score:.2f}")


if __name__ == "__main__":
    main()
