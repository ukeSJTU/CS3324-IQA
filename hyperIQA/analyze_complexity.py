"""
Compute model complexity analysis: FLOPS and inference time.

This script measures the computational complexity of the HyperIQA model
for reporting purposes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

import models
from folders import pil_loader

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(
    model: models.HyperNet,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> tuple[float, float]:
    """
    Measure FLOPS using thop library if available, otherwise estimate.

    Args:
        model: HyperNet model.
        input_size: Input tensor size (B, C, H, W).

    Returns:
        Tuple of (FLOPS, MACs) in billions.
    """
    try:
        from thop import profile, clever_format

        dummy_input = torch.randn(input_size).cuda()
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2  # MACs to FLOPS
        return flops / 1e9, macs / 1e9
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        print("Skipping FLOPS calculation.")
        return 0.0, 0.0


def measure_inference_time(
    model: models.HyperNet,
    image_path: str,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict[str, float]:
    """
    Measure inference time statistics.

    Args:
        model: HyperNet model.
        image_path: Path to test image.
        num_runs: Number of inference runs for timing.
        warmup_runs: Number of warmup runs.

    Returns:
        Dictionary with timing statistics.
    """
    # Prepare input
    transform = T.Compose([
        T.Resize((512, 384)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img = pil_loader(image_path)
    img_tensor = transform(img).unsqueeze(0).cuda()

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            params = model(img_tensor)
            model_target = models.TargetNet(params).cuda()
            _ = model_target(params["target_in_vec"])

    # Synchronize before timing
    torch.cuda.synchronize()

    # Measure time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()

            params = model(img_tensor)
            model_target = models.TargetNet(params).cuda()
            pred = model_target(params["target_in_vec"])

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "fps": float(1000 / np.mean(times)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HyperIQA model complexity."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./data/D_01.jpg",
        help="Path to test image for timing",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model (optional)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of runs for timing",
    )

    args = parser.parse_args()

    print("="*60)
    print("HyperIQA Model Complexity Analysis")
    print("="*60)

    # Initialize model
    model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()

    if args.model_path:
        checkpoint = torch.load(args.model_path)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from: {args.model_path}")

    model.eval()

    # Count parameters
    num_params = count_parameters(model)
    print(f"\n1. Model Parameters")
    print(f"   Total trainable parameters: {num_params:,}")
    print(f"   Parameters (M): {num_params / 1e6:.2f}M")

    # Measure FLOPS
    print(f"\n2. Computational Complexity")
    flops, macs = measure_flops(model)
    if flops > 0:
        print(f"   FLOPs: {flops:.2f}G")
        print(f"   MACs: {macs:.2f}G")
    else:
        print("   FLOPS calculation skipped (install thop)")

    # Measure inference time
    if Path(args.image).exists():
        print(f"\n3. Inference Time (input: {args.image})")
        timing = measure_inference_time(
            model, args.image, num_runs=args.num_runs
        )
        print(f"   Mean: {timing['mean_ms']:.2f} ms")
        print(f"   Std: {timing['std_ms']:.2f} ms")
        print(f"   Min: {timing['min_ms']:.2f} ms")
        print(f"   Max: {timing['max_ms']:.2f} ms")
        print(f"   Median: {timing['median_ms']:.2f} ms")
        print(f"   Throughput: {timing['fps']:.1f} FPS")
    else:
        print(f"\n3. Inference Time")
        print(f"   Skipped: Image not found at {args.image}")

    # GPU memory
    print(f"\n4. GPU Memory")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            params = model(dummy_input)
            model_target = models.TargetNet(params).cuda()
            _ = model_target(params["target_in_vec"])

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   Peak GPU memory: {peak_memory:.2f} GB")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
