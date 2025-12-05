"""
Compute model complexity metrics.

Usage:
    python compute_complexity.py --checkpoint ./checkpoints/best_checkpoint.pth
"""

import argparse
import json
import time

import models
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """Calculate FLOPs using fvcore."""
    try:        
        input_tensor = torch.randn(input_size).cuda()
        model.eval()
        
        with torch.no_grad():
            # Calculate FLOPs for HyperNet
            flops_counter = FlopCountAnalysis(model, input_tensor)
            hypernet_flops = flops_counter.total()
            
            # Calculate FLOPs for TargetNet
            paras = model(input_tensor)
            model_target = models.TargetNet(paras).cuda()
            target_flops_counter = FlopCountAnalysis(model_target, paras['target_in_vec'])
            target_flops = target_flops_counter.total()
            
            total_flops = hypernet_flops + target_flops
            
            return total_flops
    except Exception as e:
        print(f"Error calculating FLOPs: {str(e)}")
        return None


def measure_inference_time(model, input_tensor, num_runs=100, warmup_runs=10):
    """Measure average inference time."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            paras = model(input_tensor)
            model_target = models.TargetNet(paras).cuda()
            _ = model_target(paras['target_in_vec'])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            paras = model(input_tensor)
            model_target = models.TargetNet(paras).cuda()
            _ = model_target(paras['target_in_vec'])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
    
    return np.mean(times), np.std(times)





def main():
    parser = argparse.ArgumentParser(description='Compute HyperIQA model complexity metrics')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--patch_size', type=int, default=224,
                       help='Patch size (default: 224)')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of runs for inference time measurement (default: 100)')
    parser.add_argument('--output', type=str, default='complexity_metrics.json',
                       help='Output JSON file (default: complexity_metrics.json)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count parameters
    num_params = count_parameters(model)
    model_size_mb = num_params * 4 / (1024**2)
    
    # Calculate FLOPs
    print("Calculating FLOPs...")
    flops = calculate_flops(model, input_size=(1, 3, args.patch_size, args.patch_size))
    
    # Measure inference time
    print(f"Measuring inference time ({args.num_runs} runs)...")
    input_tensor = torch.randn(1, 3, args.patch_size, args.patch_size).cuda()
    avg_time, std_time = measure_inference_time(model, input_tensor, num_runs=args.num_runs)
    
    # Compile results
    results = {
        'parameters': {
            'total': int(num_params),
            'total_M': round(num_params / 1e6, 2),
            'model_size_MB': round(model_size_mb, 2)
        },
        'flops': {
            'total': int(flops) if flops else None,
            'total_G': round(flops / 1e9, 2) if flops else None
        },
        'inference_time': {
            'per_patch_ms': round(avg_time, 2),
            'per_patch_std_ms': round(std_time, 2),
            'per_image_25_patches_ms': round(avg_time * 25, 2),
            'throughput_per_patch_fps': round(1000.0 / avg_time, 2),
            'throughput_per_image_25_patches_fps': round(1000.0 / (avg_time * 25), 2)
        },
        'config': {
            'patch_size': args.patch_size,
            'num_runs': args.num_runs
        }
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    print(f"Parameters: {results['parameters']['total_M']}M")
    if flops:
        print(f"FLOPs: {results['flops']['total_G']}G")
    print(f"Inference time: {results['inference_time']['per_patch_ms']}ms (per patch)")


if __name__ == '__main__':
    main()
