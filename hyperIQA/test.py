"""
Testing script for HyperIQA model.

This script evaluates a trained HyperIQA model on multiple test datasets.
It supports KonIQ, SPAQ, KADID-10K, and AGIQA-3K datasets.

Usage:
    # Test on KonIQ test set
    python test.py --checkpoint ./checkpoints/best_checkpoint.pth \\
                   --test_json ../datasets/metas/koniq_test.json \\
                   --dataset_root ../datasets \\
                   --dataset_name koniq
    
    # Test on SPAQ dataset
    python test.py --checkpoint ./checkpoints/best_checkpoint.pth \\
                   --test_json ../datasets/metas/spaq_test.json \\
                   --dataset_root ../datasets \\
                   --dataset_name spaq
    
    # Test on all datasets
    python test.py --checkpoint ./checkpoints/best_checkpoint.pth \\
                   --dataset_root ../datasets \\
                   --test_all
"""

import argparse
import json
import os

# Import HyperIQA components
import models
import numpy as np
import torch
from json_dataset import JSONImageDataset, get_transforms
from scipy import stats


class HyperIQATester:
    """Tester class for HyperIQA model."""
    
    def __init__(self, checkpoint_path, config):
        self.config = config
        self.patch_num = config.patch_num
        
        # Initialize model
        self.model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Checkpoint loaded (trained for {checkpoint['epoch']} epochs)")
        if 'best_val_srcc' in checkpoint:
            print(f"Best validation SRCC: {checkpoint['best_val_srcc']:.4f}")
            print(f"Best validation PLCC: {checkpoint['best_val_plcc']:.4f}")
        print()
    
    def test(self, test_loader, dataset_name):
        """Test the model on a dataset."""
        print(f"Testing on {dataset_name} dataset...")
        print(f"Number of test images: {len(test_loader.dataset) // self.patch_num}")
        
        pred_scores = []
        gt_scores = []
        
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(test_loader):
                img = img.cuda()
                label = label.cuda()
                
                # Generate weights and predict
                paras = self.model(img)
                model_target = models.TargetNet(paras).cuda()
                model_target.eval()
                pred = model_target(paras['target_in_vec'])
                
                pred_scores.append(float(pred.item()))
                gt_scores.extend(label.cpu().tolist())
                
                # Print progress
                if (batch_idx + 1) % 100 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(test_loader)} patches")
        
        # Average predictions over patches
        pred_scores = np.mean(np.reshape(np.array(pred_scores), 
                                        (-1, self.patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), 
                                      (-1, self.patch_num)), axis=1)
        
        # Calculate metrics
        srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        
        print(f"\nResults on {dataset_name}:")
        print(f"  SRCC: {srcc:.4f}")
        print(f"  PLCC: {plcc:.4f}")
        print()
        
        return {
            'dataset': dataset_name,
            'srcc': float(srcc),
            'plcc': float(plcc),
            'num_images': len(pred_scores),
            'predictions': pred_scores.tolist(),
            'ground_truth': gt_scores.tolist()
        }


def test_single_dataset(tester, config, dataset_name, json_file):
    """Test on a single dataset."""
    # Create dataset
    test_transform = get_transforms(patch_size=config.patch_size, is_train=False)
    
    test_dataset = JSONImageDataset(
        json_path=json_file,
        root_dir=config.dataset_root,
        transform=test_transform,
        patch_num=config.patch_num
    )
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Test
    results = tester.test(test_loader, dataset_name)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test HyperIQA model on IQA datasets')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Dataset arguments
    parser.add_argument('--dataset_root', type=str, 
                       default='../datasets',
                       help='Root directory for dataset images')
    parser.add_argument('--test_json', type=str,
                       help='Path to test JSON file (required if not using --test_all)')
    parser.add_argument('--dataset_name', type=str,
                       help='Name of the dataset (required if not using --test_all)')
    
    # Test all datasets option
    parser.add_argument('--test_all', action='store_true',
                       help='Test on all available datasets')
    
    # Data arguments
    parser.add_argument('--patch_size', type=int, default=224,
                       help='Patch size for cropping (default: 224)')
    parser.add_argument('--patch_num', type=int, default=25,
                       help='Number of patches per test image (default: 25)')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save results JSON file (optional)')
    
    config = parser.parse_args()
    
    # Validate arguments
    if not config.test_all:
        if not config.test_json or not config.dataset_name:
            parser.error("--test_json and --dataset_name are required when not using --test_all")
    
    # Create tester
    tester = HyperIQATester(config.checkpoint, config)
    
    # Define dataset configurations
    dataset_configs = {
        'koniq': os.path.join(config.dataset_root, 'metas/koniq_test.json'),
        'spaq': os.path.join(config.dataset_root, 'metas/spaq_test.json'),
        'kadid': os.path.join(config.dataset_root, 'metas/kadid_test.json'),
        'agiqa': os.path.join(config.dataset_root, 'metas/agiqa_test.json'),
    }
    
    all_results = []
    
    if config.test_all:
        # Test on all datasets
        print("="*80)
        print("Testing on all datasets")
        print("="*80)
        print()
        
        for dataset_name, json_file in dataset_configs.items():
            if os.path.exists(json_file):
                try:
                    results = test_single_dataset(tester, config, dataset_name, json_file)
                    all_results.append(results)
                except Exception as e:
                    print(f"Error testing on {dataset_name}: {str(e)}")
                    print()
            else:
                print(f"Skipping {dataset_name}: JSON file not found at {json_file}")
                print()
        
        # Print summary
        print("="*80)
        print("Summary of Results")
        print("="*80)
        print(f"{'Dataset':<15} {'SRCC':<10} {'PLCC':<10} {'Num Images':<12}")
        print("-"*80)
        for result in all_results:
            print(f"{result['dataset']:<15} {result['srcc']:<10.4f} {result['plcc']:<10.4f} {result['num_images']:<12}")
        print("="*80)
        
    else:
        # Test on single dataset
        print("="*80)
        print(f"Testing on {config.dataset_name} dataset")
        print("="*80)
        print()
        
        results = test_single_dataset(tester, config, config.dataset_name, config.test_json)
        all_results.append(results)
    
    # Save results to file if specified
    if config.output_file:
        output_data = {
            'checkpoint': config.checkpoint,
            'patch_size': config.patch_size,
            'patch_num': config.patch_num,
            'results': all_results
        }
        
        with open(config.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {config.output_file}")


if __name__ == '__main__':
    main()
