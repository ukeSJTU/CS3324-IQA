"""
Evaluation script for HyperIQA on test datasets.
Evaluates trained model and saves per-image predictions for analysis.
"""

import os
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from scipy import stats

import models
from dataset import IQADataset
from logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HyperIQA on test datasets')

    # Model parameters
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model_path', type=str,
                            help='Path to model checkpoint (e.g., best_model.pkl)')
    model_group.add_argument('--checkpoint_folder', type=str,
                            help='Checkpoint folder (auto-finds best_model.pkl)')

    # Dataset parameters
    parser.add_argument('--dataset_root', type=str, default='../datasets/',
                        help='Root directory containing image folders')
    parser.add_argument('--datasets', type=str, nargs='+', default=['all'],
                        help='Datasets to evaluate: koniq_test, spaq_test, kadid_test, agiqa_test, or all')

    # Evaluation parameters
    parser.add_argument('--patch_size', type=int, default=224,
                        help='Size of image patches')
    parser.add_argument('--patch_num', type=int, default=25,
                        help='Number of patches to sample per image')

    # Output parameters
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: auto-generated in checkpoint folder)')

    return parser.parse_args()


def load_model(model_path, logger):
    """Load HyperIQA model from checkpoint"""
    logger.info(f"Loading model from: {model_path}")

    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
    model_hyper.load_state_dict(torch.load(model_path, weights_only=False))
    model_hyper.eval()

    logger.success("Model loaded successfully")
    return model_hyper


def evaluate_dataset(model_hyper, dataset_name, dataset_root, patch_size, patch_num, logger):
    """
    Evaluate model on a single dataset.

    Returns:
        dict with 'metrics' (srcc, plcc, num_samples) and 'predictions' (per-image results)
    """
    logger.info(f"Evaluating on {dataset_name}...")

    # Load dataset
    json_path = os.path.join(dataset_root, 'metas', f'{dataset_name}.json')

    if not os.path.exists(json_path):
        logger.warning(f"Dataset {dataset_name} not found at {json_path}, skipping...")
        return None

    dataset = IQADataset(
        dataset_root=dataset_root,
        json_path=json_path,
        patch_size=patch_size,
        patch_num=patch_num,
        is_train=False,
        resize_size=(512, 384)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate
    pred_scores = []
    gt_scores = []
    image_paths = []

    with torch.no_grad():
        for imgs, labels in dataloader:
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

    # Get image paths from dataset
    # Note: Each image has patch_num patches, so we need to get unique image paths
    num_images = len(dataset) // patch_num
    for i in range(num_images):
        img_path, _ = dataset.samples[i * patch_num]
        # Extract relative path from dataset_root
        rel_path = os.path.relpath(img_path, dataset_root)
        image_paths.append(rel_path)

    # Average predictions over patches
    pred_scores_avg = np.mean(np.reshape(np.array(pred_scores), (-1, patch_num)), axis=1)
    gt_scores_avg = np.mean(np.reshape(np.array(gt_scores), (-1, patch_num)), axis=1)

    # Calculate metrics
    srcc, _ = stats.spearmanr(pred_scores_avg, gt_scores_avg)
    plcc, _ = stats.pearsonr(pred_scores_avg, gt_scores_avg)

    logger.info(f"{dataset_name}: SRCC={srcc:.4f}, PLCC={plcc:.4f}, Samples={len(pred_scores_avg)}")

    # Build per-image predictions
    predictions = []
    for img_path, pred, gt in zip(image_paths, pred_scores_avg, gt_scores_avg):
        predictions.append({
            "image": img_path,
            "predicted": float(pred),
            "ground_truth": float(gt)
        })

    return {
        "metrics": {
            "srcc": float(srcc),
            "plcc": float(plcc),
            "num_samples": len(pred_scores_avg)
        },
        "predictions": predictions
    }


def save_results(results, output_file, logger):
    """Save evaluation results to JSON"""
    logger.info(f"Saving results to: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.success("Results saved successfully")


def main():
    args = parse_args()

    # Setup logger (console only for evaluation)
    logger = setup_logger()

    logger.info("="*60)
    logger.info("Evaluating HyperIQA")
    logger.info("="*60)

    # Determine model path
    if args.checkpoint_folder:
        model_path = os.path.join(args.checkpoint_folder, 'best_model.pkl')
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return
    else:
        model_path = args.model_path

    # Load model
    model_hyper = load_model(model_path, logger)

    # Determine datasets to evaluate
    if 'all' in args.datasets:
        datasets = ['koniq_test', 'spaq_test', 'kadid_test', 'agiqa_test']
    else:
        datasets = args.datasets

    logger.info(f"Datasets to evaluate: {', '.join(datasets)}")
    logger.info("")

    # Evaluate on each dataset
    results = {
        "model_path": os.path.abspath(model_path),
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patch_num": args.patch_num,
        "datasets": {}
    }

    for dataset_name in datasets:
        result = evaluate_dataset(
            model_hyper,
            dataset_name,
            args.dataset_root,
            args.patch_size,
            args.patch_num,
            logger
        )

        if result is not None:
            results["datasets"][dataset_name] = result

    # Determine output file path
    if args.output_file is None:
        if args.checkpoint_folder:
            output_dir = args.checkpoint_folder
        else:
            output_dir = os.path.dirname(args.model_path)
        output_file = os.path.join(output_dir, 'eval_results.json')
    else:
        output_file = args.output_file

    # Save results
    save_results(results, output_file, logger)

    # Print summary
    logger.info("")
    logger.info("="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    logger.info(f"{'Dataset':<20}{'SRCC':<12}{'PLCC':<12}{'Samples':<10}")
    logger.info("-"*60)

    for dataset_name, data in results["datasets"].items():
        metrics = data["metrics"]
        logger.info(
            f"{dataset_name:<20}"
            f"{metrics['srcc']:<12.4f}"
            f"{metrics['plcc']:<12.4f}"
            f"{metrics['num_samples']:<10}"
        )

    logger.info("="*60)
    logger.success(f"Evaluation complete! Results saved to: {output_file}")


if __name__ == '__main__':
    main()
