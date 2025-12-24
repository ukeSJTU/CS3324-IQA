"""
Visualization script for MS-HyperIQA training and evaluation results.
Supports plotting training curves, evaluation scatter plots, and comparisons.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MS-HyperIQA training and evaluation results')

    parser.add_argument('--mode', type=str, required=True,
                        choices=['training', 'evaluation', 'compare-training', 'compare-eval'],
                        help='Visualization mode')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='Input file(s): metrics.json for training, eval_results.json for evaluation')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for saving figures')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output figure format (default: png)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure DPI for raster formats (default: 300)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for comparison plots (must match number of inputs)')

    return parser.parse_args()


def load_training_metrics(metrics_path):
    """Load training metrics from metrics.json"""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data['epochs']


def load_evaluation_results(eval_path):
    """Load evaluation results from eval_results.json"""
    with open(eval_path, 'r') as f:
        return json.load(f)


def plot_training_curves(metrics_path, output_dir, fmt='png', dpi=300):
    """Plot training curves: loss, train SRCC, val SRCC, val PLCC"""
    metrics = load_training_metrics(metrics_path)

    epochs = [m['epoch'] for m in metrics]
    train_loss = [m['train_loss'] for m in metrics]
    train_srcc = [m['train_srcc'] for m in metrics]
    val_srcc = [m['val_srcc'] for m in metrics]
    val_plcc = [m['val_plcc'] for m in metrics]

    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss (L1)', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training SRCC
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_srcc, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training SRCC', fontsize=12)
    ax2.set_title('Training SRCC Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Validation SRCC
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, val_srcc, 'r-', linewidth=2, marker='^', markersize=4, label='Val SRCC')
    ax3.axhline(y=max(val_srcc), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(val_srcc):.4f}')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation SRCC', fontsize=12)
    ax3.set_title('Validation SRCC Curve', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1])

    # Plot 4: Validation PLCC
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, val_plcc, 'm-', linewidth=2, marker='d', markersize=4, label='Val PLCC')
    ax4.axhline(y=max(val_plcc), color='m', linestyle='--', alpha=0.5, label=f'Best: {max(val_plcc):.4f}')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation PLCC', fontsize=12)
    ax4.set_title('Validation PLCC Curve', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 1])

    # Save figure
    output_path = os.path.join(output_dir, f'training_curves.{fmt}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path


def plot_evaluation_scatter(eval_path, output_dir, fmt='png', dpi=300):
    """Plot scatter plots for each dataset: predicted vs ground truth"""
    results = load_evaluation_results(eval_path)
    datasets = results['datasets']

    num_datasets = len(datasets)
    cols = 2
    rows = (num_datasets + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
    if num_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (dataset_name, data) in enumerate(datasets.items()):
        ax = axes[idx]
        predictions = data['predictions']

        preds = np.array([p['predicted'] for p in predictions])
        gts = np.array([p['ground_truth'] for p in predictions])

        # Normalize predictions to ground truth scale for better visualization
        # This doesn't affect SRCC/PLCC which are scale-invariant
        gt_min, gt_max = gts.min(), gts.max()
        pred_min, pred_max = preds.min(), preds.max()
        preds_normalized = (preds - pred_min) / (pred_max - pred_min) * (gt_max - gt_min) + gt_min

        # Scatter plot
        ax.scatter(gts, preds_normalized, alpha=0.5, s=20, edgecolors='none')

        # Perfect prediction line
        min_val = min(gt_min, preds_normalized.min())
        max_val = max(gt_max, preds_normalized.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Metrics text
        metrics = data['metrics']
        text = f"SRCC: {metrics['srcc']:.4f}\nPLCC: {metrics['plcc']:.4f}\nN: {metrics['num_samples']}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Ground Truth', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{dataset_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide unused subplots
    for idx in range(num_datasets, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'evaluation_scatter.{fmt}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path


def plot_evaluation_metrics_bar(eval_path, output_dir, fmt='png', dpi=300):
    """Plot bar chart comparing SRCC and PLCC across datasets"""
    results = load_evaluation_results(eval_path)
    datasets = results['datasets']

    dataset_names = list(datasets.keys())
    srcc_values = [datasets[name]['metrics']['srcc'] for name in dataset_names]
    plcc_values = [datasets[name]['metrics']['plcc'] for name in dataset_names]

    x = np.arange(len(dataset_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, srcc_values, width, label='SRCC', color='steelblue')
    bars2 = ax.bar(x + width/2, plcc_values, width, label='PLCC', color='coral')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'evaluation_metrics_bar.{fmt}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path


def plot_compare_training(metrics_paths, labels, output_dir, fmt='png', dpi=300):
    """Compare training curves from multiple experiments"""
    if labels is None:
        labels = [f"Exp {i+1}" for i in range(len(metrics_paths))]

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_paths)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Comparison Across Experiments', fontsize=16, fontweight='bold')

    for idx, (path, label, color) in enumerate(zip(metrics_paths, labels, colors)):
        metrics = load_training_metrics(path)
        epochs = [m['epoch'] for m in metrics]
        train_loss = [m['train_loss'] for m in metrics]
        train_srcc = [m['train_srcc'] for m in metrics]
        val_srcc = [m['val_srcc'] for m in metrics]
        val_plcc = [m['val_plcc'] for m in metrics]

        # Plot training loss
        axes[0, 0].plot(epochs, train_loss, linewidth=2, marker='o', markersize=3,
                       color=color, label=label)

        # Plot training SRCC
        axes[0, 1].plot(epochs, train_srcc, linewidth=2, marker='s', markersize=3,
                       color=color, label=label)

        # Plot validation SRCC
        axes[1, 0].plot(epochs, val_srcc, linewidth=2, marker='^', markersize=3,
                       color=color, label=f"{label} (best: {max(val_srcc):.4f})")

        # Plot validation PLCC
        axes[1, 1].plot(epochs, val_plcc, linewidth=2, marker='d', markersize=3,
                       color=color, label=f"{label} (best: {max(val_plcc):.4f})")

    # Configure subplots
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Training Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Training SRCC', fontsize=12)
    axes[0, 1].set_title('Training SRCC', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Validation SRCC', fontsize=12)
    axes[1, 0].set_title('Validation SRCC', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation PLCC', fontsize=12)
    axes[1, 1].set_title('Validation PLCC', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'compare_training.{fmt}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path


def plot_compare_evaluation(eval_paths, labels, output_dir, fmt='png', dpi=300):
    """Compare evaluation metrics from multiple models"""
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(eval_paths))]

    # Collect all results
    all_results = []
    for path in eval_paths:
        results = load_evaluation_results(path)
        all_results.append(results)

    # Get all unique dataset names
    all_datasets = set()
    for results in all_results:
        all_datasets.update(results['datasets'].keys())
    dataset_names = sorted(all_datasets)

    # Prepare data for plotting
    x = np.arange(len(dataset_names))
    width = 0.8 / len(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Evaluation Comparison Across Models', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for idx, (results, label, color) in enumerate(zip(all_results, labels, colors)):
        srcc_values = []
        plcc_values = []

        for dataset_name in dataset_names:
            if dataset_name in results['datasets']:
                srcc_values.append(results['datasets'][dataset_name]['metrics']['srcc'])
                plcc_values.append(results['datasets'][dataset_name]['metrics']['plcc'])
            else:
                srcc_values.append(0)
                plcc_values.append(0)

        # SRCC comparison
        offset = (idx - len(labels)/2 + 0.5) * width
        ax1.bar(x + offset, srcc_values, width, label=label, color=color)

        # PLCC comparison
        ax2.bar(x + offset, plcc_values, width, label=label, color=color)

    # Configure SRCC plot
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('SRCC', fontsize=12)
    ax1.set_title('SRCC Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Configure PLCC plot
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('PLCC', fontsize=12)
    ax2.set_title('PLCC Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'compare_evaluation.{fmt}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"Visualization mode: {args.mode}")
    print(f"Output directory: {args.output}")
    print(f"Output format: {args.format} (DPI: {args.dpi})")
    print()

    if args.mode == 'training':
        if len(args.input) != 1:
            print("Error: 'training' mode requires exactly 1 input file (metrics.json)")
            return

        print(f"Plotting training curves from: {args.input[0]}")
        output_path = plot_training_curves(args.input[0], args.output, args.format, args.dpi)
        print(f"✓ Saved: {output_path}")

    elif args.mode == 'evaluation':
        if len(args.input) != 1:
            print("Error: 'evaluation' mode requires exactly 1 input file (eval_results.json)")
            return

        print(f"Plotting evaluation results from: {args.input[0]}")

        # Scatter plots
        output_path1 = plot_evaluation_scatter(args.input[0], args.output, args.format, args.dpi)
        print(f"✓ Saved: {output_path1}")

        # Metrics bar chart
        output_path2 = plot_evaluation_metrics_bar(args.input[0], args.output, args.format, args.dpi)
        print(f"✓ Saved: {output_path2}")

    elif args.mode == 'compare-training':
        if len(args.input) < 2:
            print("Error: 'compare-training' mode requires at least 2 input files")
            return

        if args.labels and len(args.labels) != len(args.input):
            print("Error: Number of labels must match number of input files")
            return

        print(f"Comparing training from {len(args.input)} experiments...")
        output_path = plot_compare_training(args.input, args.labels, args.output, args.format, args.dpi)
        print(f"✓ Saved: {output_path}")

    elif args.mode == 'compare-eval':
        if len(args.input) < 2:
            print("Error: 'compare-eval' mode requires at least 2 input files")
            return

        if args.labels and len(args.labels) != len(args.input):
            print("Error: Number of labels must match number of input files")
            return

        print(f"Comparing evaluation from {len(args.input)} models...")
        output_path = plot_compare_evaluation(args.input, args.labels, args.output, args.format, args.dpi)
        print(f"✓ Saved: {output_path}")

    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
