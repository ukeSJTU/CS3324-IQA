# HyperIQA Evaluation Guide

## Quick Start

```bash
# From hyperIQA directory
# Evaluate on all test datasets
./scripts/evaluate.sh ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/
```

## Overview

The evaluation script tests a trained HyperIQA model on test datasets and generates detailed results including:
- Aggregate metrics (SRCC, PLCC)
- Per-image predictions for error analysis
- Results saved in JSON format for post-processing

## Test Datasets

The following test datasets are available:

| Dataset      | Purpose                          | Target Performance    |
|--------------|----------------------------------|----------------------|
| koniq_test   | Main test set (KonIQ)           | SRCC, PLCC > 0.75    |
| spaq_test    | Cross-dataset test (SPAQ)       | SRCC, PLCC > 0.70    |
| kadid_test   | Cross-dataset test (KADID-10K)  | No strict requirement |
| agiqa_test   | Cross-dataset test (AGIQA-3K)   | No strict requirement |

## Evaluation Arguments

### Model Parameters
```bash
# Option 1: Specify checkpoint folder (auto-finds best_model.pkl)
--checkpoint_folder PATH

# Option 2: Specify exact model file path
--model_path PATH
```

### Dataset Parameters
```bash
--dataset_root DIR     # Root directory containing datasets (default: ../datasets/)
--datasets NAMES       # Which datasets to evaluate (default: all)
                       # Options: koniq_test, spaq_test, kadid_test, agiqa_test, all
```

### Evaluation Parameters
```bash
--patch_size SIZE      # Patch size for evaluation (default: 224)
--patch_num NUM        # Number of patches per image (default: 25)
```

### Output Parameters
```bash
--output_file PATH     # Custom output path (default: auto-saved in checkpoint folder)
```

## Usage Examples

### Basic Evaluation

**Using checkpoint folder (recommended):**
```bash
python evaluate.py --checkpoint_folder ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/
```

**Using model file path:**
```bash
python evaluate.py --model_path ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/best_model.pkl
```

### Evaluate Specific Datasets

```bash
# Only required datasets (koniq_test + spaq_test)
python evaluate.py --checkpoint_folder ../checkpoints/.../  \
    --datasets koniq_test spaq_test

# Only main test set
python evaluate.py --checkpoint_folder ../checkpoints/.../  \
    --datasets koniq_test
```

### Custom Patch Number

```bash
# Use more patches for more stable predictions (slower)
python evaluate.py --checkpoint_folder ../checkpoints/.../  \
    --patch_num 50
```

### Custom Output Location

```bash
# Save results to specific location
python evaluate.py --checkpoint_folder ../checkpoints/.../  \
    --output_file ../results/experiment1_eval.json
```

## Output Format

### File Location

By default, results are saved as `eval_results.json` in the checkpoint folder:
```
checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/
├── best_model.pkl
├── metrics.json
├── args.json
└── eval_results.json    ← Evaluation results
```

### JSON Structure

```json
{
  "model_path": "/path/to/best_model.pkl",
  "eval_time": "2025-12-23 22:45:00",
  "patch_num": 25,
  "datasets": {
    "koniq_test": {
      "metrics": {
        "srcc": 0.8234,
        "plcc": 0.8156,
        "num_samples": 2073
      },
      "predictions": [
        {
          "image": "koniq_test/10007357496.jpg",
          "predicted": 75.34,
          "ground_truth": 68.73
        },
        {
          "image": "koniq_test/10020766793.jpg",
          "predicted": 82.15,
          "ground_truth": 81.51
        },
        ...
      ]
    },
    "spaq_test": {
      "metrics": { ... },
      "predictions": [ ... ]
    }
  }
}
```

## Understanding Results

### Aggregate Metrics

**SRCC (Spearman Rank Correlation Coefficient):**
- Measures monotonic relationship between predictions and ground truth
- Range: -1 to 1 (1 = perfect ranking)
- More robust to outliers

**PLCC (Pearson Linear Correlation Coefficient):**
- Measures linear relationship between predictions and ground truth
- Range: -1 to 1 (1 = perfect linear correlation)
- Sensitive to outliers

### Per-Image Predictions

Each prediction contains:
- `image`: Relative path to the image
- `predicted`: Model's predicted quality score (0-100, higher = better)
- `ground_truth`: Human-annotated ground truth score

Use these for:
- **Error analysis:** Find images where model fails
- **Visualization:** Create scatter plots (predicted vs ground truth)
- **Additional metrics:** Calculate MAE, RMSE, etc.
- **Outlier detection:** Identify problematic images

## Post-Processing Examples

### Calculate Additional Metrics

```python
import json
import numpy as np

# Load results
with open('eval_results.json', 'r') as f:
    results = json.load(f)

# Get predictions for a dataset
preds = [p['predicted'] for p in results['datasets']['koniq_test']['predictions']]
gts = [p['ground_truth'] for p in results['datasets']['koniq_test']['predictions']]

# Calculate MAE
mae = np.mean(np.abs(np.array(preds) - np.array(gts)))
print(f"MAE: {mae:.4f}")

# Calculate RMSE
rmse = np.sqrt(np.mean((np.array(preds) - np.array(gts))**2))
print(f"RMSE: {rmse:.4f}")
```

### Find Worst Predictions

```python
import json

with open('eval_results.json', 'r') as f:
    results = json.load(f)

predictions = results['datasets']['koniq_test']['predictions']

# Sort by prediction error
errors = [(p['image'], abs(p['predicted'] - p['ground_truth']))
          for p in predictions]
errors.sort(key=lambda x: x[1], reverse=True)

# Print top 10 worst predictions
print("Top 10 worst predictions:")
for img, error in errors[:10]:
    print(f"{img}: error = {error:.2f}")
```

### Create Scatter Plot

```python
import json
import matplotlib.pyplot as plt

with open('eval_results.json', 'r') as f:
    results = json.load(f)

predictions = results['datasets']['koniq_test']['predictions']
preds = [p['predicted'] for p in predictions]
gts = [p['ground_truth'] for p in predictions]

plt.figure(figsize=(8, 8))
plt.scatter(gts, preds, alpha=0.5)
plt.plot([0, 100], [0, 100], 'r--')  # Perfect prediction line
plt.xlabel('Ground Truth')
plt.ylabel('Predicted')
plt.title('KonIQ Test Set: Predicted vs Ground Truth')
plt.savefig('scatter_plot.png')
```

## Evaluation Best Practices

1. **Use same patch_num as training:** Keep consistent with validation settings (default: 25)

2. **Evaluate on all datasets:** Even if optional, they provide insights into generalization

3. **Save per-image predictions:** Essential for error analysis and report writing

4. **Check target performance:**
   - KonIQ test: SRCC, PLCC > 0.75
   - SPAQ test: SRCC, PLCC > 0.70

5. **Compare multiple checkpoints:** Evaluate different epochs to find best model

## Troubleshooting

**Issue: Model not found**
```
Solution: Check that best_model.pkl exists in checkpoint folder
```

**Issue: Dataset not found**
```
Solution: Verify dataset exists in ../datasets/ and has corresponding JSON in metas/
```

**Issue: CUDA out of memory**
```
Solution: Evaluation uses batch_size=1, should not have memory issues
Check GPU availability with: nvidia-smi
```

**Issue: Different results from validation**
```
Possible reasons:
1. Different patch_num (validation vs test)
2. Different random seed for patch sampling
3. Using wrong model checkpoint
```
