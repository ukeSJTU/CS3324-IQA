# HyperIQA Visualization Guide

## Overview

The `visualize.py` script creates publication-quality figures from training and evaluation results using matplotlib. It supports four visualization modes for different analysis needs.

## Visualization Modes

### 1. Training Curves (`--mode training`)
Plot training progress for a single experiment:
- Training loss curve
- Training SRCC curve
- Validation SRCC curve (with best value marked)
- Validation PLCC curve (with best value marked)

**Output:** `training_curves.png` (2x2 subplot)

### 2. Evaluation Results (`--mode evaluation`)
Visualize evaluation results for a single model:
- Scatter plots: predicted vs ground truth (one per dataset)
- Bar chart: SRCC and PLCC comparison across datasets

**Output:** `evaluation_scatter.png`, `evaluation_metrics_bar.png`

### 3. Compare Training (`--mode compare-training`)
Compare training curves across multiple experiments:
- Overlay training loss curves
- Overlay training SRCC curves
- Overlay validation SRCC curves (with best values)
- Overlay validation PLCC curves (with best values)

**Output:** `compare_training.png`

### 4. Compare Evaluation (`--mode compare-eval`)
Compare evaluation metrics from multiple models:
- Side-by-side SRCC comparison across datasets
- Side-by-side PLCC comparison across datasets

**Output:** `compare_evaluation.png`

## Usage

### Basic Arguments

```bash
python visualize.py \
    --mode MODE \           # training, evaluation, compare-training, compare-eval
    --input FILE(S) \       # Input JSON file(s)
    --output DIR            # Output directory for figures
```

### Optional Arguments

```bash
--format FORMAT     # Output format: png, pdf, svg (default: png)
--dpi DPI           # DPI for raster formats (default: 300)
--labels LABELS     # Custom labels for comparison plots
```

## Examples

### 1. Plot Training Curves

```bash
# From single experiment
python visualize.py \
    --mode training \
    --input ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/metrics.json \
    --output ../figures/exp1/
```

**Generated figure:**
- `training_curves.png` - 4-panel plot showing loss and metrics over epochs

### 2. Plot Evaluation Results

```bash
# From evaluation results
python visualize.py \
    --mode evaluation \
    --input ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/eval_results.json \
    --output ../figures/exp1/
```

**Generated figures:**
- `evaluation_scatter.png` - Scatter plots for each dataset
- `evaluation_metrics_bar.png` - Bar chart comparing metrics

### 3. Compare Multiple Training Runs

```bash
# Compare 3 experiments with different learning rates
python visualize.py \
    --mode compare-training \
    --input \
        ../checkpoints/lr1e-5_bs96_ep16_val0.1_seed42/metrics.json \
        ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/metrics.json \
        ../checkpoints/lr5e-5_bs96_ep16_val0.1_seed42/metrics.json \
    --labels "LR=1e-5" "LR=2e-5" "LR=5e-5" \
    --output ../figures/lr_comparison/
```

**Generated figure:**
- `compare_training.png` - Overlay comparison of all experiments

### 4. Compare Multiple Model Evaluations

```bash
# Compare 2 models across test datasets
python visualize.py \
    --mode compare-eval \
    --input \
        ../checkpoints/model1/eval_results.json \
        ../checkpoints/model2/eval_results.json \
    --labels "HyperIQA-baseline" "HyperIQA-improved" \
    --output ../figures/model_comparison/
```

**Generated figure:**
- `compare_evaluation.png` - Side-by-side metric comparison

### 5. High-Quality Figures for Publication

```bash
# Generate PDF figures at high DPI
python visualize.py \
    --mode training \
    --input metrics.json \
    --output ../figures/paper/ \
    --format pdf \
    --dpi 600
```

## Output Formats

### PNG (default)
- Good for: Reports, presentations, quick viewing
- File size: Moderate
- Quality: High at 300 DPI

### PDF
- Good for: LaTeX papers, vector graphics
- File size: Small
- Quality: Scalable (vector)

### SVG
- Good for: Web, further editing in Illustrator/Inkscape
- File size: Small
- Quality: Scalable (vector)

## Figure Specifications

### Training Curves (`training`)
- **Size:** 12" x 10"
- **Layout:** 2x2 grid
- **Panels:**
  - Top-left: Training Loss (blue line, circle markers)
  - Top-right: Training SRCC (green line, square markers)
  - Bottom-left: Validation SRCC (red line, triangle markers, best marked)
  - Bottom-right: Validation PLCC (magenta line, diamond markers, best marked)

### Evaluation Scatter (`evaluation`)
- **Size:** 12" x 6n" (n = number of datasets / 2)
- **Layout:** n x 2 grid
- **Per panel:**
  - Scatter plot: predicted vs ground truth
  - Red dashed line: perfect prediction
  - Text box: SRCC, PLCC, sample count
  - Grid and legend

### Evaluation Bar Chart (`evaluation`)
- **Size:** 10" x 6"
- **Bars:** SRCC (steel blue), PLCC (coral)
- **Labels:** Value labels on top of bars
- **Grid:** Horizontal grid lines

### Training Comparison (`compare-training`)
- **Size:** 14" x 10"
- **Layout:** 2x2 grid
- **Colors:** Automatic color cycling (up to 10 experiments)
- **Legend:** Shows all experiment labels + best values for validation

### Evaluation Comparison (`compare-eval`)
- **Size:** 14" x 6"
- **Layout:** 1x2 (SRCC | PLCC)
- **Bars:** Grouped bars for each model
- **Colors:** Automatic color cycling

## Typical Workflow

### For Single Experiment

```bash
# Step 1: Train model
./scripts/train.sh

# Step 2: Evaluate model
./scripts/evaluate.sh ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/

# Step 3: Visualize training
python visualize.py \
    --mode training \
    --input ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/metrics.json \
    --output ../figures/exp1/

# Step 4: Visualize evaluation
python visualize.py \
    --mode evaluation \
    --input ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/eval_results.json \
    --output ../figures/exp1/
```

### For Hyperparameter Comparison

```bash
# Train multiple models with different hyperparameters
./scripts/train.sh 1e-5 96 16 0.1 42
./scripts/train.sh 2e-5 96 16 0.1 42
./scripts/train.sh 5e-5 96 16 0.1 42

# Evaluate all models
for dir in ../checkpoints/lr*_bs96_ep16_val0.1_seed42/; do
    ./scripts/evaluate.sh "$dir"
done

# Compare training curves
python visualize.py \
    --mode compare-training \
    --input \
        ../checkpoints/lr1e-5_bs96_ep16_val0.1_seed42/metrics.json \
        ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/metrics.json \
        ../checkpoints/lr5e-5_bs96_ep16_val0.1_seed42/metrics.json \
    --labels "LR=1e-5" "LR=2e-5" "LR=5e-5" \
    --output ../figures/lr_study/

# Compare evaluation results
python visualize.py \
    --mode compare-eval \
    --input \
        ../checkpoints/lr1e-5_bs96_ep16_val0.1_seed42/eval_results.json \
        ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/eval_results.json \
        ../checkpoints/lr5e-5_bs96_ep16_val0.1_seed42/eval_results.json \
    --labels "LR=1e-5" "LR=2e-5" "LR=5e-5" \
    --output ../figures/lr_study/
```

## Tips for Paper Figures

1. **Use high DPI:** `--dpi 600` for publication-quality raster images

2. **Use vector formats:** `--format pdf` or `--format svg` for LaTeX papers

3. **Meaningful labels:** Always provide `--labels` for comparison plots

4. **Consistent naming:** Keep experiment labels short and descriptive

5. **Separate output dirs:** Organize figures by experiment type
   ```
   figures/
   ├── baseline/
   ├── lr_study/
   ├── batch_size_study/
   └── final_model/
   ```

6. **Post-processing:** Generated figures can be further refined in vector editors if needed

## Troubleshooting

**Issue: "File not found"**
```
Solution: Check that metrics.json or eval_results.json exists in the checkpoint folder
```

**Issue: "Number of labels must match number of inputs"**
```
Solution: Provide exactly as many --labels as --input files for comparison modes
```

**Issue: Figures look blurry**
```
Solution: Increase --dpi (try 600) or use vector format (--format pdf)
```

**Issue: Fonts too small**
```
Solution: Font sizes are optimized for default figure sizes. If resizing externally,
regenerate with adjusted DPI instead of scaling.
```

**Issue: Comparison plot too crowded**
```
Solution: Compare fewer experiments at once (max 5-6 recommended for readability)
```
