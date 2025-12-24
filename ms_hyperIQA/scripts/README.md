# MS-HyperIQA Scripts Usage

## train.sh

Train MS-HyperIQA model on KonIQ dataset with enhanced features (multi-scale + FPN + attention + combined loss + cosine LR).

**From ms_hyperIQA directory:**
```bash
./scripts/train.sh [LR] [BATCH_SIZE] [EPOCHS] [VAL_SPLIT] [SEED]
```

**Default values:**
- LR: 2e-5
- BATCH_SIZE: 96
- EPOCHS: 30 (longer than original HyperIQA's 16)
- VAL_SPLIT: 0.1 (10% validation)
- SEED: 42

**Examples:**
```bash
# Use all defaults
./scripts/train.sh

# Custom learning rate
./scripts/train.sh 1e-5

# Custom LR and batch size
./scripts/train.sh 1e-5 64

# All custom parameters
./scripts/train.sh 5e-5 128 40 0.2 123
```

**Features enabled by default:**
- Combined loss (L1 + Rank Loss)
- Cosine annealing LR schedule
- Multi-scale feature extraction (Layers 2, 3, 4)
- Feature Pyramid Network (FPN)
- CBAM attention modules
- All epoch checkpoints saved

**Output:**
- Checkpoints saved to `../checkpoints/ms_hyper_lr{lr}_bs{bs}_ep{ep}_combined_{timestamp}/`

**Checkpoint folder contains:**
- `best_model.pkl` - Best model weights
- `metrics.json` - Training metrics for visualization
- `args.json` - All training arguments
- `train.log` - Training logs
- `epoch_X.pkl` - Per-epoch checkpoints

---

## evaluate.sh

Evaluate trained MS-HyperIQA model on test datasets.

**From ms_hyperIQA directory:**
```bash
./scripts/evaluate.sh [CHECKPOINT_FOLDER or MODEL_PATH] [DATASETS] [PATCH_NUM]
```

**Default values:**
- CHECKPOINT_FOLDER: ../checkpoints/latest/
- DATASETS: all (koniq_test, spaq_test, kadid_test, agiqa_test)
- PATCH_NUM: 25

**Examples:**
```bash
# Evaluate best model in checkpoint folder (auto-finds best_model.pkl)
./scripts/evaluate.sh ../checkpoints/ms_hyper_lr2e-05_bs96_ep30_combined_20241224_120000/

# Evaluate specific model file
./scripts/evaluate.sh ../checkpoints/ms_hyper_lr2e-05_bs96_ep30_combined_20241224_120000/best_model.pkl

# Evaluate on specific datasets only
./scripts/evaluate.sh ../checkpoints/.../best_model.pkl "koniq_test spaq_test"

# Custom patch number
./scripts/evaluate.sh ../checkpoints/.../best_model.pkl all 50
```

**Output:**
- `eval_results.json` saved in checkpoint folder
- Contains aggregate metrics (SRCC, PLCC) and per-image predictions

**eval_results.json structure:**
```json
{
  "model_type": "MS-HyperIQA",
  "eval_time": "2024-12-24 10:30:00",
  "datasets": {
    "koniq_test": {
      "metrics": {"srcc": 0.85, "plcc": 0.86, "num_samples": 2073},
      "predictions": [
        {"image": "koniq_test/xxx.jpg", "predicted": 75.3, "ground_truth": 68.7},
        ...
      ]
    }
  }
}
```

---

## run_demo.sh

Run inference on a single image to get quality score using MS-HyperIQA.

**From ms_hyperIQA directory:**
```bash
./scripts/run_demo.sh [IMAGE_PATH] [MODEL_PATH] [NUM_PATCHES]
```

**Examples:**
```bash
# Use defaults (../data/D_01.jpg, ../pretrained/ms_hyperIQA_koniq.pkl, 10 patches)
./scripts/run_demo.sh

# Custom image
./scripts/run_demo.sh path/to/image.jpg

# Custom image and model
./scripts/run_demo.sh path/to/image.jpg ../checkpoints/YOUR_FOLDER/best_model.pkl

# All custom parameters
./scripts/run_demo.sh path/to/image.jpg ../checkpoints/YOUR_FOLDER/best_model.pkl 20
```

**Output:** Quality score ranging from 0-100 (higher = better quality)

---

## Visualization

After training and evaluation, use `visualize.py` to create figures for analysis and reports.

**Quick examples:**
```bash
# Plot training curves
python visualize.py --mode training \
    --input ../checkpoints/ms_hyper_lr2e-05_bs96_ep30_combined_YYYYMMDD_HHMMSS/metrics.json \
    --output ../results/figures/

# Plot evaluation results
python visualize.py --mode evaluation \
    --input ../checkpoints/ms_hyper_lr2e-05_bs96_ep30_combined_YYYYMMDD_HHMMSS/eval_results.json \
    --output ../results/figures/

# Compare multiple experiments
python visualize.py --mode compare-training \
    --input exp1/metrics.json exp2/metrics.json exp3/metrics.json \
    --labels "Exp1" "Exp2" "Exp3" \
    --output ../results/figures/comparison/
```

---

## Differences from Original HyperIQA Scripts

| Feature | HyperIQA | MS-HyperIQA |
|---------|----------|-------------|
| Default epochs | 16 | 30 |
| Loss function | L1 only | Combined (L1 + Rank) |
| LR schedule | Step decay | Cosine annealing |
| Architecture | Single-scale | Multi-scale + FPN + Attention |
| Checkpoint naming | Simple | Includes loss type and timestamp |
| Save all epochs | Optional | Enabled by default |

---

## See Also

- `../README.md` - Complete MS-HyperIQA documentation
- `../../EXPERIMENT_GUIDE.md` - Full experiment sequence for assignment
- `../../hyperIQA/scripts/README.md` - Original HyperIQA scripts
