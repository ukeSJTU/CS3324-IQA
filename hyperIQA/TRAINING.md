# HyperIQA Training Guide

## Quick Start

```bash
# From hyperIQA directory
./scripts/train.sh
```

## Checkpoint Folder Naming

Checkpoints are automatically saved to `../checkpoints/` with a folder name containing key hyperparameters:

**Naming Format:** `lr{lr}_bs{batch_size}_ep{epochs}_val{val_split}_seed{seed}`

**Example:** `lr2e-5_bs96_ep16_val0.1_seed42`

### Included Parameters:
- `lr` - Learning rate (e.g., 2e-5)
- `bs` - Batch size (e.g., 96)
- `ep` - Number of epochs (e.g., 16)
- `val` - Validation split ratio (e.g., 0.1 = 10%)
- `seed` - Random seed (e.g., 42)

### Why These Parameters?
These are the key hyperparameters that affect model performance and need to be tracked for comparison across experiments.

## Checkpoint Folder Contents

```
checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/
├── args.json          # All training arguments
├── train.log          # Training logs
├── metrics.json       # Raw metrics for visualization
├── epoch_16.pkl       # Final epoch checkpoint
└── best_model.pkl     # Best model (highest val SRCC)
```

### File Descriptions:

**`args.json`** - Complete record of all training arguments:
```json
{
  "lr": 2e-5,
  "batch_size": 96,
  "epochs": 16,
  "val_split": 0.1,
  "seed": 42,
  ...
}
```

**`metrics.json`** - Raw training metrics per epoch for visualization:
```json
{
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 5.234,
      "train_srcc": 0.654,
      "val_srcc": 0.621,
      "val_plcc": 0.638
    },
    ...
  ]
}
```

**`train.log`** - Complete training logs with timestamps

**`best_model.pkl`** - Model weights with highest validation SRCC

**`epoch_N.pkl`** - Saved only if `--save_all_epochs` is specified

## Training Arguments

### Data Parameters
```bash
--dataset_root        # Root directory containing images (default: ../datasets/)
--train_json          # Training JSON metadata (default: ../datasets/metas/koniq_train.json)
--val_split           # Validation split ratio (default: 0.1)
```

### Training Parameters
```bash
--lr                  # Learning rate for backbone (default: 2e-5)
--lr_ratio            # LR multiplier for hypernet (default: 10)
--weight_decay        # Weight decay (default: 5e-4)
--batch_size          # Batch size (default: 96)
--epochs              # Number of epochs (default: 16)
--seed                # Random seed (default: 42)
```

### Patch Parameters
```bash
--patch_size          # Patch size (default: 224)
--train_patch_num     # Patches per train image (default: 25)
--val_patch_num       # Patches per val image (default: 25)
```

### Checkpoint Parameters
```bash
--checkpoint_dir      # Custom checkpoint path (auto-generated if not set)
--save_all_epochs     # Save every epoch (default: only save best)
```

## Usage Examples

### Basic Training (Use Defaults)
```bash
python train.py
```

### Custom Learning Rate and Batch Size
```bash
python train.py --lr 1e-5 --batch_size 64
```

### Longer Training with Different Seed
```bash
python train.py --epochs 32 --seed 123
```

### Save All Epoch Checkpoints
```bash
python train.py --save_all_epochs
```

### Multiple Experiments
```bash
# Experiment 1: Default settings
python train.py --seed 42

# Experiment 2: Higher learning rate
python train.py --lr 5e-5 --seed 42

# Experiment 3: Larger batch size
python train.py --batch_size 128 --seed 42
```

Each experiment automatically creates a separate checkpoint folder based on its hyperparameters.

## Monitoring Training

### View Logs in Real-time
```bash
tail -f ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/train.log
```

### Check Training Progress
```bash
cat ../checkpoints/lr2e-5_bs96_ep16_val0.1_seed42/metrics.json
```

## Validation Split

The training set is automatically split into train/val based on `--val_split`:
- `val_split=0.1` → 90% train, 10% val (default)
- `val_split=0.2` → 80% train, 20% val

The split is deterministic based on the `--seed` parameter for reproducibility.

## Learning Rate Schedule

The training follows the original HyperIQA learning rate schedule:
- Epochs 1-6: Initial LR
- Epochs 7-8: LR / 10 (with hypernet ratio)
- Epochs 9+: LR / 10 (hypernet ratio = 1)

## Best Model Selection

The best model is determined by **highest validation SRCC** and saved as `best_model.pkl`.
