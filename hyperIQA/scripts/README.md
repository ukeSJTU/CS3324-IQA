# Scripts Usage

## run_demo.sh

Run inference on a single image to get quality score.

**From hyperIQA directory:**
```bash
./scripts/run_demo.sh [IMAGE_PATH] [MODEL_PATH] [NUM_PATCHES]
```

**Examples:**
```bash
# Use defaults (data/D_01.jpg, pretrained/koniq_pretrained.pkl, 10 patches)
./scripts/run_demo.sh

# Custom image
./scripts/run_demo.sh path/to/image.jpg

# Custom image and model
./scripts/run_demo.sh path/to/image.jpg pretrained/my_model.pkl

# All custom parameters
./scripts/run_demo.sh path/to/image.jpg pretrained/my_model.pkl 20
```

**Output:** Quality score ranging from 0-100 (higher = better quality)

---

## train.sh

Train HyperIQA model on KonIQ dataset.

**From hyperIQA directory:**
```bash
./scripts/train.sh [LR] [BATCH_SIZE] [EPOCHS] [VAL_SPLIT] [SEED]
```

**Default values:**
- LR: 2e-5
- BATCH_SIZE: 96
- EPOCHS: 16
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
./scripts/train.sh 5e-5 128 32 0.2 123
```

**Output:**
- Checkpoints saved to `../checkpoints/lr{lr}_bs{bs}_ep{ep}_val{val}_seed{seed}/`
- See `TRAINING.md` for detailed documentation

**Checkpoint folder contains:**
- `best_model.pkl` - Best model weights
- `metrics.json` - Training metrics for visualization
- `args.json` - All training arguments
- `train.log` - Training logs

## evaluate.sh

TODO: Evaluate model on test datasets (to be implemented)
