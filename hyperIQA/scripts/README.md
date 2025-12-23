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

TODO: Train model on dataset (to be implemented)

## evaluate.sh

TODO: Evaluate model on test datasets (to be implemented)
