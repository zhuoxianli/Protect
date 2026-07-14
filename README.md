# ProTeCt: Geometry-aware reference-guided video semantic segmentation for event camera APS frames

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Official PyTorch implementation of **"ProTeCt: Geometry-aware reference-guided video semantic segmentation for event camera APS frames"** (ESWA 2026).

> 📄 **Paper:** [Expert Systems with Applications](https://www.sciencedirect.com/science/article/abs/pii/S0957417426024401)

ProTeCt performs one-shot video semantic segmentation on event-camera (DSEC) and frame-camera sequences. Given only the first frame and its ground-truth mask, it segments the entire video via prototype matching with temporal consistency.

**Core modules:** C-PIM (coordinate-aware prototype matching), TRG (temporal reliability gating), PG-ASPP (prior-guided multi-scale fusion), and a confidence-gated memory bank.

---

## Installation

```bash
git clone https://github.com/zhuoxianli/Protect.git
cd Protect

# Install PyTorch (match your CUDA version)
pip install torch torchvision    # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 12

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python >= 3.9, PyTorch >= 2.1. See `requirements.txt` for full list.

Verify setup:

```bash
python scripts/verify_model.py
```

---

## Data Preparation

### Directory Structure

```
images_root/                     masks_root/
└── seq_A/                       └── seq_A/
    ├── 000000.png                   ├── 000000.png   # or seq_A/11classes/000000.png
    ├── 000001.png                   └── ...
    └── ...
```

Masks are single-channel PNGs with pixel values = class IDs (0 = background). Frames are sorted lexicographically. Arbitrary resolutions supported — resized to 960×540 on load.

### DSEC Preprocessing

DSEC has misaligned RGB (1440×1080) and event-camera (640×480) sensors. To align them:

```bash
# Edit paths at the top of scripts/preprocess_dsec.py, then:
python scripts/preprocess_dsec.py
```

Pipeline: **Perspective warp** (RGB→event coordinates via homography) → **Crop** to 440 rows → **Resize** to 960×540. Labels only need resize (nearest-neighbor).

### Default Classes (11-class, Cityscapes-compatible)

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|----|---|---|---|---|---|---|---|---|---|---|----|
| Class | background | building | fence | person | pole | road | sidewalk | vegetation | car | wall | traffic sign |

Override with `--num-classes N` and update `class_map`/`colors` in `SegmentationConfig`.

---

## Training

```bash
python scripts/train.py \
  --train-images /path/to/train_images \
  --train-masks /path/to/train_masks \
  --val-images /path/to/val_images \
  --val-masks /path/to/val_masks \
  --epochs 100 --batch-size 2 --lr 5e-5 \
  --output-dir checkpoints
```

- Sliding windows of 2 frames (support + query) per sequence.
- Mask perturbation (dilation/erosion/occlusion, p=0.6) prevents TRG from overfitting to perfect priors.
- AdamW + cosine annealing, CrossEntropyLoss (ignore_index=255).
- Saves `best_model.pth` and `last_model.pth`.

Resume: `--resume checkpoints/last_model.pth --epochs 150`

| Key argument | Default | Description |
|-------------|---------|-------------|
| `--height / --width` | 540 / 960 | Input resolution |
| `--backbone` | resnet50 | resnet50 or resnet101 |
| `--val-interval` | 2 | Validate every N epochs |

---

## Inference

```bash
python scripts/infer_sequence.py \
  --images /path/to/test_images \
  --masks /path/to/test_masks \
  --checkpoint checkpoints/best_model.pth \
  --output-dir outputs
```

The `MemoryBankEngine` initializes prototypes from the first-frame GT. For each subsequent frame, it matches against stored prototypes with horizontal-flip TTA, then conditionally updates the bank (every 5 frames if mean confidence ≥ 0.92). Max 3 prototype sets retained (1 fixed support + 2 dynamic).

| Argument | Default | Description |
|----------|---------|-------------|
| `--update-interval` | 5 | Update check interval |
| `--confidence-threshold` | 0.92 | Min confidence to accept |
| `--max-memory-items` | 3 | Max prototype sets |
| `--no-feature-flip` | — | Disable flip TTA |

Outputs per sequence: `pred_masks/` (PNG), `vis/` (overlays), `metrics.csv` (per-frame + global mIoU).

---

## Benchmark

```bash
pip install thop
python tools/benchmark_complexity.py
```

Reports FLOPs, parameters, and latency per module at 540×960.

---

## Project Structure

```
src/protect_segmentation/   # Core library (config, model, dataset, inference, metrics, vis)
scripts/                    # train.py, infer_sequence.py, verify_model.py, preprocess_dsec.py
tools/                      # benchmark_complexity.py
```

---

## Citation

```bibtex
@article{protect2026,
  title   = {ProTeCt: Geometry-aware reference-guided video semantic segmentation for event camera APS frames},
  author  = {},
  journal = {Expert Systems with Applications},
  year    = {2026},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0957417426024401}
}
```

## License

MIT — see [LICENSE](./LICENSE). This repository does not include datasets or pretrained weights.
