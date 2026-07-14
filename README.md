# ProTeCt: Prototype-driven Temporal Consistency for Few-Shot Video Semantic Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Official PyTorch implementation of "ProTeCt: Prototype-driven Temporal Consistency for Few-Shot Video Semantic Segmentation" (ESWA 2025).**

> 📄 **Paper:** [Expert Systems with Applications](https://www.sciencedirect.com/science/article/abs/pii/S0957417426024401)

**ProTeCt** is a one-shot video semantic segmentation method designed for event-camera (DSEC) and frame-camera (DDD17 / Cityscapes) sequences. Given only the **first frame and its ground-truth annotation**, the model propagates segmentation masks through the entire video sequence without any additional human labeling.

The approach combines three key innovations:
- **Coordinate-aware Prototype Interaction (C-PIM):** Encodes spatial location into class prototypes for position-sensitive matching.
- **Temporal Reliability Gating (TRG):** Learns to suppress unreliable temporal priors caused by motion blur, occlusion, or previous-frame errors.
- **Prior-Guided ASPP (PG-ASPP):** Fuses multi-scale context from ASPP with prototype similarity maps and gated temporal priors via spatial attention.

---

## Method Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Support      │───▶│  C-PIM       │───▶│  Memory Bank │
│  (1st frame   │    │  Prototype   │    │  (inference) │
│   + GT mask)  │    │  Extraction  │    │              │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐
│  Query        │───▶│  ResNet      │───▶│  PG-ASPP +   │───▶ Predicted
│  (current     │    │  Encoder     │    │  TRG Fusion  │     Mask
│   frame)      │    │  (5-level)   │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Core Modules

| Module | Acronym | Description |
|--------|---------|-------------|
| **Coordinate-aware Prototype Interaction** | C-PIM | Extracts per-class prototypes from the support frame via masked average pooling with XY coordinate encoding. Matches prototypes against query features through cosine similarity. |
| **Temporal Reliability Gating** | TRG | Learns a gating map that assesses the reliability of the previous-frame prediction in the context of current-frame features, suppressing erroneous priors in blurry or occluded regions. |
| **Prior-Guided ASPP** | PG-ASPP | Multi-scale atrous spatial pyramid pooling augmented with a spatial attention gate driven by prototype similarity and gated temporal priors. |
| **Memory Bank** | — | Maintains a history of high-confidence prototypes (max 3 entries) during inference, updated every 5 frames, to combat the distribution shift as the sequence progresses. |

---

## Project Structure

```text
├── src/protect_segmentation/    # Core library
│   ├── __init__.py              # Package exports
│   ├── config.py                # Unified configuration dataclass
│   ├── model.py                 # Full model: C-PIM, TRG, PG-ASPP, Decoder
│   ├── dataset.py               # Sequence dataset with sliding-window sampling
│   ├── inference.py             # Memory Bank inference engine
│   ├── metrics.py               # IoU / mIoU computation
│   └── visualization.py         # Mask overlay visualization
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── infer_sequence.py        # Full-sequence inference with metrics
│   ├── verify_model.py          # Model integrity check (synthetic data)
│   └── preprocess_dsec.py       # DSEC RGB-Event alignment pipeline
├── tools/
│   └── benchmark_complexity.py  # FLOPs / Params / Latency benchmark
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Package metadata (pip install -e .)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.1 (CUDA 11.8+ recommended for GPU training)
- See `requirements.txt` for full dependency list

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/protect-segmentation.git
cd protect-segmentation

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# or
venv\Scripts\activate             # Windows

# 3. Install PyTorch (choose the version matching your CUDA)
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Optional) Install the package in development mode
pip install -e .
```

### Dependency List

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.1 | Deep learning framework |
| `torchvision` | >= 0.16 | Pre-trained ResNet backbones |
| `numpy` | >= 1.23 | Numerical computation |
| `opencv-python` | >= 4.8 | Image I/O and preprocessing |
| `albumentations` | >= 1.3 | Training data augmentation |
| `tqdm` | >= 4.66 | Progress bars |
| `PyYAML` | >= 6.0 | YAML configuration (optional) |

Optional (for benchmarking):
```bash
pip install thop    # FLOPs and parameter counting
```

---

## Quick Verification

Run the model integrity check to ensure all components are correctly connected:

```bash
python scripts/verify_model.py
```

This performs a complete forward pass with synthetic random inputs (128×192, CPU) and validates:
- Output tensor shapes match expectations
- No NaN or Inf values in outputs
- C-PIM, TRG, PG-ASPP, and Decoder sub-modules are functional
- Memory Bank inference interface works correctly

**Expected output:**
```
Device: cpu
Running forward pass...
Model verification passed!
  Input size: 128x192
  Num classes: 11
  Training output shape: (1, 11, 128, 192)
Memory Bank inference interface passed! (stride-4 output: (1, 11, 32, 48))
```

---

## Data Preparation

### Expected Directory Structure

The training and inference scripts expect data organized as follows:

```text
images_root/
└── sequence_name/           # One subdirectory per video sequence
    ├── 000000.png           # Frame images (.png / .jpg / .jpeg / .bmp)
    ├── 000001.png
    ├── 000002.png
    └── ...

masks_root/
└── sequence_name/
    ├── 000000.png           # Single-channel class-ID masks
    ├── 000001.png
    └── ...
    # Alternative subdirectory format also supported:
    # sequence_name/11classes/000000.png
```

- Frame filenames are sorted lexicographically to preserve temporal order (zero-padded frame indices recommended).
- Masks must be single-channel PNG images where each pixel value is the class ID (0 = background).
- Image and mask resolutions can be arbitrary; they will be resized to the model's input size (default 960×540) during loading.

### DSEC Dataset Preprocessing

The DSEC dataset contains two independent sensors with different resolutions and fields of view:
- **RGB camera:** 1440×1080, ~20 Hz
- **Event camera:** 640×480, μs temporal resolution
- Semantic labels are annotated in the event-camera coordinate frame (640×440 effective region).

To align the RGB images to the event-camera coordinate system, run the preprocessing script:

```bash
# 1. Edit paths in scripts/preprocess_dsec.py:
#    SOURCE_ROOT       — path to raw DSEC RGB images
#    SOURCE_LABEL_ROOT — path to raw DSEC labels
#    TARGET_ROOT       — output directory for processed images
#    TARGET_LABEL_ROOT — output directory for processed labels

# 2. Run the preprocessing pipeline
python scripts/preprocess_dsec.py
```

**Preprocessing pipeline (applied to each RGB image):**

1. **Perspective Warp (WARP_INVERSE_MAP):** Projects the 1440×1080 RGB image to the 640×480 event-camera coordinate frame using the calibrated homography matrix `H = K_rgb @ inv(K_event)`.
2. **Crop:** Trims to the top 440 rows (removes the bottom 40 rows of event-camera noise).
3. **Resize:** Scales the 640×440 result to the training resolution 960×540 (bilinear interpolation for images, nearest-neighbor for masks to preserve class IDs).

Labels are already aligned to the event-camera frame and only require resize (nearest-neighbor interpolation).

Images and labels are processed in parallel using `multiprocessing.Pool` (CPU count − 2 workers). Already-processed files are skipped to support resumption.

### Custom Datasets

For datasets other than DSEC, organize your data in the directory structure shown above. No preprocessing script is needed — the dataset loader supports arbitrary resolutions and will resize on-the-fly. Masks can be placed either directly in the sequence directory or under a `11classes/` subdirectory.

### Class Definitions

The default configuration uses 11 classes (Cityscapes-compatible subset, aligned with DSEC annotations):

| ID | Class | RGB Color |
|----|-------|-----------|
| 0 | background | (0, 0, 0) |
| 1 | building | (70, 70, 70) |
| 2 | fence | (190, 153, 153) |
| 3 | person | (0, 0, 255) |
| 4 | pole | (153, 153, 153) |
| 5 | road | (128, 64, 128) |
| 6 | sidewalk | (244, 35, 232) |
| 7 | vegetation | (107, 142, 35) |
| 8 | car | (0, 0, 142) |
| 9 | wall | (102, 102, 156) |
| 10 | traffic sign | (220, 220, 0) |

To use a different number of classes, pass `--num-classes N` to the training/inference scripts and update the `class_map` and `colors` fields in `SegmentationConfig` accordingly.

---

## Training

### Basic Usage

```bash
python scripts/train.py \
  --train-images /path/to/train_images \
  --train-masks /path/to/train_masks \
  --val-images /path/to/val_images \
  --val-masks /path/to/val_masks \
  --epochs 100 \
  --batch-size 2 \
  --lr 5e-5 \
  --output-dir checkpoints
```

### Training Strategy

- **Sample Construction:** Sliding windows of length 2 (support + query) with stride 1 over each sequence.
- **Mask Perturbation:** At training time, the previous-frame mask fed to TRG is randomly perturbed (morphological dilation/erosion, translation, random occlusion) with probability 0.6. This prevents TRG from overfitting to perfect ground-truth priors and bridges the train-test gap.
- **Loss Function:** CrossEntropyLoss with `ignore_index=255` (excludes unlabeled pixels).
- **Optimizer:** AdamW with cosine annealing from `lr` to `eta_min=1e-7`.
- **Gradient Clipping:** Max norm of 1.0.
- **Validation:** mIoU computed on support-query pairs every `--val-interval` epochs. The best model is saved as `best_model.pth`.

### Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--height` | 540 | Input image height (pixels) |
| `--width` | 960 | Input image width (pixels) |
| `--num-classes` | 11 | Number of semantic classes |
| `--backbone` | resnet50 | Encoder backbone (`resnet50` or `resnet101`) |
| `--no-pretrained` | False | Disable ImageNet pre-trained weights |
| `--epochs` | 100 | Total training epochs |
| `--batch-size` | 2 | Training batch size |
| `--lr` | 5e-5 | Initial learning rate |
| `--weight-decay` | 1e-4 | Weight decay for AdamW |
| `--num-workers` | 4 | DataLoader worker processes |
| `--val-interval` | 2 | Validate every N epochs |
| `--resume` | None | Resume from a checkpoint file |

### Resuming Training

```bash
python scripts/train.py \
  --train-images /path/to/train_images \
  --train-masks /path/to/train_masks \
  --resume checkpoints/last_model.pth \
  --epochs 150
```

### Output Files

Training produces two checkpoint files in the output directory:
- `last_model.pth` — most recent checkpoint (always saved)
- `best_model.pth` — checkpoint with the highest validation mIoU

Each checkpoint contains: model weights, optimizer state, scheduler state, epoch number, best mIoU, and configuration.

---

## Inference

### Basic Usage

```bash
python scripts/infer_sequence.py \
  --images /path/to/test_images \
  --masks /path/to/test_masks \
  --checkpoint checkpoints/best_model.pth \
  --output-dir outputs
```

### Inference Pipeline

1. **Model Loading:** Load the trained checkpoint and initialize `MemoryBankEngine`.
2. **Memory Bank Initialization:** Extract foreground-class prototypes from the first frame using its ground-truth mask. This entry is permanently retained.
3. **Per-Frame Prediction:**
   - Match current-frame features against all prototypes in the memory bank.
   - Apply horizontal-flip test-time augmentation at the feature level (mean of original and flipped similarity maps).
   - Gate the previous-frame prediction through TRG.
   - Fuse all signals through PG-ASPP and decode to logits.
   - Softmax → argmax → predicted mask.
4. **Memory Bank Update:** Every `--update-interval` frames (default 5), compute the mean per-pixel confidence from the softmax distribution. If it exceeds `--confidence-threshold` (default 0.92), extract new prototypes from the current prediction and append them to the memory bank.
5. **Capacity Control:** The memory bank holds at most `--max-memory-items` entries (default 3: 1 fixed support + 2 dynamic). When full, the oldest dynamic entry is evicted.

### Memory Bank Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--update-interval` | 5 | Frames between memory bank update checks |
| `--confidence-threshold` | 0.92 | Minimum mean confidence to accept new prototypes |
| `--max-memory-items` | 3 | Maximum number of prototype sets in the bank |
| `--no-feature-flip` | False | Disable horizontal-flip TTA |

### Output Structure

```text
outputs/
└── sequence_name/
    ├── pred_masks/          # Predicted masks (PNG, single-channel class IDs)
    │   ├── 000000.png       # Frame 0 = ground-truth (copied)
    │   ├── 000001.png       # Predicted
    │   └── ...
    ├── vis/                 # Visualization overlays (original + semi-transparent mask)
    │   ├── 000000.png
    │   └── ...
    └── metrics.csv          # Per-frame mIoU + per-class IoU, with global average row
```

---

## Evaluation Metrics

The `src/protect_segmentation/metrics.py` module provides standard semantic segmentation metrics:

- **Intersection over Union (IoU):** Computed per class. Pixels with `ignore_index` (default 255) are excluded from statistics.
- **Mean IoU (mIoU):** Average of per-class IoU over all classes present in the dataset. Classes that never appear (union = 0) are excluded from the mean.

The inference script automatically computes these metrics frame-by-frame and outputs a summary CSV. Metrics can also be accumulated across multiple frames before computing the global mIoU, as demonstrated in `scripts/infer_sequence.py`.

---

## Computational Complexity

Run the benchmark tool to measure parameters, FLOPs, and latency per component:

```bash
# Requires thop: pip install thop
python tools/benchmark_complexity.py
```

The tool isolates each module (Baseline encoder-decoder, C-PIM, TRG, Memory Bank update) and reports incremental costs at 540×960 input resolution, enabling targeted optimization.

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{protect2025,
  title   = {ProTeCt: Prototype-driven Temporal Consistency for Few-Shot Video Semantic Segmentation},
  author  = {},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {Accepted},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0957417426024401}
}
```

---

## License

This project is released under the [MIT License](./LICENSE).

**This repository does NOT contain:**
- The DSEC, DDD17, or Cityscapes datasets
- Pre-trained model weights
- Paper figures or experimental intermediate results

---

## Contact

For questions about the code or paper, please open a GitHub issue or contact the authors.
