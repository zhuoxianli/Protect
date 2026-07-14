"""
Sequence Inference Script
=========================

Runs inference on full video sequences using a trained ProTeCt model with the
Memory Bank engine.

Inference Pipeline Overview
---------------------------

1. Load model weights and dataset.
2. For each sequence:
   a. Initialize the memory bank using the first-frame GT mask.
   b. Frame-by-frame prediction (using memory bank + horizontal flip TTA).
   c. Save predicted masks (PNG), visualization overlays, and CSV metrics.
3. Summarize and output the global mIoU for each sequence.

Output File Structure::

    output_dir/
    └── sequence_name/
        ├── pred_masks/      # Pure predicted mask PNG (single-channel class ID)
        │   ├── 000000.png   # Frame 0 = GT mask (copied directly)
        │   ├── 000001.png   # Prediction
        │   └── ...
        ├── vis/             # Visualization overlays (original image + semi-transparent mask)
        │   ├── 000000.png
        │   └── ...
        └── metrics.csv      # Per-frame mIoU + per-class IoU, including global average row

Usage Example::

    python scripts/infer_sequence.py \
      --images /path/to/test_images \
      --masks /path/to/test_masks \
      --checkpoint checkpoints/best_model.pth \
      --output-dir outputs \
      --update-interval 5 \
      --confidence-threshold 0.92
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protect_segmentation import FewShotSegmenter, SegmentationConfig
from protect_segmentation.dataset import DSECFewShotDataset
from protect_segmentation.inference import MemoryBankConfig, MemoryBankEngine
from protect_segmentation.metrics import intersection_and_union, mean_iou
from protect_segmentation.model import load_checkpoint
from protect_segmentation.visualization import save_overlay


# =====================================================================
# Command-Line Arguments
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequence inference using Memory Bank"
    )
    # ---- Data ----
    parser.add_argument("--images", required=True, help="Root directory for test images")
    parser.add_argument(
        "--masks", required=True,
        help="Root directory for labels (provides first-frame support mask and enables metric computation)"
    )

    # ---- Model ----
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")

    # ---- Output ----
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Output directory (masks + visualizations + CSV)"
    )

    # ---- Model configuration ----
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument(
        "--backbone", choices=["resnet50", "resnet101"], default="resnet50"
    )
    parser.add_argument("--no-pretrained", action="store_true")

    # ---- Memory Bank parameters ----
    parser.add_argument("--update-interval", type=int, default=5,
                        help="Memory bank update interval (in frames)")
    parser.add_argument("--confidence-threshold", type=float, default=0.92,
                        help="Minimum confidence for adding to memory bank")
    parser.add_argument("--max-memory-items", type=int, default=3,
                        help="Maximum number of prototype groups retained in memory bank")
    parser.add_argument("--no-feature-flip", action="store_true",
                        help="Disable horizontal flip TTA")

    return parser.parse_args()


# =====================================================================
# Helper Functions
# =====================================================================

def save_mask(mask: np.ndarray, save_path: Path) -> None:
    """Save a single-channel class mask to a PNG file.

    Args:
        mask: Predicted mask, ``[H, W]`` uint8.
        save_path: Output path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), mask.astype(np.uint8))


# =====================================================================
# Main Function
# =====================================================================

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build configuration and model ----
    config = SegmentationConfig(
        input_height=args.height,
        input_width=args.width,
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    )

    model = FewShotSegmenter(config).to(config.device)
    load_checkpoint(model, args.checkpoint, config.device)
    model.eval()

    # ---- Build dataset ----
    dataset = DSECFewShotDataset(
        image_root=args.images,
        mask_root=args.masks,
        sequence_length=100,   # A large value; is_training=False returns the full sequence
        is_training=False,
        config=config,
    )
    # batch_size=1 because different sequences have different lengths and cannot be stacked
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ---- Initialize inference engine ----
    engine = MemoryBankEngine(
        model,
        MemoryBankConfig(
            update_interval=args.update_interval,
            max_items=args.max_memory_items,
            confidence_threshold=args.confidence_threshold,
            use_feature_flip=not args.no_feature_flip,
        ),
    )

    # ==================================================================
    # Per-Sequence Inference
    # ==================================================================
    for sequence_index, (images, masks, paths) in enumerate(loader):
        # batch_size=1, remove batch dimension -> [T, C, H, W]
        images = images.squeeze(0)
        masks = masks.squeeze(0)

        # Flatten path list (DataLoader may return nested structures)
        frame_paths = list(paths)
        frame_paths = [
            p[0] if isinstance(p, (tuple, list)) else p for p in frame_paths
        ]

        # ---- Determine sequence name and output paths ----
        sequence_name = Path(frame_paths[0]).parent.name
        sequence_dir = output_dir / sequence_name
        pred_dir = sequence_dir / "pred_masks"
        vis_dir = sequence_dir / "vis"
        csv_path = sequence_dir / "metrics.csv"
        sequence_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing sequence {sequence_index + 1}/{len(dataset)}: {sequence_name}")

        # ---- Initialize memory bank ----
        engine.reset()
        support_image = images[0:1].to(config.device)
        support_mask = masks[0:1].to(config.device)
        with torch.no_grad():
            support_features = model.extract_features(support_image)
        engine.initialize(support_features, support_mask)

        # ---- First frame: use GT mask directly ----
        previous_mask = support_mask.float()
        total_intersection = np.zeros(config.num_classes, dtype=np.float64)
        total_union = np.zeros(config.num_classes, dtype=np.float64)

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            # CSV header
            writer.writerow(
                ["frame", "mIoU"] + [f"IoU_{i}" for i in range(config.num_classes)]
            )

            # Write first frame (GT, IoU=1.0)
            first_mask = support_mask.squeeze().cpu().numpy().astype(np.uint8)
            save_mask(first_mask, pred_dir / Path(frame_paths[0]).name)
            save_overlay(
                frame_paths[0], first_mask,
                vis_dir / Path(frame_paths[0]).name, config.colors,
            )
            writer.writerow(
                [Path(frame_paths[0]).name, "1.0000"] + ["1.0000"] * config.num_classes
            )

            # ---- Per-frame prediction ----
            for frame_index in tqdm(
                range(1, images.shape[0]), leave=False, desc=sequence_name
            ):
                query_image = images[frame_index : frame_index + 1].to(config.device)

                # Inference
                pred_mask = engine.predict(
                    query_image, previous_mask, frame_index
                )
                previous_mask = pred_mask.float()

                # Save results
                pred_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
                frame_name = Path(frame_paths[frame_index]).name
                save_mask(pred_np, pred_dir / frame_name)
                save_overlay(
                    frame_paths[frame_index], pred_np,
                    vis_dir / frame_name, config.colors,
                )

                # Compute metrics
                target_np = masks[frame_index].squeeze().numpy().astype(np.uint8)
                inter, union = intersection_and_union(
                    pred_np, target_np, config.num_classes
                )
                total_intersection += inter
                total_union += union

                # Write per-frame metrics
                frame_miou, frame_ious = mean_iou(inter, union)
                iou_values = [
                    f"{x:.4f}" if not np.isnan(x) else "N/A" for x in frame_ious
                ]
                writer.writerow([frame_name, f"{frame_miou:.4f}"] + iou_values)

            # ---- Sequence summary ----
            final_miou, final_ious = mean_iou(total_intersection, total_union)
            iou_values = [
                f"{x:.4f}" if not np.isnan(x) else "N/A" for x in final_ious
            ]
            writer.writerow(["Global_Mean", f"{final_miou:.4f}"] + iou_values)
            print(f"{sequence_name} global mIoU: {final_miou:.4f}")

    print(f"\nAll sequences processed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
