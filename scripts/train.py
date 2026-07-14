"""
Training Entry Script
=====================

Complete training pipeline for the ProTeCt few-shot video semantic segmentation model.

Training Strategy Overview
--------------------------

1. **Sample Construction**:
   Uses the sliding-window mode of ``DSECFewShotDataset``, generating
   (support, query) pairs with sequence_length=2 and stride=1 per sequence.
   Frame 1 = support, frame 2 = query.

2. **Mask Perturbation**:
   During training, the previous-frame GT mask is randomly perturbed (morphological
   dilation/erosion, translation, random occlusion) before being fed into TRG.
   Purpose: simulate the imperfect predictions encountered during inference,
   preventing TRG from over-relying on "always-perfect GT priors". This is the
   most important gap bridger between training and inference.

3. **Loss Function**:
   CrossEntropyLoss, ignore_index=255 (excludes "don't care" regions in annotations).

4. **Optimizer & Scheduler**:
   AdamW + Cosine Annealing (CosineAnnealingLR), annealing from initial lr to
   eta_min=1e-7.

5. **Validation**:
   Every ``val_interval`` epochs, mIoU is computed on the validation set's
   (support, query) pairs. The best model weights are saved as ``best_model.pth``.

Usage Example::

    python scripts/train.py \
      --train-images /path/to/train_images \
      --train-masks /path/to/train_masks \
      --val-images /path/to/val_images \
      --val-masks /path/to/val_masks \
      --epochs 100 \
      --batch-size 2 \
      --output-dir checkpoints
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure the src/ directory is on the Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protect_segmentation import FewShotSegmenter, SegmentationConfig
from protect_segmentation.dataset import DSECFewShotDataset
from protect_segmentation.metrics import intersection_and_union, mean_iou
from protect_segmentation.model import load_checkpoint


# =====================================================================
# Mask Perturbation — Critical Data Augmentation During Training
# =====================================================================

def perturb_mask(mask: torch.Tensor, prob: float = 0.6) -> torch.Tensor:
    """Randomly perturb the previous-frame mask to simulate prediction noise at inference.

    This is the key alignment between training and inference: during training,
    the previous-frame mask seen by TRG should not be a perfect GT, but should
    contain various prediction errors just as it would at inference time.

    Perturbation types (applied cumulatively with given probabilities):

    1. **Morphological perturbation** (after entering the perturbation branch at p=0.6,
       p=0.5 dilation / p=0.5 erosion):
       Uses 3/5/7-sized pooling kernels to simulate boundary expansion/contraction
       of segmentation borders — object boundaries are the most error-prone regions
       during inference.

    2. **Translation error** (p=0.4):
       Random translation of +/-5 pixels, simulating prediction shifts caused by motion.

    3. **Random occlusion** (p=0.2):
       Randomly zeros out a rectangular region, simulating complete misses due to
       occlusion or objects leaving the field of view during inference.

    Args:
        mask: Original GT mask, shape ``[B, 1, H, W]``.
        prob: Probability of entering the perturbation branch. prob=0.6 means
              clean GT is returned 40% of the time.

    Returns:
        Perturbed mask ``[B, 1, H, W]`` float32.
    """
    # With probability (1 - prob), return clean GT directly
    if random.random() > prob:
        return mask.float()

    mask_float = mask.float()
    _, _, height, width = mask_float.shape

    # ---- 1. Morphological perturbation (dilation or erosion) ----
    kernel_size = random.choice([3, 5, 7])
    padding = kernel_size // 2
    if random.random() < 0.5:
        # Dilation: expand foreground regions (simulates over-segmentation)
        mask_float = F.max_pool2d(
            mask_float, kernel_size, stride=1, padding=padding
        )
    else:
        # Erosion: shrink foreground regions (simulates under-segmentation)
        mask_float = 1.0 - F.max_pool2d(
            1.0 - mask_float, kernel_size, stride=1, padding=padding
        )

    # ---- 2. Translation perturbation ----
    if random.random() < 0.4:
        dx = random.randint(-5, 5)   # X-direction translation
        dy = random.randint(-5, 5)   # Y-direction translation
        mask_float = torch.roll(mask_float, shifts=(dy, dx), dims=(2, 3))

    # ---- 3. Random occlusion ----
    if random.random() < 0.2:
        # Occlusion region size: 10 to 1/4 of image dimensions
        drop_h = random.randint(10, max(10, height // 4))
        drop_w = random.randint(10, max(10, width // 4))
        y = random.randint(0, max(0, height - drop_h))
        x = random.randint(0, max(0, width - drop_w))
        mask_float[:, :, y: y + drop_h, x: x + drop_w] = 0

    return mask_float


# =====================================================================
# Validation
# =====================================================================

@torch.no_grad()
def validate(
    model: FewShotSegmenter,
    loader: DataLoader,
    config: SegmentationConfig,
) -> Tuple[float, torch.Tensor]:
    """Evaluate the model on the validation set's support-query pairs.

    Note: During validation, the previous-frame mask uses clean GT (unperturbed),
    so that C-PIM's few-shot segmentation capability can be measured independently
    without being affected by TRG perturbation.

    Args:
        model: The model to evaluate.
        loader: Validation set DataLoader.
        config: Model configuration.

    Returns:
        (miou, per_class_ious) tuple.
    """
    model.eval()
    total_intersection = None
    total_union = None

    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images = images.to(config.device, non_blocking=True)
        masks = masks.to(config.device, non_blocking=True)

        # Split into support and query
        support_image, support_mask = images[:, 0], masks[:, 0]
        query_image, query_mask = images[:, 1], masks[:, 1].squeeze(1)

        # Use clean GT as previous-frame mask during validation
        previous_mask = support_mask.float()

        logits = model(support_image, support_mask, query_image, previous_mask)
        pred_mask = torch.argmax(logits, dim=1)

        # Per-sample accumulation of statistics
        pred_np = pred_mask.cpu().numpy()
        target_np = query_mask.cpu().numpy()
        for batch_index in range(pred_np.shape[0]):
            inter, union = intersection_and_union(
                pred_np[batch_index],
                target_np[batch_index],
                num_classes=config.num_classes,
                ignore_index=config.ignore_index,
            )
            total_intersection = (
                inter if total_intersection is None else total_intersection + inter
            )
            total_union = (
                union if total_union is None else total_union + union
            )

    if total_intersection is None or total_union is None:
        return 0.0, torch.zeros(config.num_classes)

    miou, ious = mean_iou(total_intersection, total_union)
    return miou, torch.from_numpy(ious)


# =====================================================================
# DataLoader Construction
# =====================================================================

def build_loader(
    image_root: str,
    mask_root: str,
    config: SegmentationConfig,
    batch_size: int,
    is_training: bool,
    use_augmentation: bool,
    num_workers: int,
) -> DataLoader:
    """Build a DataLoader for the sequence dataset.

    Args:
        image_root: Root directory for images.
        mask_root: Root directory for labels.
        config: Model configuration.
        batch_size: Batch size (shared by training and validation).
        is_training: Whether to enable sliding-window sampling.
        use_augmentation: Whether to enable data augmentation.
        num_workers: Number of DataLoader worker processes.

    Returns:
        A configured DataLoader.
    """
    dataset = DSECFewShotDataset(
        image_root=image_root,
        mask_root=mask_root,
        sequence_length=2,   # support + query
        is_training=is_training,
        config=config,
    )

    # Disable augmentation during validation (keep sliding-window but no extra transforms)
    if not use_augmentation:
        dataset.augment = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_augmentation,          # Shuffle during training
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Accelerate CPU->GPU transfer
        drop_last=use_augmentation,         # Drop incomplete last batch during training
    )


# =====================================================================
# Command-Line Arguments
# =====================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    All path arguments are required (training images and labels) to avoid
    hardcoding paths in code. Validation paths are optional — no validation
    is performed if not provided.
    """
    parser = argparse.ArgumentParser(
        description="Train the ProTeCt few-shot video semantic segmentation model"
    )
    # ---- Data paths ----
    parser.add_argument("--train-images", required=True, help="Root directory for training images")
    parser.add_argument("--train-masks", required=True, help="Root directory for training labels")
    parser.add_argument("--val-images", default=None, help="Root directory for validation images (optional)")
    parser.add_argument("--val-masks", default=None, help="Root directory for validation labels (optional)")

    # ---- Output ----
    parser.add_argument("--output-dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume training from")

    # ---- Training hyperparameters ----
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    # ---- Model configuration ----
    parser.add_argument("--height", type=int, default=540, help="Input image height")
    parser.add_argument("--width", type=int, default=960, help="Input image width")
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument(
        "--backbone", choices=["resnet50", "resnet101"], default="resnet50"
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Disable ImageNet pretrained weights"
    )

    # ---- Validation frequency ----
    parser.add_argument(
        "--val-interval", type=int, default=2,
        help="Run validation every N epochs"
    )

    return parser.parse_args()


# =====================================================================
# Main Function
# =====================================================================

def main() -> None:
    args = parse_args()

    # ---- Create output directory ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build configuration ----
    config = SegmentationConfig(
        input_height=args.height,
        input_width=args.width,
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
    )

    # ---- Build DataLoaders ----
    train_loader = build_loader(
        args.train_images, args.train_masks, config,
        batch_size=args.batch_size,
        is_training=True,
        use_augmentation=True,
        num_workers=args.num_workers,
    )

    val_loader: Optional[DataLoader] = None
    if args.val_images and args.val_masks:
        val_loader = build_loader(
            args.val_images, args.val_masks, config,
            batch_size=args.batch_size,
            is_training=True,          # Still uses training-mode sliding-window sampling
            use_augmentation=False,    # But data augmentation is turned off
            num_workers=args.num_workers,
        )

    # ---- Initialize model, optimizer, scheduler, loss function ----
    model = FewShotSegmenter(config).to(config.device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Cosine annealing: learning rate smoothly decays from lr to eta_min
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index).to(config.device)

    # ---- Resume training (if --resume is specified) ----
    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, config.device)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_miou = float(checkpoint.get("best_miou", 0.0))
        print(f"Resumed training from epoch {start_epoch}, current best mIoU: {best_miou:.4f}")

    # ==================================================================
    # Training Main Loop
    # ==================================================================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, masks in progress:
            # ---- Transfer data to GPU ----
            images = images.to(config.device, non_blocking=True)
            masks = masks.to(config.device, non_blocking=True)

            # ---- Split into support and query ----
            support_image, support_mask = images[:, 0], masks[:, 0]
            query_image, query_mask = images[:, 1], masks[:, 1].squeeze(1)

            # ---- Mask perturbation (critical! simulates prediction noise at inference) ----
            previous_mask = perturb_mask(support_mask)

            # ---- Forward pass ----
            logits, _ = model(support_image, support_mask, query_image, previous_mask)

            # ---- Loss computation ----
            loss = criterion(logits, query_mask)

            # ---- Backward pass ----
            optimizer.zero_grad(set_to_none=True)  # set_to_none is slightly faster than zero_()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # ---- Progress display ----
            running_loss += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                best=f"{best_miou:.4f}",
            )

        # ---- End of epoch ----
        scheduler.step()

        # ---- Validation ----
        current_miou = None
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            current_miou, _ = validate(model, val_loader, config)
            print(f"Validation mIoU: {current_miou:.4f}")
            if current_miou > best_miou:
                best_miou = current_miou
                print(f"  -> New best model! (mIoU: {best_miou:.4f})")

        # ---- Save checkpoint ----
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "config": config.__dict__,
        }
        # Always save the latest model
        torch.save(checkpoint, output_dir / "last_model.pth")
        # Save the best model separately
        if current_miou is not None and current_miou >= best_miou:
            torch.save(checkpoint, output_dir / "best_model.pth")

    print(f"Training complete! Best mIoU: {best_miou:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
