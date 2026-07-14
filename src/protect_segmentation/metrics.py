"""
Segmentation Metric Computation
===============================

Provides standard evaluation metrics for semantic segmentation:
**Intersection over Union (IoU)** and **mean IoU (mIoU)**.

Computation Flow::

    pred, target   (H, W numpy arrays)
         |
         v
    intersection_and_union(pred, target, num_classes, ignore_index)
         |
         v
    (intersections, unions)   # Per-class cumulative statistics
         |
         v
    mean_iou(intersections, unions)
         |
         v
    (miou, per_class_ious)    # Final metrics

Design Rationale
----------------
Separating "statistic computation" from "metric aggregation" into two independent
functions allows accumulating intersection / union across many frames in an
inference loop and computing the global mIoU once at the end.

Usage::

    from protect_segmentation.metrics import intersection_and_union, mean_iou

    # Accumulate across frames
    total_inter, total_union = None, None
    for pred, target in frames:
        inter, union = intersection_and_union(pred, target, num_classes=11)
        if total_inter is None:
            total_inter, total_union = inter, union
        else:
            total_inter += inter
            total_union += union

    # Aggregate
    miou, per_class_ious = mean_iou(total_inter, total_union)
    print(f"mIoU: {miou:.4f}")
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def intersection_and_union(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-class intersection and union pixel counts between prediction and ground truth.

    These are the fundamental statistics for semantic segmentation evaluation.
    For each class c:

        intersection[c] = number of pixels predicted as c AND labeled as c
        union[c]        = number of pixels predicted as c OR labeled as c

    Label pixels with value ``ignore_index`` are excluded (do not count toward
    any class), which is essential for datasets like Cityscapes that label
    certain regions as "don't care".

    Args:
        pred: Predicted mask, shape ``[H, W]``, values are class IDs (int).
        target: Ground-truth label mask, shape ``[H, W]``, values are class IDs (int).
        num_classes: Total number of classes (including background).
        ignore_index: Pixel value in the labels to ignore; default 255.

    Returns:
        (intersections, unions) tuple.
        - intersections: ``[num_classes]`` float64 array.
        - unions: ``[num_classes]`` float64 array.
        Both are float64 to prevent overflow during accumulation.
    """
    # ---- Flatten and filter ----
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    # Exclude pixels labeled as ignore
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    intersections = np.zeros(num_classes, dtype=np.float64)
    unions = np.zeros(num_classes, dtype=np.float64)

    # ---- Per-class statistics ----
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        intersections[class_id] = np.logical_and(pred_mask, target_mask).sum()
        unions[class_id] = np.logical_or(pred_mask, target_mask).sum()

    return intersections, unions


def mean_iou(
    intersections: np.ndarray,
    unions: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute mIoU and per-class IoU from accumulated intersection / union statistics.

    Per-class IoU is defined as ``intersection / union``.
    If a class's union is 0 (i.e. the class never appears in any frame of the
    dataset), its IoU is set to NaN and excluded from the mIoU computation.

    Args:
        intersections: Per-class cumulative intersection pixel counts, ``[num_classes]``.
        unions: Per-class cumulative union pixel counts, ``[num_classes]``.

    Returns:
        (miou, per_class_ious) tuple.
        - miou: Mean IoU across all valid classes (float).
        - per_class_ious: ``[num_classes]`` per-class IoU; absent classes are NaN.
    """
    # Suppress divide-by-zero warnings — NaN on union==0 is expected behavior
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = intersections / unions

    # Flag absent classes
    ious[unions == 0] = np.nan

    # Compute mean only over valid classes
    valid = ~np.isnan(ious)
    miou = float(np.mean(ious[valid])) if np.any(valid) else 0.0
    return miou, ious
