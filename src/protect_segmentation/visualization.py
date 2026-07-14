"""
Segmentation Result Visualization
==================================

Provides functions for semi-transparently overlaying class masks onto source images
and saving the results.

Color Mapping
-------------
RGB colors are assigned per class according to ``config.colors``.
Background (class 0) is left uncolored; the original image is preserved.

Blending Formula::

    output[p] = image[p]                                  (background pixels)
    output[p] = (1 - alpha) * image[p] + alpha * color[class[p]]   (foreground pixels)

Usage::

    from protect_segmentation.visualization import overlay_mask, save_overlay

    # In-memory overlay
    vis = overlay_mask(image_rgb, pred_mask, config.colors, alpha=0.5)

    # Direct read → overlay → save
    save_overlay("input/000001.png", pred_mask, "output/vis/000001.png", config.colors)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    colors: Sequence[Sequence[int]],
    alpha: float = 0.5,
) -> np.ndarray:
    """Semi-transparently overlay a class mask onto an RGB image.

    Only foreground pixels (mask > 0) are colored. Background (class 0) pixels
    are left as the original image for a cleaner visualization.

    If the mask dimensions differ from the image, the mask is automatically
    resized with nearest-neighbor interpolation (to preserve class ID values).

    Args:
        image_rgb: Original RGB image, shape ``[H, W, 3]``, dtype uint8.
        mask: Predicted / ground-truth mask, shape ``[H, W]``, values are class IDs.
        colors: Per-class RGB color list, indexed by class ID.
        alpha: Blending transparency. 0.0 = fully transparent (original image only),
            1.0 = fully opaque (color blocks only).

    Returns:
        Blended RGB image ``[H, W, 3]`` uint8.
    """
    # ---- Size alignment ----
    if mask.shape != image_rgb.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # ---- Build color mask ----
    color_mask = np.zeros_like(image_rgb)
    # Start from class 1 (skip background)
    max_class = min(len(colors), int(mask.max()) + 1)
    for class_id in range(1, max_class):
        color_mask[mask == class_id] = colors[class_id]

    # ---- Blend ----
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, color_mask, alpha, 0.0)

    # Replace only foreground regions with blended output;
    # background regions stay as the original image
    output = image_rgb.copy()
    foreground = mask > 0
    output[foreground] = blended[foreground]
    return output


def save_overlay(
    image_path: Union[str, Path],
    mask: np.ndarray,
    save_path: Union[str, Path],
    colors: Sequence[Sequence[int]],
    alpha: float = 0.5,
) -> None:
    """Read the original image, overlay the predicted mask, and save to disk.

    Convenience function that wraps the full "read → overlay → save" pipeline.
    Internally calls ``overlay_mask`` for blending and automatically creates
    the output directory if it does not exist.

    Args:
        image_path: Path to the original image file.
        mask: Predicted mask, shape ``[H, W]``.
        save_path: Output save path (PNG format recommended).
        colors: Per-class RGB color list.
        alpha: Blending transparency.

    Raises:
        ValueError: If the image file cannot be read.
    """
    # ---- Read original image ----
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image for visualization: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---- Overlay ----
    vis_rgb = overlay_mask(image_rgb, mask, colors=colors, alpha=alpha)

    # ---- Save (convert back to BGR for writing) ----
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis_bgr)
