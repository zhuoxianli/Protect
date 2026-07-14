"""
DSEC Data Preprocessing
========================

Processes the raw DSEC dataset into a format directly usable by the ProTeCt model.

Background
----------
The DSEC dataset includes two independent sensors:
1. **RGB Camera** (1440x1080 resolution, approx. 20 Hz frame rate)
2. **Event Camera** (640x480 resolution, microsecond-level temporal resolution)

The two sensors have different fields of view, resolutions, and intrinsic parameters.
Semantic segmentation labels are annotated in the **event camera** coordinate system
(640x440 effective region).

Preprocessing Pipeline
----------------------
For each RGB image:

1. **Perspective Warp**:
   Using the calibrated homography matrix H, reproject the 1440x1080 RGB image
   into the event camera's 640x480 coordinate system.
   ``H = K_rgb @ inv(K_event)``

2. **Crop**:
   Take the first 440 rows (discard the bottom 40 rows of event camera noise region).

3. **Resize**:
   Resize the 640x440 cropped image to 960x540 (training resolution).

Labels are already aligned to the event camera coordinate system; only a resize
operation is needed (nearest-neighbor interpolation to preserve class IDs).

Usage Instructions
------------------
1. Modify the path constants in this file (``SOURCE_ROOT``, ``SOURCE_LABEL_ROOT``,
   ``TARGET_ROOT``, ``TARGET_LABEL_ROOT``).
2. Run: ``python scripts/preprocess_dsec.py``

Important Notes
---------------
- The ``WARP_INVERSE_MAP`` flag is used: H is the event->RGB forward mapping,
  but we actually need the RGB->event inverse mapping, hence this flag.
- Parallel processing uses ``multiprocessing.Pool`` with num_processes = CPU cores - 2
  to avoid monopolizing all system resources.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
import argparse


# =====================================================================
# Configuration Area — Modify the following paths before use
# =====================================================================

# Root directory for raw RGB images (1440x1080)
SOURCE_ROOT = r'E:\Datasets\DSEC\APS\test'

# Root directory for raw labels (640x440, already in event camera coordinate system)
SOURCE_LABEL_ROOT = r'E:\Datasets\DSEC\test\labels'

# Output directory for preprocessed images (960x540)
TARGET_ROOT = r'E:\Datasets\DSEC\APS\test_aligned_preprocessed'

# Output directory for preprocessed labels (960x540)
TARGET_LABEL_ROOT = r'E:\Datasets\DSEC\APS\test_labels_preprocessed'

# =====================================================================
# DSEC Camera Calibration Parameters
# =====================================================================

# Target training resolution (matches input_height/input_width in config.py)
TARGET_SIZE = (960, 540)

# Event camera intrinsic matrix (640x480 resolution)
K_event = np.array([
    [583.3081203392971, 0.0, 336.83414459228516],
    [0.0, 583.3081203392971, 220.91131019592285],
    [0.0, 0.0, 1.0],
])

# RGB camera intrinsic matrix (1440x1080 resolution)
K_rgb = np.array([
    [1150.8249465165975, 0.0, 724.4121398925781],
    [0.0, 1150.8249465165975, 569.1058044433594],
    [0.0, 0.0, 1.0],
])

# Homography matrix: event camera -> RGB camera coordinate mapping
# For any event camera pixel p_event, its corresponding position in the RGB image is:
#   p_rgb = H @ p_event
# Therefore, mapping an RGB image back to event camera coordinates requires
# the inverse mapping.
H_map = K_rgb @ np.linalg.inv(K_event)


# =====================================================================
# Processing Functions
# =====================================================================

def process_image(args):
    """Process a single RGB image: Warp -> Crop -> Resize.

    Uses a perspective transform to align the high-resolution RGB image
    into the event camera coordinate system, then crops and resizes to
    the training resolution.

    Args:
        args: (src_path, dst_path) tuple — source image path and target
              save path. If the target file already exists, it is skipped
              (supports resumable processing).
    """
    src_path, dst_path = args

    # Skip already-processed files (resumable)
    if os.path.exists(dst_path):
        return

    img = cv2.imread(src_path)
    if img is None:
        print(f"Warning: Cannot read image {src_path}, skipping")
        return

    # 1. Warp (perspective transform): RGB 1440x1080 -> Event 640x480
    # WARP_INVERSE_MAP means H defines a "target -> source" mapping,
    # but we are doing "source -> target".
    img_warped = cv2.warpPerspective(
        img, H_map, (640, 480),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    # 2. Crop: keep only the first 440 rows (valid label region)
    img_cropped = img_warped[:440, :]

    # 3. Resize to training resolution 960x540
    img_final = cv2.resize(
        img_cropped, TARGET_SIZE, interpolation=cv2.INTER_LINEAR
    )

    # Save
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, img_final)


def process_label(args):
    """Process a single label: Resize.

    Labels are already in the event camera coordinate system (640x440);
    only a resize to the training resolution is needed.
    Uses INTER_NEAREST interpolation to preserve integer class IDs.

    Args:
        args: (src_path, dst_path) tuple.
              If the target file already exists, it is skipped.
    """
    src_path, dst_path = args

    if os.path.exists(dst_path):
        return

    mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Cannot read label {src_path}, skipping")
        return

    # Resize to training resolution (nearest-neighbor to preserve class IDs)
    mask_final = cv2.resize(
        mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST
    )

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, mask_final)


# =====================================================================
# Main Function
# =====================================================================

def main():
    """Scan and process all images and labels in parallel."""

    # ---- 1. Scan images ----
    print(f"Scanning image directory: {SOURCE_ROOT}")
    img_tasks = []
    for root, _, files in os.walk(SOURCE_ROOT):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(root, file)
                # Preserve the original directory structure
                rel = os.path.relpath(src, SOURCE_ROOT)
                dst = os.path.join(TARGET_ROOT, rel)
                img_tasks.append((src, dst))

    if not img_tasks:
        print("Error: No image files found! Please check the SOURCE_ROOT path.")
        return

    print(f"Found {len(img_tasks)} images, starting parallel processing (Warp + Crop + Resize)...")
    # Use CPU cores - 2 to avoid monopolizing all system resources
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(num_cores) as pool:
        list(tqdm(
            pool.imap_unordered(process_image, img_tasks),
            total=len(img_tasks),
            desc="Processing images",
        ))

    # ---- 2. Scan labels ----
    print(f"\nScanning label directory: {SOURCE_LABEL_ROOT}")
    lbl_tasks = []
    for root, _, files in os.walk(SOURCE_LABEL_ROOT):
        for file in files:
            if file.lower().endswith('.png'):
                src = os.path.join(root, file)
                rel = os.path.relpath(src, SOURCE_LABEL_ROOT)
                dst = os.path.join(TARGET_LABEL_ROOT, rel)
                lbl_tasks.append((src, dst))

    if not lbl_tasks:
        print("Warning: No label files found! Please check the SOURCE_LABEL_ROOT path.")
        print("Image processing completed (labels skipped).")
        print(f"Processed image path: {TARGET_ROOT}")
        return

    print(f"Found {len(lbl_tasks)} labels, starting parallel processing (Resize)...")
    with multiprocessing.Pool(num_cores) as pool:
        list(tqdm(
            pool.imap_unordered(process_label, lbl_tasks),
            total=len(lbl_tasks),
            desc="Processing labels",
        ))

    # ---- Done ----
    print("\n" + "=" * 50)
    print("All preprocessing completed!")
    print(f"Processed image path: {TARGET_ROOT}")
    print(f"Processed label path: {TARGET_LABEL_ROOT}")
    print("=" * 50)


if __name__ == '__main__':
    main()
