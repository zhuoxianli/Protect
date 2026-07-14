"""
Sequence Dataset Loading and Augmentation
==========================================

Provides the ``DSECFewShotDataset`` class for reading DSEC / DDD17-style preprocessed
sequence data. During training, a sliding window generates (support, query) pairs; during
inference, full sequences are returned.

Data Augmentation Policy
------------------------
Training uses the albumentations library for synchronized image and mask augmentation:

1. **Horizontal Flip** (p=0.5): improves left-right symmetry invariance.
2. **Shift / Scale / Rotate** (p=0.5): simulates slight camera shake and viewpoint changes.
3. **Noise / Brightness / Motion Blur** (p=0.3, choose one):
   - Gaussian noise: simulates sensor noise.
   - Random brightness/contrast: simulates lighting changes.
   - Motion blur: simulates streaking during fast motion.

*Augmentation strength is kept restrained* — overly aggressive spatial transforms
disrupt the geometric relationship between video frames and harm the learning
of temporal consistency modules.

Usage::

    from protect_segmentation import SegmentationConfig
    from protect_segmentation.dataset import DSECFewShotDataset

    config = SegmentationConfig()

    # Training mode
    train_dataset = DSECFewShotDataset(
        image_root="/data/train_images",
        mask_root="/data/train_masks",
        sequence_length=2,
        is_training=True,
        config=config,
    )

    # Inference mode
    test_dataset = DSECFewShotDataset(
        image_root="/data/test_images",
        mask_root="/data/test_masks",
        is_training=False,
        config=config,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SegmentationConfig


# =====================================================================
# Global optimization: disable OpenCV internal multi-threading
# =====================================================================
#
# Reason: when PyTorch's DataLoader uses multiple worker processes (num_workers > 0)
# for parallel data loading, if OpenCV inside each worker also spawns threads,
# thread contention results — CPU utilization fluctuates sharply and throughput
# actually degrades. Two lines are disabled: system thread count and OpenCL acceleration.

cv2.setNumThreads(0)        # Prevent OpenCV from using extra threads
cv2.ocl.setUseOpenCL(False)  # Disable OpenCL (PyTorch already uses GPU, no need for OpenCV acceleration)


class DSECFewShotDataset(Dataset):
    """Sequence dataset for few-shot video semantic segmentation.

    Directory Scanning Rules
    ------------------------
    Each subdirectory is treated as an independent video sequence. Frame files are
    sorted by filename to ensure correct temporal order (filenames are typically
    zero-padded frame numbers, e.g. 000000.png).

    Training Mode (``is_training=True``):
        - Sliding window sampling with window size ``sequence_length`` and stride=1.
        - Sequences shorter than ``sequence_length`` are skipped.
        - Returns ``(image_stack, mask_stack)`` with shape ``[T, C, H, W]``.

    Inference Mode (``is_training=False``):
        - Returns the full sequence.
        - Additionally returns a list of frame paths for matching filenames when saving results.
        - Returns ``(image_stack, mask_stack, frame_paths)``.

    Mask Path Resolution
    --------------------
    Because different datasets are organized differently, the code tries two common
    mask path structures:

    1. ``masks/sequence_name/11classes/frame.png`` (DSEC preprocessing default)
    2. ``masks/sequence_name/frame.png`` (simple flat structure)

    If mask_root is None or no matching file is found, an all-zero mask is returned
    (suitable for unlabeled inference).

    Attributes:
        image_root: Root directory for image data.
        mask_root: Root directory for labels (optional).
        sequence_length: Training sliding window length.
        is_training: Whether in training mode.
        config: Model and data configuration.
    """

    # Supported image file extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(
        self,
        image_root: Union[str, Path],
        mask_root: Optional[Union[str, Path]] = None,
        sequence_length: int = 2,
        is_training: bool = True,
        config: Optional[SegmentationConfig] = None,
    ) -> None:
        """
        Args:
            image_root: Root directory for images; each subdirectory is a sequence.
            mask_root: Root directory for labels. If None, all-zero masks are returned
                (unlabeled inference mode).
            sequence_length: Training sliding window length; 2 (support + query) is recommended.
            is_training: Whether to enable sliding window sampling and data augmentation.
            config: Model configuration object containing target size and other parameters.
        """
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root) if mask_root is not None else None
        self.sequence_length = sequence_length
        self.is_training = is_training
        self.config = config or SegmentationConfig()

        # Data augmentation pipeline (training only)
        self.augment = self._build_augmentations() if is_training else None

        # Scan data directory and build sample index
        self.samples = self._scan_sequences()

    def _build_augmentations(self) -> A.Compose:
        """Build the albumentations data augmentation pipeline.

        Augmentation policy rationale:
            - HorizontalFlip (p=0.5):
              Horizontal flip is the only safe symmetric transform; it does not
              break temporal continuity.
            - ShiftScaleRotate (p=0.5, small parameters):
              Simulates slight camera shake. Translation capped at 5%, scale at ±5%,
              rotation at ±10°. Uses constant padding (0); mask also padded with 0
              (background).
            - OneOf[GaussNoise / Brightness / MotionBlur] (p=0.3):
              Randomly picks one of three photometric / noise perturbations to
              simulate sensor and lighting variation.

        Returns:
            albumentations.Compose object; must be called with both image and mask.
        """
        return A.Compose(
            [
                # ---- Spatial transforms ----
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,       # ±5% translation
                    scale_limit=0.05,       # ±5% scaling
                    rotate_limit=10,        # ±10° rotation
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,                 # image padding: black
                    mask_value=0,            # mask padding: background
                    p=0.5,
                ),
                # ---- Photometric / noise perturbations (pick one) ----
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0)),         # variance 10~50
                        A.RandomBrightnessContrast(p=1.0),             # brightness / contrast
                        A.MotionBlur(blur_limit=3),                    # small-kernel motion blur
                    ],
                    p=0.3,
                ),
            ]
        )

    def _scan_sequences(self) -> List[List[Path]]:
        """Scan all sequence directories and build the sample index list.

        Training mode:
            Each sequence produces (len - sequence_length + 1) sliding-window samples.
            Sequences that are too short (length < sequence_length) are skipped.

        Inference mode:
            Each sequence is treated as one holistic sample; the full frame list is returned.

        Returns:
            List of sample indices. Each element is a list of Path objects
            (one window's or one full sequence's frame paths).

        Raises:
            FileNotFoundError: If the image root directory does not exist.
            RuntimeError: If no image files were found.
        """
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root directory does not exist: {self.image_root}")

        # Subdirectories sorted by name = individual sequences
        sequence_dirs = sorted(p for p in self.image_root.iterdir() if p.is_dir())
        samples: List[List[Path]] = []

        for sequence_dir in sequence_dirs:
            # rglob("*") recursively finds all files (supports nested structure);
            # sort by path to ensure temporal continuity
            frames = sorted(
                p
                for p in sequence_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in self.image_extensions
            )
            if not frames:
                continue

            if self.is_training:
                # Sliding window: every consecutive sequence_length frames as one sample
                if len(frames) < self.sequence_length:
                    continue  # Sequence too short, skip
                for start in range(0, len(frames) - self.sequence_length + 1):
                    samples.append(frames[start : start + self.sequence_length])
            else:
                # Full sequence
                samples.append(frames)

        if not samples:
            raise RuntimeError(f"No image sequences found under {self.image_root}")

        return samples

    def _candidate_mask_paths(self, image_path: Path) -> Sequence[Path]:
        """Generate candidate mask paths from an image path.

        Supports two common directory structures:

        Format 1 (DSEC preprocessing default):
            image:  train_images/zurich_city_01/000000.png
            mask:   train_masks/zurich_city_01/11classes/000000.png

        Format 2 (flat structure):
            image:  train_images/seq_01/000000.png
            mask:   train_masks/seq_01/000000.png

        Args:
            image_path: Full path to an image file.

        Returns:
            Tuple of candidate mask paths, ordered by priority.
        """
        if self.mask_root is None:
            return []

        sequence_name = image_path.parent.name
        mask_name = image_path.with_suffix(".png").name

        return (
            self.mask_root / sequence_name / "11classes" / mask_name,
            self.mask_root / sequence_name / mask_name,
        )

    def _load_pair(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load an (image, mask) pair.

        Images are read as RGB uint8. Masks are read as grayscale; pixel values
        are class IDs. If no mask file is found, an all-zero background mask is returned.

        Args:
            image_path: Path to the image file.

        Returns:
            (image, mask) tuple, both resized to the config-specified dimensions.
            image: ``[H, W, 3]`` uint8 RGB.
            mask: ``[H, W]`` uint8 class IDs.

        Raises:
            ValueError: If the image file cannot be read.
        """
        # ---- Read image ----
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ---- Read mask (try candidates in order) ----
        mask = None
        for mask_path in self._candidate_mask_paths(image_path):
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                break

        if mask is None:
            # Unlabeled mode: all zero (background)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # ---- Resize to unified dimensions ----
        target_size = (self.config.input_width, self.config.input_height)
        if image.shape[:2] != (self.config.input_height, self.config.input_width):
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
            # Use nearest-neighbor for mask to preserve class IDs
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        return image, mask

    def __getitem__(self, index: int):
        """Retrieve the index-th sample.

        Training mode returns:
            ``(image_stack, mask_stack)``
            - image_stack: ``[T, C, H, W]`` float32, range [0, 1]
            - mask_stack: ``[T, 1, H, W]`` int64

        Inference mode returns:
            ``(image_stack, mask_stack, frame_paths)``
            - frame_paths: list of strings — full paths to each frame
        """
        frame_paths = self.samples[index]
        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        for frame_path in frame_paths:
            image, mask = self._load_pair(frame_path)

            # ---- Data augmentation (training mode only) ----
            if self.augment is not None:
                augmented = self.augment(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # ---- Format conversion ----
            # Image: HWC uint8 → CHW float32, range [0, 1]
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            # Mask: HW int → [1, H, W] int64 (extra channel dim for convenience downstream)
            mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)

            images.append(image_tensor)
            masks.append(mask_tensor)

        # ---- Stack along the temporal dimension ----
        image_stack = torch.stack(images, dim=0)   # [T, C, H, W]
        mask_stack = torch.stack(masks, dim=0)      # [T, 1, H, W]

        if self.is_training:
            return image_stack, mask_stack

        # Inference mode additionally returns paths for matching filenames when saving results
        return image_stack, mask_stack, [str(p) for p in frame_paths]

    def __len__(self) -> int:
        """Total number of samples.

        Training mode: sum of sliding windows across all sequences.
        Inference mode: number of sequences.
        """
        return len(self.samples)
