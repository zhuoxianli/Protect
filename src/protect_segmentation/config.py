"""
ProTeCt Model Configuration Module
====================================

Centralized management of all hyperparameters, class definitions, and color mappings
needed for reproducible experiments. The release version contains no local absolute
paths — data and output paths are passed via script command-line arguments.

Usage::

    from protect_segmentation import SegmentationConfig

    # Use default configuration (540x960, 11 classes, ResNet50)
    config = SegmentationConfig()

    # Customize selected parameters
    config = SegmentationConfig(
        input_height=480,
        input_width=640,
        num_classes=19,       # Full Cityscapes 19 classes
        backbone="resnet101",
    )

    # Device auto-selects CUDA / CPU
    device = config.device
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch


@dataclass
class SegmentationConfig:
    """Unified configuration dataclass for the ProTeCt model and training.

    All fields have sensible defaults; users may override as needed.
    Using ``dataclass`` rather than a plain dict provides three advantages:
    (1) IDE auto-completion and type checking for field names;
    (2) support for ``@property`` (e.g. automatic device selection);
    (3) convenient serialization / deserialization.

    Attributes:
        input_height: Input image height in pixels. All inputs are resized to this size.
        input_width: Input image width in pixels.
        num_classes: Total number of semantic classes, including background (index 0).
        backbone: ResNet variant name from torchvision. Supports ``resnet50`` and ``resnet101``.
        pretrained: Whether to load ImageNet pretrained weights. Set to False for offline environments.
        replace_stride_with_dilation: Controls whether ResNet layer2/3/4 use dilated convolutions
            instead of stride. Default ``(False, False, True)`` keeps layer4 at stride=16
            (output resolution = 1/16 of input), which is critical for segmentation —
            overly small feature maps lose fine-grained object details.
        batch_size: Training batch size.
        learning_rate: AdamW initial learning rate.
        weight_decay: AdamW weight decay coefficient (L2 regularization strength).
        num_epochs: Total training epochs.
        ignore_index: Label pixel value to ignore in the loss function (typically 255 for "don't care" regions).
        class_map: Mapping from class name to ID. ID 0 is always reserved for background.
        colors: RGB visualization color per class, indexed by class ID. Background is typically black (0,0,0).
    """

    # ---- Input / Output Size ----
    input_height: int = 540
    input_width: int = 960

    # ---- Semantic Classes ----
    num_classes: int = 11

    # ---- Encoder (Backbone) ----
    backbone: str = "resnet50"
    pretrained: bool = True
    replace_stride_with_dilation: Sequence[bool] = (False, False, True)
    # Note: the three booleans correspond to ResNet layer2, layer3, and layer4 respectively.
    # layer2 stride=8  — keep stride=8  (segmentation benefits from higher resolution)
    # layer3 stride=16 — keep stride=16
    # layer4 stride=32 — use dilation=2, keeps stride=16 (larger receptive field without resolution loss)

    # ---- Training Hyperparameters ----
    batch_size: int = 2
    learning_rate: float = 5e-5          # 5e-5, small LR suits pretrained backbones
    weight_decay: float = 1e-4           # 1e-4
    num_epochs: int = 100
    ignore_index: int = 255              # Standard ignore value, consistent with Cityscapes and similar datasets

    # ---- Class Name Mapping ----
    class_map: Dict[str, int] = field(
        default_factory=lambda: {
            "background": 0,              # no class / ignored region
            "building": 1,                # building
            "fence": 2,                   # fence
            "person": 3,                  # person / pedestrian
            "pole": 4,                    # utility pole / lamppost
            "road": 5,                    # road
            "sidewalk": 6,                # sidewalk
            "vegetation": 7,              # vegetation
            "car": 8,                     # car
            "wall": 9,                    # wall
            "traffic sign": 10,           # traffic sign
        }
    )

    # ---- Visualization Color Mapping (RGB, 0-255) ----
    colors: List[List[int]] = field(
        default_factory=lambda: [
            [0, 0, 0],                    # 0: background - black
            [70, 70, 70],                 # 1: building - dark gray
            [190, 153, 153],              # 2: fence - light brown
            [0, 0, 255],                  # 3: person - red
            [153, 153, 153],              # 4: pole - gray
            [128, 64, 128],               # 5: road - purple
            [244, 35, 232],               # 6: sidewalk - magenta
            [107, 142, 35],               # 7: vegetation - green
            [0, 0, 142],                  # 8: car - dark blue
            [102, 102, 156],              # 9: wall - gray-blue
            [220, 220, 0],                # 10: traffic sign - yellow
        ]
    )

    # ===== Convenience properties below (not dataclass fields) =====

    @property
    def device(self) -> torch.device:
        """Auto-select available device: prefer CUDA GPU, fall back to CPU.

        Designed as ``@property`` rather than a fixed value in ``__init__`` for two reasons:
        1. The ``CUDA_VISIBLE_DEVICES`` environment variable can be changed at test time;
        2. No need to recreate the config object when switching devices.

        In multi-GPU environments where a specific card is needed, override manually:
        ``config._device = torch.device("cuda:1")``
        """
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
