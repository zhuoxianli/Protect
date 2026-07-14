"""
ProTeCt — Prototype-driven Temporal Consistency for Few-Shot Video Semantic Segmentation
========================================================================================

Usage::

    from protect_segmentation import SegmentationConfig, FewShotSegmenter

    # Create configuration
    config = SegmentationConfig(
        input_height=540,
        input_width=960,
        num_classes=11,
        backbone="resnet50",
    )

    # Instantiate the model
    model = FewShotSegmenter(config)
    model.to(config.device)

    # Load weights
    from protect_segmentation.model import load_checkpoint
    load_checkpoint(model, "checkpoints/best_model.pth", config.device)
"""

from .config import SegmentationConfig
from .model import FewShotSegmenter

__all__ = ["SegmentationConfig", "FewShotSegmenter"]
