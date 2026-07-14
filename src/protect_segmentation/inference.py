"""
Memory Bank Inference Engine
=============================

Provides the ``MemoryBankEngine`` class for sequence inference with a prototype memory bank.

Core Idea
---------

In few-shot video segmentation, only the first frame is annotated (support set). As time
progresses, the scene gradually changes (lighting, viewpoint, motion), making the first
frame's prototypes increasingly inaccurate. The memory bank mitigates this drift by
accumulating prototypes from subsequent high-confidence frames.

Memory Bank Policy
------------------

1. **Initialization**: extract foreground prototypes from the first frame's GT mask
   as the fixed base entry in the memory bank.
2. **Frame-by-Frame Prediction**: use all historical prototypes in the memory bank
   plus the previous frame's predicted mask to predict the current frame.
3. **Conditional Update**: every ``update_interval`` frames, check:
   - Compute the **mean confidence** of the current prediction (average of Softmax max probability).
   - If confidence >= ``confidence_threshold``, extract prototypes from the current
     predicted mask and add them to the memory bank.
4. **Capacity Control**: the memory bank keeps at most ``max_items`` prototype groups.
   On overflow, retain the first frame (index 0) and drop the oldest dynamic prototype (index 1).

Design Considerations
---------------------

- **Why not update every frame?** Consecutive frames are highly correlated; frequent
  updates lead to redundant prototypes (nearly identical) and erroneous predictions
  may contaminate the memory bank. Interval updates give the model time to "verify"
  the current prediction quality.

- **Why keep the first frame?** The first frame is the only definitively correct
  annotation. Keeping it permanently provides a "corrective anchor" if subsequent
  predictions drift.

- **Why a confidence threshold?** Low-confidence frames (e.g. severe motion blur)
  produce unreliable predictions. Adding them to the memory bank would introduce
  erroneous prototypes and harm future predictions.

Usage::

    from protect_segmentation import FewShotSegmenter, SegmentationConfig
    from protect_segmentation.inference import MemoryBankEngine, MemoryBankConfig

    config = SegmentationConfig()
    model = FewShotSegmenter(config).to(config.device).eval()

    engine = MemoryBankEngine(model, MemoryBankConfig(
        update_interval=5,
        max_items=3,
        confidence_threshold=0.92,
        use_feature_flip=True,
    ))

    # Initialize
    first_image = ...
    first_mask = ...
    features = model.extract_features(first_image)
    engine.initialize(features, first_mask)

    # Predict frame by frame
    previous_mask = first_mask.float()
    for frame in remaining_frames:
        pred_mask = engine.predict(frame, previous_mask, frame_index)
        previous_mask = pred_mask.float()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from .model import FewShotSegmenter, PrototypeList


@dataclass
class MemoryBankConfig:
    """Parameters controlling the prototype memory bank update behavior.

    Attributes:
        update_interval: Memory bank update interval (in frames).
            Set to 0 to completely disable dynamic updates and use only
            the first-frame prototypes. The default of 5 frames is an
            empirically determined value — frequent enough to adapt to
            scene changes but not so frequent that redundant prototypes
            accumulate.
        max_items: Maximum number of prototype groups kept in the memory bank.
            The first frame is always retained, so at most
            max_items - 1 dynamic prototype groups are stored.
            Set to 1 to effectively disable updates.
        confidence_threshold: Minimum confidence for adding to the memory bank.
            The mean Softmax confidence across all pixels of the current
            prediction must be >= this threshold for its prototypes to be
            added. Higher values → cleaner memory bank but fewer updates.
            Experiments show 0.92 provides a good balance between cleanliness
            and coverage.
        use_feature_flip: Whether to enable horizontal-flip TTA (test-time augmentation).
            Flipping is done at the feature level (not pixel level) to avoid
            interpolation loss. Adds roughly 2x overhead on the feature-matching
            portion but yields 1–2% mIoU improvement.
    """

    update_interval: int = 5
    max_items: int = 3
    confidence_threshold: float = 0.92
    use_feature_flip: bool = True


class MemoryBankEngine:
    """Sequence inference engine using support prototypes and confidence history.

    The first-frame label is used to initialize the memory bank. Thereafter,
    every ``update_interval`` frames, if the mean confidence of the current
    prediction is high enough, new prototypes are extracted from the current
    predicted mask and added to the memory bank.

    Attributes:
        model: Loaded ProTeCt segmentation model.
        config: Memory bank behavior configuration.
        memory_bank: Current list of prototype history (updated as inference progresses).
    """

    def __init__(
        self,
        model: FewShotSegmenter,
        config: Optional[MemoryBankConfig] = None,
    ) -> None:
        """
        Args:
            model: Initialized ProTeCt model instance.
            config: Memory bank configuration. Uses defaults if None.
        """
        self.model = model
        self.config = config or MemoryBankConfig()
        self.memory_bank: List[PrototypeList] = []

    def reset(self) -> None:
        """Clear the memory bank.

        Call before inferring on a new sequence to ensure prototypes from a
        previous sequence do not leak.
        """
        self.memory_bank = []

    @torch.no_grad()
    def initialize(
        self,
        support_features: List[torch.Tensor],
        support_mask: torch.Tensor,
    ) -> None:
        """Initialize the memory bank from the first-frame annotation (support).

        Extracts prototypes for all foreground classes in the first frame and
        stores them as the first entry. This entry is kept permanently — even
        upon capacity overflow, it is never evicted.

        Args:
            support_features: 5-level encoder feature list from the first frame.
            support_mask: Ground-truth label mask for the first frame.
        """
        prototypes = self.model.extract_prototypes_from_features(
            support_features, support_mask
        )
        # prototypes shape: List[PrototypeList]; the outermost level is batch.
        # Take the sample at batch_index=0.
        self.memory_bank = [prototypes[0]]

    @torch.no_grad()
    def predict(
        self,
        query_image: torch.Tensor,
        previous_mask: torch.Tensor,
        frame_index: int,
        query_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict one frame and update the memory bank when conditions are met.

        Full steps:
            1. Encode the current frame (if pre-encoded features are not provided).
            2. Forward pass using memory bank prototypes + previous-frame mask.
            3. Softmax → argmax → predicted mask.
            4. Check whether the memory bank should and can be updated.
            5. If update conditions are met, extract prototypes from the current
               prediction and add them to the memory bank.

        Args:
            query_image: Current frame image ``[1, C, H, W]``.
            previous_mask: Previous-frame predicted mask ``[1, 1, H, W]``.
            frame_index: Zero-based index of the current frame in the sequence.
            query_features: Pre-encoded current-frame features. If
                ``model.extract_features()`` was already called externally,
                passing it here saves computation. If None, encoding is
                done internally.

        Returns:
            Predicted mask ``[1, 1, H, W]`` int64.
        """
        # ---- 1. Encode ----
        if query_features is None:
            query_features = self.model.extract_features(query_image)

        # ---- 2. Forward pass ----
        logits = self.model.predict_with_memory(
            query_features,
            previous_mask,
            self.memory_bank,
            use_feature_flip=self.config.use_feature_flip,
        )

        # ---- 3. Softmax → predicted mask ----
        probabilities = torch.softmax(logits, dim=1)
        # Upsample to original image resolution
        probabilities = F.interpolate(
            probabilities,
            size=query_image.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        pred_mask = torch.argmax(probabilities, dim=1, keepdim=True)

        # ---- 4. Decide whether to update the memory bank ----
        should_update = (
            frame_index > 0                                     # Not the first frame (GT already used for init)
            and self.config.update_interval > 0                 # Dynamic updates enabled
            and frame_index % self.config.update_interval == 0  # Multiple of update interval
        )

        if should_update:
            # Compute the mean confidence of the current prediction
            confidence = probabilities.max(dim=1).values.mean()
            if confidence.item() >= self.config.confidence_threshold:
                self._update_memory(query_features, pred_mask)

        return pred_mask

    @torch.no_grad()
    def _update_memory(
        self,
        query_features: List[torch.Tensor],
        pred_mask: torch.Tensor,
    ) -> None:
        """Extract prototypes from a high-confidence prediction and append to the memory bank.

        Steps:
            1. Resize the predicted mask to the feature-map resolution (nearest neighbor).
            2. Extract prototypes for all foreground classes in the current frame.
            3. Append to the memory bank.
            4. If capacity is exceeded, drop the oldest dynamic prototype (index 1),
               preserving the first-frame support prototype at index 0.

        Args:
            query_features: Encoder features for the current frame.
            pred_mask: Predicted mask for the current frame (full resolution).
        """
        # Resize mask to feature-map resolution
        resized_mask = F.interpolate(
            pred_mask.float(),
            size=query_features[4].shape[-2:],
            mode="nearest",
        )

        # Extract prototypes
        prototypes = self.model.extract_prototypes_from_features(
            query_features, resized_mask
        )[0]

        # If no foreground prototypes were extracted (all background), skip
        if not prototypes:
            return

        self.memory_bank.append(prototypes)

        # Capacity control: on overflow, drop the oldest dynamic prototype
        # (pop(1) keeps index=0, the first-frame support prototypes, intact)
        if len(self.memory_bank) > self.config.max_items:
            self.memory_bank.pop(1)
