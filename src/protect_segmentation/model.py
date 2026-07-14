"""
ProTeCt Model Definition
========================

This file contains the complete forward pass of a few-shot video semantic segmentation
model, including the following core modules:

1. **ResNetEncoder**           — 5-level feature pyramid extraction
2. **CoordinatePrototypeInteraction (C-PIM)** — Coordinate-aware prototype matching
3. **TemporalReliabilityGating (TRG)**        — Temporal reliability gating
4. **PriorGuidedASPP (PG-ASPP)**             — Prior-guided multi-scale fusion
5. **FewShotSegmenter**                      — End-to-end model wrapper

Data flow (training)::

    First-frame image+Mask ──▶ ResNetEncoder ──▶ C-PIM extract prototypes ──▶ cosine similarity matching
                                                        │
    Current-frame image ──▶ ResNetEncoder ──▶ query features ──┘  ──▶ similarity map
                                                        │
    Previous-frame Mask ──────────────────────── TRG ──┘  ──▶ PG-ASPP ──▶ Decoder ──▶ Logits

Data flow (inference, with Memory Bank)::

    Accumulated historical prototypes ──▶ C-PIM.match_prototypes ──▶ match with current frame features ──▶ similarity map ──▶ PG-ASPP ──▶ Logits

Fully compatible with the original experiments: legacy .pth weights can be loaded
directly via ``load_checkpoint``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .config import SegmentationConfig


# =====================================================================
# Type Aliases
# =====================================================================

# Each prototype is a pair of (high_level_vec, mid_level_vec), both C-dim vectors
Prototype = Tuple[torch.Tensor, torch.Tensor]
# All foreground prototypes for a single sample
PrototypeList = List[Prototype]


# =====================================================================
# 1. Encoder — Multi-scale Feature Extraction
# =====================================================================

class ResNetEncoder(nn.Module):
    """Five-level feature pyramid encoder based on torchvision ResNet.

    Built on ImageNet-pretrained ResNet50/101, using dilated convolutions to
    preserve spatial resolution in higher layers, suitable for semantic
    segmentation tasks that require fine-grained output spatial resolution.

    Output feature levels::

        f1: stride=2,  channels=64     — low-level texture/edges
        f2: stride=4,  channels=256    — low-level geometry/local patterns (used for decoder fusion)
        f3: stride=8,  channels=512    — mid-level semantics
        f4: stride=16, channels=1024   — upper-mid semantics (C-PIM mid-level)
        f5: stride=16, channels=2048   — high-level semantics (C-PIM high-level, stride 16 preserved via dilation)

    Notes:
        - Grayscale input images are replicated to 3 channels to match ImageNet pretrained expectations.
        - For multi-channel inputs (>3), only the first 3 channels are used.
    """

    # Channel counts for each layer (consistent with torchvision ResNet)
    out_channels = [64, 256, 512, 1024, 2048]

    def __init__(self, config: SegmentationConfig) -> None:
        """
        Args:
            config: Model configuration object, from which backbone name,
                    pretrained flag, and dilation settings are read.
        """
        super().__init__()
        self.config = config

        # ---- Select backbone variant ----
        if config.backbone == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if config.pretrained else None
            backbone = models.resnet101(
                weights=weights,
                replace_stride_with_dilation=list(config.replace_stride_with_dilation),
            )
        elif config.backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if config.pretrained else None
            backbone = models.resnet50(
                weights=weights,
                replace_stride_with_dilation=list(config.replace_stride_with_dilation),
            )
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}. Options: resnet50, resnet101")

        # ---- Split ResNet into sub-modules by function for easy intermediate feature extraction ----
        self.conv1 = backbone.conv1          # 7x7 conv, stride=2 -> 1/2
        self.bn1 = backbone.bn1              # BatchNorm
        self.relu = backbone.relu            # ReLU activation
        self.maxpool = backbone.maxpool      # 3x3 maxpool, stride=2 -> 1/4
        self.layer1 = backbone.layer1        # Bottleneck x3, stride=4 preserved -> 1/4
        self.layer2 = backbone.layer2        # Bottleneck x4, stride=8 -> 1/8 (or dilation-preserved 1/8)
        self.layer3 = backbone.layer3        # Bottleneck x6/23, stride=16 -> 1/16
        self.layer4 = backbone.layer4        # Bottleneck x3, stride=16 (dilation-preserved)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract a five-level feature pyramid from the input image.

        Args:
            x: Input tensor, shape ``[B, C, H, W]``.

        Returns:
            A list of 5 feature maps ``[f1, f2, f3, f4, f5]``.
        """
        # ---- Channel count adaptation ----
        if x.shape[1] == 1:
            # Single-channel (grayscale) images are replicated to three channels
            # for compatibility with pretrained RGB weights.
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            # Multi-channel inputs (e.g., RGB + events) — keep only the first 3 channels.
            x = x[:, :3]

        # ---- Forward pass, recording layer by layer ----
        x = self.conv1(x)
        x = self.bn1(x)
        f1 = self.relu(x)                    # [B, 64,   H/2,  W/2]

        x = self.maxpool(f1)
        f2 = self.layer1(x)                  # [B, 256,  H/4,  W/4]
        f3 = self.layer2(f2)                 # [B, 512,  H/8,  W/8]
        f4 = self.layer3(f3)                 # [B, 1024, H/16, W/16]
        f5 = self.layer4(f4)                 # [B, 2048, H/16, W/16]

        return [f1, f2, f3, f4, f5]


# =====================================================================
# 2. C-PIM — Coordinate-aware Prototype Interaction Module
# =====================================================================

class CoordinatePrototypeInteraction(nn.Module):
    """Coordinate-aware Prototype Interaction Module (C-PIM).

    How it works
    ------------
    1. **Coordinate Augmentation**:
       Two normalized XY coordinate channels (range [-1, 1]) are appended to the
       feature map, so that a prototype encodes not only "what this class looks
       like" but also "where in the image it tends to appear".
       E.g., "car usually appears in the lower half, road usually around the centre."

    2. **Prototype Extraction**:
       For each foreground class in the support mask, masked average pooling
       aggregates the features of the region covered by that class into a single
       vector (prototype). Both high-level (2048-d semantic) and mid-level
       (1024-d detail) prototypes are extracted.

    3. **Prototype Matching**:
       Each prototype is compared via inner product (equivalent to cosine
       similarity after L2 normalisation) with the normalised feature vector at
       every spatial position of the query feature map, then scaled by a
       temperature coefficient to amplify differences. The per-position maximum
       response across all prototypes is taken to produce a similarity map.
       Output has 2 channels: [high_sim, mid_sim].

    Design rationale
    ----------------
    - Coordinate channels make matching position-sensitive, producing tighter activation regions.
    - High- and mid-level matching complements semantic and detail information.
    - A memory bank is supported at inference time — a list of historically accumulated prototypes can be passed in.

    Notes
    -----
    - The layer names ``coord_conv_high`` / ``coord_conv_low`` are preserved for full compatibility with legacy checkpoints.
    """

    def __init__(
        self,
        high_level_channels: int,      # Number of high-level feature channels (typically 2048)
        mid_level_channels: int,       # Number of mid-level feature channels (typically 1024)
        high_embed_channels: int = 256, # High-level embedding dimension after projection
        mid_embed_channels: int = 128,  # Mid-level embedding dimension after projection
        temperature: float = 20.0,      # Temperature coefficient for cosine similarity (higher -> sharper matching)
    ) -> None:
        super().__init__()
        self.temperature = temperature

        # 1x1 convolution projects [feature_channels + 2 coordinates] into a unified embedding space.
        # +2 accounts for the appended X and Y coordinate channels.
        self.coord_conv_high = nn.Conv2d(
            high_level_channels + 2, high_embed_channels, kernel_size=1
        )
        self.coord_conv_low = nn.Conv2d(
            mid_level_channels + 2, mid_embed_channels, kernel_size=1
        )

    @staticmethod
    def _add_coordinate_channels(feature: torch.Tensor) -> torch.Tensor:
        """Append normalised XY coordinate channels to the feature map.

        Coordinate range is [-1, 1], with top-left as (-1, -1) and bottom-right
        as (1, 1). This normalisation keeps coordinate values on the same order
        of magnitude as feature magnitudes, making 1x1 convolution fusion
        straightforward.

        Args:
            feature: shape ``[B, C, H, W]``.

        Returns:
            Tensor of shape ``[B, C+2, H, W]``.
        """
        batch_size, _, height, width = feature.shape
        device, dtype = feature.device, feature.dtype

        # Generate linearly-spaced coordinate vectors from -1 to 1.
        y_range = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x_range = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)

        # Build grid. indexing="ij" ensures the first dim is Y (rows) and the second is X (columns).
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")

        # Expand to the batch dimension.
        x_grid = x_grid.expand(batch_size, 1, height, width)
        y_grid = y_grid.expand(batch_size, 1, height, width)

        return torch.cat([feature, x_grid, y_grid], dim=1)

    def project_features(
        self,
        high_feature: torch.Tensor,
        mid_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project raw backbone features into the prototype matching space.

        This operation is shared between support and query, ensuring both are
        compared in the same embedding space.

        Args:
            high_feature: f5, shape ``[B, 2048, H/16, W/16]``.
            mid_feature: f4, shape ``[B, 1024, H/16, W/16]``.

        Returns:
            (projected_high, projected_mid), the projected features.
        """
        high = self.coord_conv_high(self._add_coordinate_channels(high_feature))
        mid = self.coord_conv_low(self._add_coordinate_channels(mid_feature))
        return high, mid

    def extract_prototypes(
        self,
        support_high: torch.Tensor,
        support_mid: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> List[PrototypeList]:
        """Extract a prototype for each foreground class from support features and mask.

        Extraction formula::

            prototype_c = sum( feature[p] * 1[mask[p] == c] ) / ( sum 1[mask[p] == c] + eps )

        i.e., feature averaging over all pixels covered by class c.

        Special handling:
            - When there are no foreground classes, a zero prototype is returned to
              prevent empty batches early in training from crashing the matching branch.
            - The mask is automatically resized to the feature map resolution
              (nearest-neighbour interpolation, preserving class IDs).

        Args:
            support_high: Projected support high-level features, ``[B, C_high, H, W]``.
            support_mid: Projected support mid-level features, ``[B, C_mid, H, W]``.
            support_mask: Support labels, ``[B, 1, H, W]`` or ``[B, H, W]``.

        Returns:
            Outer list length = batch_size.
            Inner list length = number of foreground classes in that sample
            (1 when there are none, containing a zero prototype).
            Each element is a ``(high_proto_vec, mid_proto_vec)`` tuple.
        """
        # ---- Unify mask shape to [B, 1, H, W] ----
        if support_mask.dim() == 3:
            support_mask = support_mask.unsqueeze(1)

        batch_size, high_channels, height, width = support_high.shape
        mid_channels = support_mid.shape[1]

        # Resize the mask to the same spatial resolution as the feature maps.
        resized_mask = F.interpolate(
            support_mask.float(), size=(height, width), mode="nearest"
        )

        batch_prototypes: List[PrototypeList] = []
        for batch_index in range(batch_size):
            # Find all class IDs in the current sample's mask (excluding background=0).
            class_ids = torch.unique(resized_mask[batch_index])
            class_ids = class_ids[class_ids != 0]
            sample_prototypes: PrototypeList = []

            if class_ids.numel() == 0:
                # No foreground — insert a zero-prototype placeholder to keep
                # the batch dimension aligned.
                zero_high = torch.zeros(
                    high_channels, device=support_high.device, dtype=support_high.dtype
                )
                zero_mid = torch.zeros(
                    mid_channels, device=support_mid.device, dtype=support_mid.dtype
                )
                sample_prototypes.append((zero_high, zero_mid))
            else:
                for class_id in class_ids:
                    # Binary mask for the current class.
                    class_mask = (resized_mask[batch_index] == class_id).float()
                    pixel_count = class_mask.sum()
                    if pixel_count < 1:
                        continue

                    # Masked average pooling -> prototype vector.
                    high_proto = (
                        support_high[batch_index] * class_mask
                    ).sum(dim=(1, 2)) / (pixel_count + 1e-6)
                    mid_proto = (
                        support_mid[batch_index] * class_mask
                    ).sum(dim=(1, 2)) / (pixel_count + 1e-6)
                    sample_prototypes.append((high_proto, mid_proto))

            batch_prototypes.append(sample_prototypes)

        return batch_prototypes

    @staticmethod
    def _flatten_prototypes(prototypes) -> PrototypeList:
        """Recursively flatten various prototype container formats into a single list.

        Supported input formats:
            - A single (high, mid) tuple
            - A list of [(high, mid), ...]
            - A nested list [[(high, mid), ...], ...] (e.g., a memory bank)

        This flexibility allows match_prototypes to work with both training-time
        batch prototypes and inference-time memory banks.
        """
        flat: PrototypeList = []
        if isinstance(prototypes, tuple):
            flat.append(prototypes)
        elif isinstance(prototypes, list):
            for item in prototypes:
                if isinstance(item, tuple):
                    flat.append(item)
                elif isinstance(item, list):
                    flat.extend(
                        CoordinatePrototypeInteraction._flatten_prototypes(item)
                    )
        return flat

    def match_prototypes(
        self,
        prototypes,                     # Flexible prototype container
        query_high: torch.Tensor,       # Raw query high-level features (before projection)
        query_mid: torch.Tensor,        # Raw query mid-level features (before projection)
    ) -> torch.Tensor:
        """Match query features against a prototype set via cosine similarity to produce a similarity map.

        Matching process (per batch sample):
            1. Project query features into the same embedding space as the prototypes via project_features.
            2. L2-normalise the query and each prototype.
            3. Compute per-pixel inner product (equivalent to cosine similarity), scaled by the temperature coefficient.
            4. Take the per-position maximum across all prototypes -> a single response map.
            5. Compute separately for high and mid levels, then concatenate into ``[B, 2, H, W]``.

        Smart batch adaptation:
            - If the 0-th level of ``prototypes`` is a batch dimension (length == batch_size
              and each element is a list), match is performed per sample.
            - Otherwise all samples share the same prototype set (e.g., a memory bank at inference time).

        Args:
            prototypes: Prototype data in a flexible format (see _flatten_prototypes).
            query_high: Query f5 features, ``[B, 2048, H/16, W/16]``.
            query_mid: Query f4 features, ``[B, 1024, H/16, W/16]``.

        Returns:
            Two-channel similarity map ``[B, 2, H, W]``:
            channel 0 = high-level similarity,
            channel 1 = mid-level similarity.
            Mid-level is upsampled to match the high-level spatial resolution if needed.
        """
        # Project query features into the embedding space.
        query_high, query_mid = self.project_features(query_high, query_mid)
        batch_size = query_high.shape[0]

        # Determine whether each sample has its own prototypes.
        is_batch_prototypes = (
            isinstance(prototypes, list)
            and len(prototypes) == batch_size
            and batch_size > 0
            and isinstance(prototypes[0], list)
        )

        high_maps: List[torch.Tensor] = []
        mid_maps: List[torch.Tensor] = []

        for batch_index in range(batch_size):
            # Retrieve the prototype set for the current sample.
            current_prototypes = (
                prototypes[batch_index] if is_batch_prototypes else prototypes
            )
            flat_prototypes = self._flatten_prototypes(current_prototypes)

            # L2-normalise query features (per-channel independent normalisation).
            q_high = F.normalize(query_high[batch_index : batch_index + 1], p=2, dim=1)
            q_mid = F.normalize(query_mid[batch_index : batch_index + 1], p=2, dim=1)

            if not flat_prototypes:
                # No prototypes -> output an all-zero similarity map.
                high_maps.append(torch.zeros_like(q_high[:, :1]))
                mid_maps.append(torch.zeros_like(q_mid[:, :1]))
                continue

            # Compute cosine similarity per prototype.
            high_scores: List[torch.Tensor] = []
            mid_scores: List[torch.Tensor] = []
            for high_proto, mid_proto in flat_prototypes:
                # Reshape the prototype vector into a 1x1 convolution kernel.
                high_proto = high_proto.view(1, -1, 1, 1)
                mid_proto = mid_proto.view(1, -1, 1, 1)

                # L2-normalise prototypes.
                high_proto = F.normalize(high_proto, p=2, dim=1)
                mid_proto = F.normalize(mid_proto, p=2, dim=1)

                # Per-pixel inner product = cosine similarity, scaled by temperature.
                high_scores.append(
                    (q_high * high_proto).sum(dim=1, keepdim=True) * self.temperature
                )
                mid_scores.append(
                    (q_mid * mid_proto).sum(dim=1, keepdim=True) * self.temperature
                )

            # Max across all prototypes -> best matching response.
            high_map = torch.stack(high_scores, dim=1).max(dim=1).values
            mid_map = torch.stack(mid_scores, dim=1).max(dim=1).values
            high_maps.append(high_map)
            mid_maps.append(mid_map)

        # ---- Reassemble batch ----
        high_sim = torch.cat(high_maps, dim=0)
        mid_sim = torch.cat(mid_maps, dim=0)

        # If the two levels have different spatial resolutions (mid is typically
        # smaller), upsample mid to match high's size.
        if mid_sim.shape[-2:] != high_sim.shape[-2:]:
            mid_sim = F.interpolate(
                mid_sim,
                size=high_sim.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        return torch.cat([high_sim, mid_sim], dim=1)

    def forward(
        self,
        support_high: torch.Tensor,
        support_mid: torch.Tensor,
        support_mask: torch.Tensor,
        query_high: torch.Tensor,
        query_mid: torch.Tensor,
    ) -> torch.Tensor:
        """One-shot support-to-query prototype matching for training.

        Equivalent to ``extract_prototypes + match_prototypes`` chained together.

        Args:
            support_high: Support f5 features.
            support_mid: Support f4 features.
            support_mask: Support labels.
            query_high: Query f5 features.
            query_mid: Query f4 features.

        Returns:
            Two-channel similarity map ``[B, 2, H, W]``.
        """
        # Project support features and extract prototypes.
        support_high, support_mid = self.project_features(support_high, support_mid)
        prototypes = self.extract_prototypes(support_high, support_mid, support_mask)
        # Match.
        return self.match_prototypes(prototypes, query_high, query_mid)


# =====================================================================
# 3. TRG — Temporal Reliability Gating
# =====================================================================

class TemporalReliabilityGating(nn.Module):
    """Temporal Reliability Gating module (TRG).

    Problem background
    ------------------
    The predicted mask from the previous frame can serve as a strong temporal
    prior for the current frame — scene changes between adjacent frames are
    usually small. However, this prior can fail in certain situations:
    - **Motion blur**: rapid motion blurs image regions, making previous-frame
      boundaries unreliable;
    - **Occlusion**: object movement exposes previously occluded regions;
    - **Previous-frame prediction errors**: errors propagate across time (error accumulation).

    Solution
    --------
    TRG learns a **gate map** whose inputs are the current-frame high-level
    features concatenated with the previous-frame mask, and whose output is a
    per-pixel weight in [0, 1] representing "is the previous frame's mask
    trustworthy at this position?".

    Operation::

        gate = Sigmoid( Conv( [current_feature, previous_mask] ) )
        gated_previous = previous_mask * gate

    The current features have discriminative power regarding "whether this
    position agrees with the previous mask":
    - Clear texture consistent with the previous-frame class -> gate ~ 1 (retain)
    - Blurry texture or disagreement with previous class -> gate ~ 0 (suppress)

    Notes
    -----
    - The layer name ``reliability_conv`` is preserved for full compatibility with legacy checkpoints.
    - The gate itself can also be used for visualisation and debugging (as an intermediate feature output).
    """

    def __init__(self, feature_channels: int) -> None:
        """
        Args:
            feature_channels: Number of channels in the current-frame high-level features (typically 2048).
        """
        super().__init__()
        self.reliability_conv = nn.Sequential(
            # Input: feature_channels + 1 (previous mask is single-channel)
            nn.Conv2d(feature_channels + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: single-channel gate map, squashed to [0, 1] via Sigmoid
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        current_feature: torch.Tensor,
        previous_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the gated previous-frame mask.

        Args:
            current_feature: Current-frame high-level features, ``[B, C, H, W]``.
            previous_mask: Previous-frame predicted mask, ``[B, 1, H, W]`` or ``[B, H, W]``.

        Returns:
            (gated_mask, gate_map) tuple.
            - gated_mask: ``[B, 1, H, W]`` gated previous-frame mask.
            - gate_map: ``[B, 1, H, W]`` raw gate map (0~1), useful for visualisation.
        """
        # ---- Unify shape ----
        if previous_mask.dim() == 3:
            previous_mask = previous_mask.unsqueeze(1)

        # If the previous mask resolution differs from the feature map,
        # align it using nearest-neighbour interpolation.
        if previous_mask.shape[-2:] != current_feature.shape[-2:]:
            previous_mask = F.interpolate(
                previous_mask.float(),
                size=current_feature.shape[-2:],
                mode="nearest",
            )

        # ---- Gate inference ----
        gate_input = torch.cat([current_feature, previous_mask.float()], dim=1)
        gate = self.reliability_conv(gate_input)  # [B, 1, H, W], range [0, 1]

        return previous_mask.float() * gate, gate


# =====================================================================
# 4. PG-ASPP — Prior-Guided Multi-scale Context Fusion
# =====================================================================

class PriorGuidedASPP(nn.Module):
    """Prior-Guided ASPP module (PG-ASPP).

    Architecture overview
    ---------------------
    The input is a concatenation of two parts:
    - **High-level features** (2048 channels): backbone f5
    - **Priors** (3 channels):
        1. C-PIM high-level similarity map
        2. C-PIM mid-level similarity map
        3. TRG-gated previous-frame mask

    Processing flow::

        High-level features ──▶ 1x1 reduction ──▶ ASPP multi-branch ──▶ fusion ──┐
                                  (dilated conv + global pooling)                 │
                                                                                  ├──▶ final features
        Priors ──▶ prior_conv ──▶ spatial Gate ──▶ attention weighting ──────────┘
               │
               └──▶ prior_project ──▶ residual connection ───────────────────────┘

    Key design choices
    ------------------
    1. **Residual attention**: gate is in [0, 1] and used for enhancement rather
       than suppression. ``guided = aspp_feature * (1 + gate)``.
       When gate ~ 0, the original ASPP feature is preserved; when gate ~ 1,
       regions with reliable priors are enhanced.
    2. **Residual connection**: the prior is projected via 1x1 convolution and
       added directly to the attention-weighted ASPP feature, allowing prior
       information to be transmitted without relying on the attention gate.
    3. **Global pooling branch**: uses GroupNorm instead of BatchNorm (BatchNorm
       behaviour is unstable during batch=1 inference).

    Notes
    -----
    - The ``feat_dim`` field name is retained from the original codebase; it
      points to the number of high-level feature channels (= in_channels - 3).
    - Layer names (``reduce_conv``, ``aspp_branches``, etc.) are kept for compatibility with legacy checkpoints.
    """

    def __init__(
        self,
        in_channels: int,                      # Total input channels = high-level feature channels + 3
        out_channels: int = 256,               # ASPP output channels
        dilation_rates: Sequence[int] = (6, 12, 18),  # Dilation rates for the dilated convolutions
    ) -> None:
        super().__init__()
        # High-level feature channels = total inputs - 3 (prior channels)
        self.feat_dim = in_channels - 3
        if self.feat_dim <= 0:
            raise ValueError(
                "PriorGuidedASPP requires at least 3 prior channels plus feature channels."
            )

        # ---- ASPP part ----

        # 1x1 reduction: project high-dimensional features to a unified dimension.
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.feat_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ASPP multi-scale branches:
        # - Branch 0: 1x1 convolution (original scale)
        # - Branches 1..N: 3x3 dilated convolution with dilation = 6, 12, 18 (different receptive fields)
        branches: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in dilation_rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,       # Key to preserving output spatial size
                        dilation=rate,      # Larger dilation -> larger receptive field
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.aspp_branches = nn.ModuleList(branches)

        # Global average pooling branch (captures whole-image context).
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),     # Global pooling to 1x1
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            # BatchNorm is unstable at 1x1 spatial size -> use GroupNorm
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

        # Multi-branch fusion: concatenate all branch outputs, then 1x1 compress back to out_channels.
        branch_count = len(dilation_rates) + 2   # +1 original scale, +1 global pooling
        self.aspp_fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * branch_count, out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                    # Regularisation, prevents overfitting
        )

        # ---- Prior-guided part ----

        # Map the 3-channel prior to a spatial attention gate.
        self.prior_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),    # Single-channel gate
            nn.Sigmoid(),                        # Squash to [0, 1]
        )

        # Project the 3-channel prior into an out_channels-dimensional residual signal.
        self.prior_project = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final fusion: refine after residual attention + residual connection.
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Concatenated input ``[B, C+3, H, W]``; the first C channels are
               high-level features, the last 3 are priors.

        Returns:
            ``[B, out_channels, H, W]`` fused feature map.
        """
        # ---- Split input ----
        high_feature = x[:, :-3]    # High-level backbone features
        prior = x[:, -3:]           # 3-channel priors

        # ---- ASPP multi-scale processing ----
        reduced = self.reduce_conv(high_feature)  # Reduce to out_channels

        branch_outputs = [branch(reduced) for branch in self.aspp_branches]

        # Global pooling branch
        global_feature = self.global_avg_pool(reduced)
        global_feature = F.interpolate(
            global_feature,
            size=reduced.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        branch_outputs.append(global_feature)

        # Concatenate all branches and fuse.
        aspp_feature = self.aspp_fusion(torch.cat(branch_outputs, dim=1))

        # ---- Prior guidance ----
        gate = self.prior_conv(prior)              # Spatial attention gate
        prior_feature = self.prior_project(prior)  # Projected prior

        # Residual attention: gate is in [0, 1]; multiplying by (1 + gate)
        # ensures the original feature is at minimum preserved.
        guided_feature = aspp_feature * (1.0 + gate)

        # Final fusion: attention-weighted + residual prior connection.
        return self.final_fusion(guided_feature + prior_feature)


# =====================================================================
# 5. FewShotSegmenter — End-to-End Model Wrapper
# =====================================================================

class FewShotSegmenter(nn.Module):
    """ProTeCt end-to-end few-shot semantic segmentation model.

    Assembles all sub-modules into a complete forward pass.

    Sub-modules::

        self.encoder      = ResNetEncoder       (shared feature extraction)
        self.pim          = C-PIM               (prototype matching)
        self.trg          = TRG                 (temporal gating)
        self.aspp         = PG-ASPP             (prior-guided fusion)
        self.low_level_conv                      (low-level feature projection)
        self.decoder_conv                       (final decoder)

    forward (training)::

        support_features = encoder(support_image)
        query_features   = encoder(query_image)
        similarity       = pim(support_f4/5, mask, query_f4/5)
        gated_prev, _    = trg(query_f5, previous_mask)
        high_feature     = aspp(concat(query_f5, similarity, gated_prev))
        logits           = decoder(high_feature + low_level_f2)

    predict_with_memory (inference)::

        query_features   = encoder(query_image)          # or use pre-extracted features
        similarity       = pim.match(memory_bank, f4/5)  # horizontal-flip TTA
        gated_prev, _    = trg(f5, previous_pred_mask)
        high_feature     = aspp(concat(f5, similarity, gated_prev))
        logits           = decoder(high_feature + low_level_f2)

    Compatibility
    -------------
    All sub-module layer names are consistent with the original experiment code;
    legacy weights can be loaded directly via ``load_state_dict``.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        """
        Args:
            config: Model configuration. If None, the default SegmentationConfig() is used.
        """
        super().__init__()
        self.config = config or SegmentationConfig()

        # ---- 1. Encoder ----
        self.encoder = ResNetEncoder(self.config)
        channels = self.encoder.out_channels  # [64, 256, 512, 1024, 2048]

        # ---- 2. C-PIM: prototype interaction ----
        # high_level = f5 (2048-d), mid_level = f4 (1024-d)
        self.pim = CoordinatePrototypeInteraction(
            high_level_channels=channels[4],    # 2048
            mid_level_channels=channels[3],     # 1024
        )

        # ---- 3. TRG: temporal reliability gating ----
        self.trg = TemporalReliabilityGating(feature_channels=channels[4])

        # ---- 4. PG-ASPP: prior-guided ASPP ----
        # Input = f5(2048) + similarity(2) + gated_prev(1) = 2048+3
        self.aspp = PriorGuidedASPP(
            in_channels=channels[4] + 3,        # 2048 + 3
            out_channels=256,
        )

        # ---- 5. Low-level feature projection ----
        # Project f2 (stride=4, 256-d) to 48-d for decoder fusion of fine boundaries.
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(channels[1], 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ---- 6. Decoder ----
        # Input: ASPP output (256-d) + low-level features (48-d) = 304-d
        # Output: logits with num_classes channels
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 1x1 output layer, produces per-class logits
            nn.Conv2d(256, self.config.num_classes, kernel_size=1),
        )

        # ---- 7. Projection head (extensible to contrastive loss during training) ----
        # The current training script returns this feature but does not add an
        # extra loss on it; retained for future extensions.
        self.proj_head = nn.Conv2d(channels[4], 128, kernel_size=1)

    # ==================================================================
    # Convenience API: for use by the inference engine
    # ==================================================================

    def extract_features(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Extract a five-level feature pyramid for a single frame.

        Can be called independently during inference to avoid re-instantiating
        the encoder within the memory bank pipeline.
        """
        return self.encoder(image)

    def extract_prototypes_from_features(
        self,
        features: List[torch.Tensor],
        mask: torch.Tensor,
    ) -> List[PrototypeList]:
        """Extract prototypes from already-encoded features and a mask.

        Args:
            features: Five-level feature list from the encoder.
            mask: Label mask.

        Returns:
            List of prototypes for each batch sample.
        """
        high, mid = self.pim.project_features(features[4], features[3])
        return self.pim.extract_prototypes(high, mid, mask)

    def _decode_from_high_feature(
        self,
        high_feature: torch.Tensor,    # ASPP output [B, 256, H/16, W/16]
        low_feature: torch.Tensor,     # encoder f2 [B, 256, H/4, W/4]
    ) -> torch.Tensor:
        """Fuse high-level semantic features with low-level detail features and output logits.

        Steps:
            1. Upsample high-level features to the same resolution as low-level features.
            2. Project low-level features to 48 channels via a 1x1 convolution.
            3. Concatenate and pass through the decoder convolution to produce logits.
        """
        # Upsample to f2 resolution.
        high_feature = F.interpolate(
            high_feature,
            size=low_feature.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        low_feature = self.low_level_conv(low_feature)
        fused = torch.cat([high_feature, low_feature], dim=1)
        return self.decoder_conv(fused)

    def predict_with_memory(
        self,
        query_features: List[torch.Tensor],
        previous_mask: torch.Tensor,
        memory_bank,                      # List of historical prototypes
        use_feature_flip: bool = True,    # Whether to use horizontal-flip TTA
        return_features: bool = False,    # Whether to return intermediate features (for visualisation)
    ):
        """Predict the current frame using historical prototypes from the memory bank.

        This is the main entry point for inference. Unlike forward(), it does not
        depend on a support frame; instead it uses prototypes accumulated in the
        memory bank.

        TTA (Test-Time Augmentation)
        ----------------------------
        If ``use_feature_flip=True``, horizontal-flip TTA is applied at the
        feature level:
        - Similarity maps are computed for both the original features and their
          horizontally flipped versions.
        - The flipped result is flipped back and averaged with the original.
        This is more effective than rotation or scaling in validation because
        flipping involves no interpolation.

        Args:
            query_features: Encoder output for the current frame (5-level feature list).
            previous_mask: Predicted mask from the previous frame.
            memory_bank: Accumulated prototype list (from the support frame and confident past frames).
            use_feature_flip: Whether to enable horizontal-flip TTA.
            return_features: Whether to also return intermediate feature maps (for visualisation / debugging).

        Returns:
            - If return_features=False: logits ``[B, num_classes, H, W]``.
            - If return_features=True: (logits, features_dict),
              where features_dict contains ``query_high``, ``similarity``, ``refined_prior``, ``gate_map``.
        """
        query_high = query_features[4]   # f5, stride=16
        query_mid = query_features[3]    # f4, stride=16
        low_feature = query_features[1]  # f2, stride=4 (used for decoder fusion)

        # ---- C-PIM matching ----
        similarity = self.pim.match_prototypes(memory_bank, query_high, query_mid)

        # ---- Horizontal-flip TTA ----
        if use_feature_flip:
            flipped_similarity = self.pim.match_prototypes(
                memory_bank,
                torch.flip(query_high, dims=[3]),  # Horizontal flip of high-level features
                torch.flip(query_mid, dims=[3]),   # Horizontal flip of mid-level features
            )
            # Flip back and average.
            similarity = 0.5 * (
                similarity + torch.flip(flipped_similarity, dims=[3])
            )

        # ---- TRG temporal gating ----
        gated_previous, gate_map = self.trg(query_high, previous_mask)

        # ---- PG-ASPP fusion ----
        prior_input = torch.cat([query_high, similarity, gated_previous], dim=1)
        high_feature = self.aspp(prior_input)

        # ---- Decoding ----
        logits = self._decode_from_high_feature(high_feature, low_feature)

        if return_features:
            return logits, {
                "query_high": query_high,
                "similarity": similarity,
                "refined_prior": gated_previous,
                "gate_map": gate_map,
            }
        return logits

    # ==================================================================
    # Standard forward (training)
    # ==================================================================

    def forward(
        self,
        support_image: torch.Tensor,      # First-frame image [B, C, H, W]
        support_mask: torch.Tensor,        # First-frame labels [B, 1, H, W]
        query_image: torch.Tensor,         # Current-frame image [B, C, H, W]
        previous_mask: torch.Tensor,       # Previous-frame mask [B, 1, H, W] (perturbed GT during training)
    ):
        """One-shot support-query forward pass for training.

        Training pipeline:
            1. Encode support and query images.
            2. C-PIM: extract prototypes from support and match against query features.
            3. TRG: gate the previous-frame mask.
            4. PG-ASPP: fuse query features + similarity map + gated prior.
            5. Decoder: produce final logits and upsample to the original image resolution.

        Args:
            support_image: Support-frame image.
            support_mask: Support labels.
            query_image: Query-frame image.
            previous_mask: Previous-frame mask (perturbed support_mask during training).

        Returns:
            Training mode: (logits, support_projection) tuple.
            Inference mode: logits only.
        """
        # ---- 1. Encoding ----
        support_features = self.encoder(support_image)
        query_features = self.encoder(query_image)

        # ---- 2. C-PIM prototype matching ----
        similarity = self.pim(
            support_features[4],   # support f5
            support_features[3],   # support f4
            support_mask,
            query_features[4],     # query f5
            query_features[3],     # query f4
        )

        # ---- 3. TRG temporal gating ----
        gated_previous, _ = self.trg(query_features[4], previous_mask)

        # ---- 4. PG-ASPP ----
        prior_input = torch.cat(
            [query_features[4], similarity, gated_previous], dim=1
        )
        high_feature = self.aspp(prior_input)

        # ---- 5. Decoding ----
        logits = self._decode_from_high_feature(high_feature, query_features[1])

        # ---- 6. Upsample to original image resolution ----
        logits = F.interpolate(
            logits,
            size=query_image.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        # During training, return projected features for potential contrastive loss extension.
        if self.training:
            support_projection = self.proj_head(support_features[4])
            return logits, support_projection

        return logits


# =====================================================================
# Utility function — Load Checkpoint
# =====================================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Dict:
    """Load a model checkpoint, compatible with two save formats.

    Supported formats:
        1. Full training state dict: ``{"model_state_dict": ..., "optimizer_state_dict": ..., ...}``
        2. Raw model weight dict: ``OrderedDict/state_dict``

    Loads with ``strict=False``, allowing the checkpoint to have missing or
    extra keys (e.g., a legacy model without proj_head loaded into a new model
    that has one — this will not error).

    Args:
        model: Target model instance.
        checkpoint_path: Path to the .pth file.
        device: Target device.

    Returns:
        A dictionary containing at least the ``model_state_dict`` key.
        For training state, additional fields such as optimizer state are also present.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both raw state_dict and full training state formats.
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)

    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model_state_dict": state_dict}
