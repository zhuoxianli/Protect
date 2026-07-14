"""
Model Verification Script
==========================

Verifies the correctness of the ProTeCt model's tensor flow using synthetic random
inputs. This is a lightweight sanity check that runs on CPU in seconds — no real
data or GPU required. Run before training or after modifying the model code.

Checks performed:
  1. All sub-modules are correctly connected (C-PIM, TRG, PG-ASPP, Decoder).
  2. Output tensor shapes match expected dimensions.
  3. No NaN or Inf values in outputs.
  4. Memory Bank inference interface (predict_with_memory) works correctly.

Usage::

    python scripts/verify_model.py

Expected output::

    Device: cpu
    Running forward pass...
    Model verification passed!
      Input size: 128x192
      Num classes: 11
      Training output shape: (1, 11, 128, 192)
    Memory Bank inference interface passed! (stride-4 output: (1, 11, 32, 48))
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protect_segmentation import FewShotSegmenter, SegmentationConfig


def main() -> None:
    """Run a complete model integrity check with synthetic inputs.

    Uses a small 128x192 input size to keep the test fast.  All sub-modules
    (C-PIM prototype extraction/matching, TRG gating, PG-ASPP multi-scale
    fusion, decoder upsampling) are exercised in a single forward pass.
    """
    # ---- Lightweight test configuration (CPU-friendly) ----
    config = SegmentationConfig(
        input_height=128,
        input_width=192,
        pretrained=False,   # Skip ImageNet weights for speed
        batch_size=1,
    )
    print(f"Device: {config.device}")

    model = FewShotSegmenter(config).eval()

    # ---- Build synthetic inputs ----
    # Support image (first frame with ground-truth label)
    support_image = torch.rand(1, 3, config.input_height, config.input_width)
    support_mask = torch.randint(
        0, config.num_classes,
        (1, 1, config.input_height, config.input_width),
    )

    # Query image (current frame to segment)
    query_image = torch.rand(1, 3, config.input_height, config.input_width)

    # Previous-frame mask (simulates the prediction from the last time step)
    previous_mask = torch.randint(
        0, config.num_classes,
        (1, 1, config.input_height, config.input_width),
    ).float()

    # ---- Training-mode forward pass ----
    print("Running forward pass...")
    with torch.no_grad():
        logits = model(support_image, support_mask, query_image, previous_mask)

    # ---- Validate output shape ----
    expected = (1, config.num_classes, config.input_height, config.input_width)
    assert tuple(logits.shape) == expected, (
        f"Incorrect output shape! Expected {expected}, got {tuple(logits.shape)}"
    )

    # ---- Validate numerical stability ----
    assert not torch.isnan(logits).any(), "Output contains NaN values!"
    assert not torch.isinf(logits).any(), "Output contains Inf values!"

    print("Model verification passed!")
    print(f"  Input size: {config.input_height}x{config.input_width}")
    print(f"  Num classes: {config.num_classes}")
    print(f"  Training output shape: {tuple(logits.shape)}")

    # ---- Extra check: Memory Bank inference interface ----
    print("\nTesting Memory Bank inference interface...")
    with torch.no_grad():
        features = model.extract_features(query_image)
        memory_bank = model.extract_prototypes_from_features(features, support_mask)
        logits2 = model.predict_with_memory(
            features, previous_mask, memory_bank, use_feature_flip=True
        )

    # predict_with_memory returns logits at stride-4 (decoder output resolution).
    # The inference engine (MemoryBankEngine.predict) handles upsampling to the
    # original image resolution.
    expected_mem = (
        1, config.num_classes,
        config.input_height // 4, config.input_width // 4,
    )
    assert tuple(logits2.shape) == expected_mem, (
        f"Memory Bank output shape incorrect! "
        f"Expected {expected_mem}, got {tuple(logits2.shape)}"
    )
    print(
        f"Memory Bank inference interface passed! "
        f"(stride-4 output: {tuple(logits2.shape)})"
    )


if __name__ == "__main__":
    main()
