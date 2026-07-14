"""
Computational Complexity Benchmark
==================================

Independently measures parameter count, FLOPs, and inference latency of each
ProTeCt module.

Methodology
-----------
1. Uses ``thop.profile`` to measure parameter count (Params) and floating-point
   operations (FLOPs).
2. Uses ``torch.cuda.Event`` for high-precision CUDA latency measurement (ms),
   including warmup iterations and multi-run averaging.

Measurement Granularity
-----------------------
Each module is independently wrapped and measured to obtain accurate incremental costs:

- **Baseline**: Pure ResNet50 encoder + decoder (no C-PIM/TRG).
- **C-PIM Module**: Coordinate Prototype Interaction (projection + prototype extraction + matching).
- **TRG Module**: Temporal Reliability Gating.
- **Memory Bank**: Prototype extraction from predicted masks (incremental cost at inference).

Dependencies
------------
``thop`` library for FLOPs/Params statistics::

    pip install thop

Output
------
A Markdown-formatted table showing absolute and incremental metrics per component.

Usage::

    python tools/benchmark_complexity.py
"""

from __future__ import annotations

import sys
import gc
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protect_segmentation import FewShotSegmenter, SegmentationConfig
from protect_segmentation.model import (
    ResNetEncoder,
    CoordinatePrototypeInteraction,
    TemporalReliabilityGating,
    PrototypeList,
)


# =====================================================================
# Test Configuration
# =====================================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (540, 960)
WARMUP_ITERS = 50
MEASURE_ITERS = 200
NUM_RUNS = 10

torch.backends.cudnn.benchmark = True


# =====================================================================
# Lightweight wrappers for per-module measurement
# =====================================================================

class MemoryBankUpdateWrapper(nn.Module):
    """Wraps the prototype extraction operation to measure memory-bank update
    overhead."""
    def __init__(self, pim_module: CoordinatePrototypeInteraction):
        super().__init__()
        self.pim = pim_module

    def forward(self, q4: torch.Tensor, q3: torch.Tensor, mask: torch.Tensor):
        h, l = self.pim.project_features(q4, q3)
        return self.pim.extract_prototypes(h, l, mask)


class CPIM_Wrapper(nn.Module):
    """Wraps the full C-PIM forward pass: projection + prototype extraction +
    matching."""
    def __init__(self, pim_module: CoordinatePrototypeInteraction):
        super().__init__()
        self.pim = pim_module

    def forward(self, s4, s3, s_mask, q4, q3):
        return self.pim(s4, s3, s_mask, q4, q3)


class TRG_Wrapper(nn.Module):
    """Wraps the TRG gating process."""
    def __init__(self, trg_module: TemporalReliabilityGating):
        super().__init__()
        self.trg = trg_module

    def forward(self, curr_feat, prev_mask):
        return self.trg(curr_feat, prev_mask)


class BaselineDecoder(nn.Module):
    """Encoder + decoder only (no C-PIM/TRG), used to measure the Baseline.

    Shares the same encoder + decoder architecture as FewShotSegmenter,
    but replaces C-PIM with an identity mapping (returns an all-zero
    similarity map) and substitutes zero tensors for the TRG gating
    output.
    """

    def __init__(self, config: SegmentationConfig):
        super().__init__()
        self.config = config
        self.encoder = ResNetEncoder(config)
        channels = self.encoder.out_channels

        self.aspp = nn.Sequential(
            nn.Conv2d(channels[4] + 3, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(channels[1], 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, config.num_classes, kernel_size=1),
        )

    def forward(self, support_image, support_mask, query_image, previous_mask):
        s_feats = self.encoder(support_image)
        q_feats = self.encoder(query_image)

        # Baseline: all-zero similarity + all-zero gating prior
        similarity = torch.zeros(
            q_feats[4].shape[0], 2, q_feats[4].shape[2], q_feats[4].shape[3],
            device=q_feats[4].device,
        )
        gated_previous = torch.zeros_like(previous_mask)
        if gated_previous.shape[-2:] != q_feats[4].shape[-2:]:
            gated_previous = F.interpolate(
                gated_previous, size=q_feats[4].shape[-2:], mode="nearest",
            )

        prior_input = torch.cat([q_feats[4], similarity, gated_previous], dim=1)
        high_feature = self.aspp(prior_input)
        high_feature = F.interpolate(
            high_feature, size=q_feats[1].shape[-2:], mode="bilinear", align_corners=True,
        )
        low_feature = self.low_level_conv(q_feats[1])
        fused = torch.cat([high_feature, low_feature], dim=1)
        logits = self.decoder_conv(fused)
        return F.interpolate(
            logits, size=query_image.shape[-2:], mode="bilinear", align_corners=True,
        )


# =====================================================================
# High-Precision Latency Measurement
# =====================================================================

def measure_latency(module: nn.Module, *inputs: torch.Tensor) -> float:
    """High-precision latency measurement using CUDA Events (ms).
    Outliers are trimmed before taking the mean."""
    module.eval()
    latencies: List[float] = []

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = module(*inputs)
        torch.cuda.synchronize()

        for _ in range(NUM_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(MEASURE_ITERS):
                _ = module(*inputs)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end) / MEASURE_ITERS)

    if len(latencies) >= 6:
        latencies = sorted(latencies)[2:-2]
    return float(np.mean(latencies))


def profile_module(module: nn.Module, name: str, *inputs: torch.Tensor):
    """Measure parameter count / FLOPs / latency."""
    from thop import profile as thop_profile

    module.eval()
    with torch.no_grad():
        flops, params = thop_profile(module, inputs=inputs, verbose=False)
        latency = measure_latency(module, *inputs)

    print(f"[Profile] {name} — Latency: {latency:.4f} ms, "
          f"Params: {params / 1e6:.2f}M, FLOPs: {flops / 1e9:.2f}G")
    return params / 1e6, flops / 1e9, latency


# =====================================================================
# Main Function
# =====================================================================

def main() -> None:
    print(f"{'=' * 60}")
    print(f"ProTeCt Computational Complexity Benchmark")
    print(f"Device: {DEVICE}")
    print(f"Input size: {INPUT_SIZE[0]}x{INPUT_SIZE[1]}")
    print(f"{'=' * 60}\n")

    config = SegmentationConfig(
        input_height=INPUT_SIZE[0],
        input_width=INPUT_SIZE[1],
        pretrained=False,
    )

    # ---- Dummy inputs ----
    s_img = torch.randn(1, 3, *INPUT_SIZE).to(DEVICE)
    s_mask = torch.randint(0, config.num_classes, (1, 1, *INPUT_SIZE)).float().to(DEVICE)
    q_img = torch.randn(1, 3, *INPUT_SIZE).to(DEVICE)
    prev_mask = torch.randint(0, config.num_classes, (1, 1, *INPUT_SIZE)).float().to(DEVICE)

    # ---- Run the full model once to extract intermediate features for per-module benchmarking ----
    print("Extracting intermediate features for per-module benchmarking...")
    model_full = FewShotSegmenter(config).to(DEVICE)
    model_full.eval()

    with torch.no_grad():
        s_feats = model_full.extract_features(s_img)
        q_feats = model_full.extract_features(q_img)
        s4, s3 = s_feats[4].detach(), s_feats[3].detach()
        q4, q3 = q_feats[4].detach(), q_feats[3].detach()
        pred_mask_dn = torch.ones((1, 1, q4.shape[2], q4.shape[3])).to(DEVICE)

    # ==================================================================
    # 1. Baseline (encoder + decoder, zero priors)
    # ==================================================================
    model_base = BaselineDecoder(config).to(DEVICE)
    base_p, base_f, base_l = profile_module(
        model_base, "Baseline network",
        s_img, s_mask, q_img, prev_mask,
    )

    # ==================================================================
    # 2-4. Independent per-module measurement
    # ==================================================================
    cpim_w = CPIM_Wrapper(model_full.pim).to(DEVICE)
    trg_w = TRG_Wrapper(model_full.trg).to(DEVICE)
    mem_w = MemoryBankUpdateWrapper(model_full.pim).to(DEVICE)

    cpim_p, cpim_f, cpim_l = profile_module(
        cpim_w, "C-PIM module", s4, s3, s_mask, q4, q3,
    )
    trg_p, trg_f, trg_l = profile_module(
        trg_w, "TRG module", q4, prev_mask,
    )
    mem_p, mem_f, mem_l = profile_module(
        mem_w, "Memory Bank update", q4, q3, pred_mask_dn,
    )

    del model_full, model_base
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # Results Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("              ProTeCt Computational Complexity Analysis Summary")
    print("=" * 60 + "\n")

    total_lat = base_l + cpim_l + trg_l + mem_l

    print(f"| Component | Params (M) | Delta-P | FLOPs (G) | Delta-F | Latency (ms) | Latency % |")
    print(f"|-----------|-----------|--------|----------|---------|-------------|-----------|")
    print(f"| Baseline | {base_p:.2f} | — | {base_f:.2f} | — | {base_l:.2f} | {base_l / total_lat * 100:.1f}% |")
    print(f"| + C-PIM | {base_p + cpim_p:.2f} | +{cpim_p:.2f} | {base_f + cpim_f:.2f} | +{cpim_f:.2f} | +{cpim_l:.2f} | +{cpim_l / total_lat * 100:.1f}% |")
    print(f"| + TRG | {base_p + cpim_p + trg_p:.2f} | +{trg_p:.2f} | {base_f + cpim_f + trg_f:.2f} | +{trg_f:.2f} | +{trg_l:.2f} | +{trg_l / total_lat * 100:.1f}% |")
    print(f"| + Memory Bank | {base_p + cpim_p + trg_p + mem_p:.2f} | +{mem_p:.2f} | {base_f + cpim_f + trg_f + mem_f:.2f} | +{mem_f:.2f} | +{mem_l:.2f} | +{mem_l / total_lat * 100:.1f}% |")
    print(f"| **ProTeCt (Full)** | **{base_p + cpim_p + trg_p + mem_p:.2f}** | **+{cpim_p + trg_p + mem_p:.2f}** | **{base_f + cpim_f + trg_f + mem_f:.2f}** | **+{cpim_f + trg_f + mem_f:.2f}** | **{total_lat:.2f}** | **100.0%** |")

    print()
    print("Note: Memory Bank does not add model parameters; it only introduces additional"
          " matching computation at inference time.")


if __name__ == "__main__":
    main()
