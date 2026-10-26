"""
Ultra-lightweight depthwise-separable residual classifier (`TinyPneuNet`) with
configurable front-end `TaskGuidedEnhancer` and channel attention (`none`, `se`, `eca`, `cbam`).

Used across:
- Stage 1 (Exp 1): Isolating the effect of Fixed Image Enhancement (Inverted-U curve) vs.
  Differentiable Task-Guided Adaptive Enhancement (`TGAE`) under a tight parameter budget.
- Stage 3 (Exp 3): Attention & Parameter-Budget Pareto scaling.
"""

from typing import Tuple

import torch
import torch.nn as nn

from pneuvision.data.enhancement import TaskGuidedEnhancer
from pneuvision.models.attention import build_attention_block


class DepthwiseSeparableResBlock(nn.Module):
    """
    Lightweight Inverted/Depthwise-Separable Residual Block with optional Channel Attention.
    Uses 3x3 depthwise conv + 1x1 pointwise conv + BatchNorm + SiLU + Attention + Shortcut.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        attention_mode: str = "none",
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU(inplace=True)

        self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU(inplace=True)

        self.att = build_attention_block(attention_mode, out_ch, reduction=se_reduction)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.dw_conv(x)))
        out = self.act2(self.bn2(self.pw_conv(out)))
        out = self.att(out)
        return out + self.shortcut(x)


class TinyPneuNet(nn.Module):
    """
    Compact Pediatric Chest X-Ray Classifier integrating:
    - Front-End `TaskGuidedEnhancer` (0 params for fixed modes, 2 params for global,
      58 params for content-adaptive per-image enhancer).
    - 4 progressive depthwise-separable residual stages downsampling 256 -> 16.
    - Dual Global Pooling (Concatenated Global Average + Global Max Pooling to capture both
      diffuse interstitial opacity and focal lobar consolidation).
    """

    def __init__(
        self,
        in_channels: int = 1,
        enhancement_mode: str = "raw",
        stage_channels: Tuple[int, int, int, int] = (16, 24, 36, 48),
        attention_mode: str = "none",
        se_reduction: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.enhancement_mode = enhancement_mode
        self.enhancer = TaskGuidedEnhancer(mode=enhancement_mode)

        c1, c2, c3, c4 = stage_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        # 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        self.stage1 = DepthwiseSeparableResBlock(c1, c1, stride=2, attention_mode=attention_mode, se_reduction=se_reduction)
        self.stage2 = DepthwiseSeparableResBlock(c1, c2, stride=2, attention_mode=attention_mode, se_reduction=se_reduction)
        self.stage3 = DepthwiseSeparableResBlock(c2, c3, stride=2, attention_mode=attention_mode, se_reduction=se_reduction)
        self.stage4 = DepthwiseSeparableResBlock(c3, c4, stride=2, attention_mode=attention_mode, se_reduction=se_reduction)

        self.dropout = nn.Dropout(p=dropout)
        # Dual pooling (Avg + Max) -> 2 * c4 features -> 1 binary logit
        self.head = nn.Linear(2 * c4, 1)

    def forward_with_extras(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(logits, x_enhanced, enhancement_reg_loss)`."""
        x_enh, reg_loss = self.enhancer(x)
        feat = self.stem(x_enh)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)

        avg_pool = torch.mean(feat, dim=(2, 3))
        max_pool = torch.amax(feat, dim=(2, 3))
        pooled = self.dropout(torch.cat([avg_pool, max_pool], dim=1))
        logits = self.head(pooled)
        return logits, x_enh, reg_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning `(B, 1)` classification logits."""
        logits, _, _ = self.forward_with_extras(x)
        return logits
