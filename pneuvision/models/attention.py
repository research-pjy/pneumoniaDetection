"""
Ultra-lightweight Channel & Spatial Attention modules (Idea 2).

Implements:
1. `SEBlock`: Squeeze-and-Excitation (Hu et al., CVPR 2018) — Global Avg Pool -> FC(C -> C/r)
   -> ReLU -> FC(C/r -> C) -> Sigmoid channel recalibration.
2. `ECABlock`: Efficient Channel Attention (Wang et al., CVPR 2020) — replaces the 2 FC
   layers with a 1D convolution of kernel size 3 across channels (only 3 trainable parameters!).
3. `CBAMBlock`: Convolutional Block Attention Module (Woo et al., ECCV 2018) — combines
   Channel Attention with a 7x7 Spatial Attention gate over [ChannelAvg, ChannelMax].
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        if not self.training:
            self.last_weights = w.detach()
        return x * w


class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA) using a 3-parameter 1D convolution across channels."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.act = nn.Sigmoid()
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, 1, c)
        w = self.act(self.conv1d(y)).view(b, c, 1, 1)
        if not self.training:
            self.last_weights = w.detach()
        return x * w


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (SE Channel Attention + 7x7 Spatial Attention)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.channel_att = SEBlock(channels, reduction=reduction)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_w = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_w


def build_attention_block(mode: str, channels: int, reduction: int = 4) -> nn.Module:
    """Factory for channel/spatial attention blocks."""
    if mode == "none" or mode is None:
        return nn.Identity()
    if mode == "se":
        return SEBlock(channels, reduction=reduction)
    if mode == "eca":
        return ECABlock(channels, kernel_size=3)
    if mode == "cbam":
        return CBAMBlock(channels, reduction=reduction)
    raise ValueError(f"Unknown attention_mode '{mode}'. Choose from ['none', 'se', 'eca', 'cbam'].")
