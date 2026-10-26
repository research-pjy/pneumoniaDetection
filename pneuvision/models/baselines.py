"""
Stage 3 (Exp 3): Standard Vision Baselines (`ResNet-18`, `MobileNetV2`, `ShuffleNetV2_x0.5`,
`SqueezeNet1_1`) adapted for 1-channel Pediatric Chest Radiographs, alongside the four
parameter-budget tiers of our unified `TG-SECAE` architecture (`Nano`, `Micro`, `Tiny`, `Small`).
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torchvision.models as tvm

from pneuvision.models.autoencoder import UnifiedTGSECAE


TGSECAE_TIERS: Dict[str, Dict[str, Any]] = {
    "nano": {
        "base_channels": 8,
        "latent_channels": 10,
        "downsample_factor": 8,
        "cls_stage_channels": (14, 20, 28),
    },
    "micro": {
        "base_channels": 12,
        "latent_channels": 16,
        "downsample_factor": 8,
        "cls_stage_channels": (24, 36, 48),
    },
    "tiny": {
        "base_channels": 20,
        "latent_channels": 28,
        "downsample_factor": 8,
        "cls_stage_channels": (44, 68, 96),
    },
    "small": {
        "base_channels": 32,
        "latent_channels": 48,
        "downsample_factor": 8,
        "cls_stage_channels": (80, 128, 192),
    },
}


def build_tgsecae_tier(
    tier: str = "micro",
    attention_mode: str = "se",
    se_reduction: int = 4,
    enhancement_mode: str = "task_guided_adaptive",
    dropout: float = 0.2,
) -> UnifiedTGSECAE:
    """Instantiate one of the 4 parameter-budget tiers of `UnifiedTGSECAE`."""
    if tier not in TGSECAE_TIERS:
        raise ValueError(f"Unknown tier '{tier}'. Choose from {list(TGSECAE_TIERS.keys())}.")
    cfg = TGSECAE_TIERS[tier]
    return UnifiedTGSECAE(
        in_channels=1,
        enhancement_mode=enhancement_mode,
        base_channels=cfg["base_channels"],
        latent_channels=cfg["latent_channels"],
        downsample_factor=cfg["downsample_factor"],
        cls_stage_channels=cfg["cls_stage_channels"],
        attention_mode=attention_mode,
        se_reduction=se_reduction,
        dropout=dropout,
    )


class TorchvisionGrayscaleBaseline(nn.Module):
    """
    Wraps standard `torchvision` vision backbones (`resnet18`, `mobilenet_v2`,
    `shufflenet_v2_x0_5`, `squeezenet1_1`) trained from scratch on 1-channel `(B, 1, 256, 256)`
    pediatric chest radiographs for fair, controlled Pareto-frontier benchmarking.
    """

    def __init__(self, arch: str) -> None:
        super().__init__()
        self.arch = arch

        if arch == "resnet18":
            net = tvm.resnet18(weights=None)
            net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            net.fc = nn.Linear(net.fc.in_features, 1)
            self.net = net
        elif arch == "mobilenet_v2":
            net = tvm.mobilenet_v2(weights=None)
            first_conv = net.features[0][0]
            net.features[0][0] = nn.Conv2d(
                1,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
            in_f = net.classifier[1].in_features
            net.classifier[1] = nn.Linear(in_f, 1)
            self.net = net
        elif arch == "shufflenet_v2_x0_5":
            net = tvm.shufflenet_v2_x0_5(weights=None)
            first_conv = net.conv1[0]
            net.conv1[0] = nn.Conv2d(
                1,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
            net.fc = nn.Linear(net.fc.in_features, 1)
            self.net = net
        elif arch == "squeezenet1_1":
            net = tvm.squeezenet1_1(weights=None)
            first_conv = net.features[0]
            net.features[0] = nn.Conv2d(
                1,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
            )
            net.classifier[1] = nn.Conv2d(512, 1, kernel_size=1)
            self.net = net
        else:
            raise ValueError(f"Unsupported baseline architecture '{arch}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if out.ndim == 1:
            return out.unsqueeze(1)
        return out.view(x.shape[0], 1)
