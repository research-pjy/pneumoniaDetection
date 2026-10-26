"""
Stage 2 (Idea 1): Lightweight 2D Spatial Convolutional Autoencoder, 1D Vector Autoencoder
(ablation baseline), and Latent-Space Pneumonia Classifier (`UnifiedTGSECAE`).

Key architectural design:
- `SpatialAutoencoder2D`: Compresses a `(1, 256, 256)` radiograph into a compact 2D feature
  grid `(C_latent, H_lat, W_lat)` (e.g., `16 x 32 x 32` for `downsample_factor=8`, a 64x
  reduction in spatial area) using depthwise-separable convolutions, and reconstructs back
  to `(1, 256, 256)` via bilinear upsampling + depthwise-separable refinement.
- At inference time, the Decoder `D` is discarded (`0` inference parameters and `0` MACs).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pneuvision.data.enhancement import TaskGuidedEnhancer
from pneuvision.models.classifier import DepthwiseSeparableResBlock


class UpSepConvBlock(nn.Module):
    """Bilinear 2x upsampling followed by depthwise-separable 3x3 refinement."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class SpatialEncoder2D(nn.Module):
    """
    Lightweight 2D Spatial Encoder `E: (B, C_in, 256, 256) -> (B, C_latent, H_lat, W_lat)`.
    Supports `downsample_factor` in {4 (64x64), 8 (32x32), 16 (16x16)}.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 12,
        latent_channels: int = 16,
        downsample_factor: int = 8,
        attention_mode: str = "none",
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        if downsample_factor not in (4, 8, 16):
            raise ValueError("downsample_factor must be 4 (64x64), 8 (32x32), or 16 (16x16).")

        self.downsample_factor = downsample_factor
        self.latent_channels = latent_channels

        # 256 -> 128
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        # 128 -> 64
        self.down2 = DepthwiseSeparableResBlock(
            base_channels, latent_channels, stride=2, attention_mode=attention_mode, se_reduction=se_reduction
        )

        blocks = []
        if downsample_factor >= 8:
            # 64 -> 32
            blocks.append(
                DepthwiseSeparableResBlock(
                    latent_channels, latent_channels, stride=2, attention_mode=attention_mode, se_reduction=se_reduction
                )
            )
        if downsample_factor == 16:
            # 32 -> 16
            blocks.append(
                DepthwiseSeparableResBlock(
                    latent_channels, latent_channels, stride=2, attention_mode=attention_mode, se_reduction=se_reduction
                )
            )
        self.extra_downs = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.down2(z)
        z = self.extra_downs(z)
        return z


class SpatialDecoder2D(nn.Module):
    """
    Lightweight 2D Spatial Decoder `D: (B, C_latent, H_lat, W_lat) -> (B, 1, 256, 256)`.
    Used only during Stage 1 AE pretraining or joint multi-task training; discarded at inference.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        base_channels: int = 12,
        out_channels: int = 1,
        downsample_factor: int = 8,
    ) -> None:
        super().__init__()
        up_blocks = []
        if downsample_factor == 16:
            # 16 -> 32
            up_blocks.append(UpSepConvBlock(latent_channels, latent_channels))
        if downsample_factor >= 8:
            # 32 -> 64
            up_blocks.append(UpSepConvBlock(latent_channels, latent_channels))

        # 64 -> 128
        up_blocks.append(UpSepConvBlock(latent_channels, base_channels))
        # 128 -> 256
        up_blocks.append(UpSepConvBlock(base_channels, base_channels))
        self.ups = nn.Sequential(*up_blocks)

        self.out_head = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out_head(self.ups(z))


class VectorAutoencoder1D(nn.Module):
    """
    1D Vector-Bottleneck Autoencoder (`(1, 256, 256) -> 16x16x8 -> Flatten(2048) -> Linear(64)`).
    Included strictly for Exp 2A to prove quantitatively why flattening chest X-rays to a 1D
    vector inflates parameters and destroys spatial lung geometry.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 12,
        vec_dim: int = 64,
    ) -> None:
        super().__init__()
        self.spatial_enc = SpatialEncoder2D(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=8,
            downsample_factor=16,  # (8, 16, 16) = 2048
        )
        self.to_vec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 16 * 16, vec_dim),
            nn.LayerNorm(vec_dim),
            nn.SiLU(inplace=True),
        )
        self.from_vec = nn.Sequential(
            nn.Linear(vec_dim, 8 * 16 * 16),
            nn.SiLU(inplace=True),
        )
        self.spatial_dec = SpatialDecoder2D(
            latent_channels=8,
            base_channels=base_channels,
            out_channels=1,
            downsample_factor=16,
        )
        self.cls_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(vec_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_vec(self.spatial_enc(x))

    def decode(self, z_vec: torch.Tensor) -> torch.Tensor:
        feat = self.from_vec(z_vec).view(-1, 8, 16, 16)
        return self.spatial_dec(feat)

    def forward_autoencoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_vec = self.encode(x)
        x_rec = self.decode(z_vec)
        return x_rec, z_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_vec = self.encode(x)
        return self.cls_head(z_vec)


class LatentPneuClassifier(nn.Module):
    """
    Downstream Pneumonia Classifier operating strictly on the compressed 2D Spatial Latent
    representation `z in (B, C_latent, H_lat, W_lat)` (e.g., `16 x 32 x 32`).
    """

    def __init__(
        self,
        latent_channels: int = 16,
        stage_channels: Tuple[int, int, int] = (24, 36, 48),
        attention_mode: str = "none",
        se_reduction: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2, c3 = stage_channels
        self.block1 = DepthwiseSeparableResBlock(
            latent_channels, c1, stride=2, attention_mode=attention_mode, se_reduction=se_reduction
        )
        self.block2 = DepthwiseSeparableResBlock(
            c1, c2, stride=2, attention_mode=attention_mode, se_reduction=se_reduction
        )
        self.block3 = DepthwiseSeparableResBlock(
            c2, c3, stride=1, attention_mode=attention_mode, se_reduction=se_reduction
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(2 * c3, 1)

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        f = self.block1(z)
        f = self.block2(f)
        f = self.block3(f)
        return f

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = self.forward_features(z)
        avg_p = torch.mean(f, dim=(2, 3))
        max_p = torch.amax(f, dim=(2, 3))
        pooled = self.dropout(torch.cat([avg_p, max_p], dim=1))
        return self.head(pooled)


class UnifiedTGSECAE(nn.Module):
    """
    Unified Publication Architecture combining all four research ideas:
      1. `enhancer`: Differentiable Task-Guided Adaptive Enhancer (`TGAE`, 58 params)
      2. `encoder`: 2D Spatial Convolutional Encoder (`E`, compressing 256x256 -> 32x32)
      3. `decoder`: 2D Spatial Decoder (`D`, used ONLY as a training-time reconstruction
         regularizer; excluded from `forward()` and discarded at inference time!)
      4. `classifier`: Channel-Recalibrated Latent Classifier (`SE` / `ECA` / `CBAM` / `none`)
    """

    def __init__(
        self,
        in_channels: int = 1,
        enhancement_mode: str = "task_guided_adaptive",
        base_channels: int = 12,
        latent_channels: int = 16,
        downsample_factor: int = 8,
        cls_stage_channels: Tuple[int, int, int] = (24, 36, 48),
        attention_mode: str = "none",
        se_reduction: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.enhancement_mode = enhancement_mode
        self.downsample_factor = downsample_factor
        self.latent_channels = latent_channels

        self.enhancer = TaskGuidedEnhancer(mode=enhancement_mode)
        self.encoder = SpatialEncoder2D(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            downsample_factor=downsample_factor,
            attention_mode=attention_mode,
            se_reduction=se_reduction,
        )
        self.decoder = SpatialDecoder2D(
            latent_channels=latent_channels,
            base_channels=base_channels,
            out_channels=1,
            downsample_factor=downsample_factor,
        )
        self.classifier = LatentPneuClassifier(
            latent_channels=latent_channels,
            stage_channels=cls_stage_channels,
            attention_mode=attention_mode,
            se_reduction=se_reduction,
            dropout=dropout,
        )

    def set_encoder_trainable(self, trainable: bool = True) -> None:
        """Freeze or unfreeze Encoder `E` (and `enhancer`) for Two-Stage Frozen vs. Fine-Tuned."""
        for p in self.encoder.parameters():
            p.requires_grad = trainable
        for p in self.enhancer.parameters():
            p.requires_grad = trainable

    def forward_autoencoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run `(enhancer -> encoder -> decoder)` and return `(x_rec, z_latent, enh_reg_loss)`."""
        x_enh, reg_loss = self.enhancer(x)
        z = self.encoder(x_enh)
        x_rec = self.decoder(z)
        return x_rec, z, reg_loss

    def forward_full(
        self,
        x: torch.Tensor,
        compute_reconstruction: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Full training forward pass returning logits, latent representation, enhanced image,
        optional reconstruction `x_rec`, and enhancement regularization loss.
        """
        x_enh, enh_reg_loss = self.enhancer(x)
        z = self.encoder(x_enh)
        logits = self.classifier(z)
        x_rec = self.decoder(z) if compute_reconstruction else None
        return {
            "logits": logits,
            "z_latent": z,
            "x_enhanced": x_enh,
            "x_reconstructed": x_rec,
            "enh_reg_loss": enh_reg_loss,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure inference forward pass (`enhancer -> encoder -> classifier`).
        Does NOT execute `self.decoder`, so inference MACs/FLOPs and latency only reflect
        the deployed classification path.
        """
        x_enh, _ = self.enhancer(x)
        z = self.encoder(x_enh)
        return self.classifier(z)

    def inference_parameter_breakdown(self) -> Dict[str, int]:
        """
        Return parameter counts broken down by module, distinguishing deployed inference
        parameters (`enhancer + encoder + classifier`) from the training-only `decoder`.
        """
        enh_p = sum(p.numel() for p in self.enhancer.parameters())
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        cls_p = sum(p.numel() for p in self.classifier.parameters())
        dec_p = sum(p.numel() for p in self.decoder.parameters())
        return {
            "enhancer_params": enh_p,
            "encoder_params": enc_p,
            "classifier_params": cls_p,
            "inference_total_params": enh_p + enc_p + cls_p,
            "decoder_training_only_params": dec_p,
        }
