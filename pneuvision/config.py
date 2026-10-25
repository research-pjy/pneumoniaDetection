"""
Configuration dataclasses for PneuVision.

Strictly separates scientific / architectural parameters from hardware execution
settings for the L40S GPU server (`rama`), per `l40s_execution_context.txt`.
"""

from dataclasses import dataclass
from typing import Literal, Tuple


EnhancementMode = Literal[
    "raw",                  # 1-ch raw normalized radiograph [0, 1]
    "fixed_clahe",          # 1-ch blended CLAHE at fixed strength alpha
    "multiview_3ch",        # 3-ch stack: [Raw, CLAHE, HighPass Parenchymal Edge]
    "task_guided_global",   # Learnable global enhancement weights driven by classification loss
    "task_guided_adaptive", # Per-image content-adaptive gating network g_phi(X) driven by classification loss
]

AttentionMode = Literal[
    "none",                 # Standard narrow residual block without channel attention
    "se",                   # Squeeze-and-Excitation (Hu et al., CVPR 2018)
    "eca",                  # Efficient Channel Attention (1D conv across channels, 3 params/block)
    "cbam",                 # Convolutional Block Attention Module (Channel + Spatial attention)
]

CouplingMode = Literal[
    "two_stage_frozen",     # Stage 1: train AE -> Stage 2: freeze Encoder E, train Classifier
    "two_stage_finetune",   # Stage 1: train AE -> Stage 2: fine-tune Encoder E + Classifier
    "enhance_ae_finetune",  # Stage 1: train AE to map Raw -> Enhanced -> Stage 2: fine-tune E + Classifier
    "joint_multitask",      # End-to-end multi-task: L = L_BCE + lambda_rec * L_rec (Decoder discarded at inference)
]


@dataclass
class HardwareConfig:
    """
    Hardware and execution settings for the L40S machine (`rama`).
    Kept strictly separate from the scientific methodology.
    """
    device: str = "cuda"
    use_bf16: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset paths, image resolution, and split settings."""
    dataset_dir: str = "./dataset"
    image_size: int = 256
    in_channels: int = 1
    include_unlabeled_in_ae: bool = False  # Transductive unsupervised AE pretraining on test_unlabeled/


@dataclass
class EnhancementConfig:
    """
    Configuration for Stage 1 Image Enhancement & Task-Guided Adaptive Enhancer (TGAE).
    """
    mode: EnhancementMode = "task_guided_adaptive"
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    alpha_strength: float = 0.5            # Blending factor for fixed_clahe sweep (0.0 = raw, 1.0 = full CLAHE, >1.0 = over-enhanced)
    alpha_sweep_values: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
    gate_hidden_dim: int = 8               # Hidden channels for per-image adaptive enhancer gate (< 100 params)
    noise_penalty_weight: float = 0.01     # Regularization preventing runaway over-enhancement


@dataclass
class AutoencoderConfig:
    """
    Configuration for Stage 2 2D Spatial Convolutional Autoencoder.
    Default compresses (1, 256, 256) -> (latent_channels, 32, 32) [64x area reduction].
    """
    spatial_downsample_factor: int = 8     # 4 -> 64x64, 8 -> 32x32, 16 -> 16x16
    latent_channels: int = 16              # C in (C, 32, 32) spatial bottleneck
    base_channels: int = 12                # Initial stem channels
    use_depthwise_separable: bool = True   # Keeps Encoder under ~15k parameters
    use_1d_bottleneck: bool = False        # Only enabled for the 1D-vs-2D ablation baseline
    bottleneck_1d_dim: int = 64


@dataclass
class ClassifierConfig:
    """
    Configuration for Stage 3 Channel-Recalibrated Latent Classifier.
    Operates on the compact (latent_channels, 32, 32) 2D feature grid.
    """
    tier: Literal["nano", "micro", "tiny", "small"] = "tiny"
    stage_channels: Tuple[int, int, int] = (24, 48, 64)
    attention_mode: AttentionMode = "se"
    se_reduction: int = 4
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Training hyperparameters for AE pretraining and Classifier training."""
    coupling_mode: CouplingMode = "joint_multitask"
    ae_epochs: int = 20
    cls_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    recon_loss_weight: float = 0.25        # lambda_rec for joint_multitask
    use_class_weights: bool = True         # Automatically weights BCE by N_neg / N_pos to handle pediatric class imbalance
