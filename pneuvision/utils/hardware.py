"""
Hardware verification, deterministic seeding, and parameter/MACs profiling utilities
for the `rama` L40S GPU server.
"""

import os
import platform
import random
import shutil
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_deterministic_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inspect_environment(check_path: str = ".") -> Dict[str, Any]:
    """
    Collect hardware, CUDA, BF16, VRAM, and disk metrics on `rama` (or local machine).
    """
    info: Dict[str, Any] = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": None,
        "vram_total_gib": 0.0,
        "vram_free_gib": 0.0,
        "bf16_supported": False,
        "disk_free_gib": 0.0,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        info["gpu_name"] = props.name
        info["vram_total_gib"] = round(total_bytes / (1024 ** 3), 2)
        info["vram_free_gib"] = round(free_bytes / (1024 ** 3), 2)
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())

    os.makedirs(check_path, exist_ok=True)
    disk_usage = shutil.disk_usage(check_path)
    info["disk_free_gib"] = round(disk_usage.free / (1024 ** 3), 2)
    return info


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Return exact parameter count for a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def profile_model_efficiency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 1, 256, 256),
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Measure total trainable parameters, model size in MB (FP32 & BF16), and exact
    Multiply-Accumulate operations (MACs) / FLOPs for a single forward pass using
    lightweight forward hooks (zero external library dependencies).
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    macs_total = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal macs_total
        batch_size = out.shape[0]
        out_c, out_h, out_w = out.shape[1], out.shape[2], out.shape[3]
        k_h, k_w = module.kernel_size
        in_c = module.in_channels
        groups = module.groups
        macs_per_elem = (in_c // groups) * k_h * k_w
        macs = batch_size * out_c * out_h * out_w * macs_per_elem
        if module.bias is not None:
            macs += batch_size * out_c * out_h * out_w
        macs_total += int(macs)

    def linear_hook(module: nn.Linear, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal macs_total
        num_elements = out.numel()
        macs = num_elements * module.in_features
        macs_total += int(macs)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    batch = input_shape[0]
    macs_per_image = macs_total // max(batch, 1)
    flops_per_image = 2 * macs_per_image

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "params_k": round(trainable_params / 1e3, 2),
        "params_m": round(trainable_params / 1e6, 4),
        "size_fp32_mb": round((trainable_params * 4) / (1024 ** 2), 3),
        "size_bf16_mb": round((trainable_params * 2) / (1024 ** 2), 3),
        "macs_per_image": macs_per_image,
        "macs_m": round(macs_per_image / 1e6, 2),
        "flops_m": round(flops_per_image / 1e6, 2),
    }
