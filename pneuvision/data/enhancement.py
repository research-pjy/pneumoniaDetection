"""
Radiograph enhancement algorithms and the differentiable Task-Guided Adaptive Enhancer (TGAE).

Supports:
1. Deterministic Tile-Interpolated CLAHE (pure NumPy, zero OpenCV dependency required)
2. Progressive Enhancement Strength dial `alpha` (for the Stage 1 Inverted-U Curve study)
3. Multi-View 3-Channel Stack `[Raw, CLAHE, HighPass]`
4. Differentiable `TaskGuidedEnhancer` (`nn.Module`) driven by downstream classification loss:
   - `task_guided_global`: 2 learnable scalar parameters (contrast gain & detail gain)
   - `task_guided_adaptive`: Per-image content-adaptive gating network (< 60 parameters)
     that predicts optimal enhancement strengths from radiograph statistics.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_clahe_numpy(
    img_01: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
    nbins: int = 256,
) -> np.ndarray:
    """
    Pure-NumPy Contrast Limited Adaptive Histogram Equalization (CLAHE) with
    bilinear interpolation across tile centers.

    Args:
        img_01: 2D float32 array in [0.0, 1.0] of shape (H, W).
        clip_limit: Relative contrast clip limit (e.g., 2.0). If <= 0, returns img_01.
        grid_size: Number of tiles along each spatial axis (e.g., 8x8).
        nbins: Histogram bins (256).

    Returns:
        CLAHE-enhanced 2D float32 array in [0.0, 1.0] of shape (H, W).
    """
    if clip_limit <= 0.0:
        return img_01.copy()

    h, w = img_01.shape
    img_q = np.clip(np.round(img_01 * (nbins - 1)), 0, nbins - 1).astype(np.int32)

    th = max(h // grid_size, 1)
    tw = max(w // grid_size, 1)
    cdfs = np.zeros((grid_size, grid_size, nbins), dtype=np.float32)

    for r in range(grid_size):
        y0 = r * th
        y1 = h if r == grid_size - 1 else (r + 1) * th
        for c in range(grid_size):
            x0 = c * tw
            x1 = w if c == grid_size - 1 else (c + 1) * tw
            tile = img_q[y0:y1, x0:x1]
            hist = np.bincount(tile.ravel(), minlength=nbins).astype(np.float32)

            n_pixels = float(tile.size)
            limit = max(clip_limit * (n_pixels / float(nbins)), 1.0)
            excess = np.maximum(hist - limit, 0.0).sum()
            hist = np.minimum(hist, limit) + (excess / float(nbins))

            cdf = np.cumsum(hist)
            cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0.0
            denom = max(n_pixels - cdf_min, 1e-6)
            cdfs[r, c] = np.clip((cdf - cdf_min) / denom, 0.0, 1.0)

    # Bilinear interpolation across tile centers
    ys = (np.arange(h, dtype=np.float32) + 0.5) / float(th) - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) / float(tw) - 0.5

    r0 = np.clip(np.floor(ys).astype(np.int32), 0, grid_size - 1)
    r1 = np.clip(r0 + 1, 0, grid_size - 1)
    wy = np.clip(ys - r0, 0.0, 1.0)[:, None]

    c0 = np.clip(np.floor(xs).astype(np.int32), 0, grid_size - 1)
    c1 = np.clip(c0 + 1, 0, grid_size - 1)
    wx = np.clip(xs - c0, 0.0, 1.0)[None, :]

    r0_2d = np.broadcast_to(r0[:, None], (h, w))
    r1_2d = np.broadcast_to(r1[:, None], (h, w))
    c0_2d = np.broadcast_to(c0[None, :], (h, w))
    c1_2d = np.broadcast_to(c1[None, :], (h, w))

    v00 = cdfs[r0_2d, c0_2d, img_q]
    v01 = cdfs[r0_2d, c1_2d, img_q]
    v10 = cdfs[r1_2d, c0_2d, img_q]
    v11 = cdfs[r1_2d, c1_2d, img_q]

    top = (1.0 - wx) * v00 + wx * v01
    bot = (1.0 - wx) * v10 + wx * v11
    out = (1.0 - wy) * top + wy * bot
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_highpass_numpy(img_01: np.ndarray) -> np.ndarray:
    """
    Compute local high-frequency parenchymal detail map using a 5x5 box-blur residual,
    shifted and scaled to [0.0, 1.0].
    """
    t = torch.from_numpy(img_01)[None, None, :, :]
    smooth = F.avg_pool2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    detail = (t - smooth).squeeze().numpy()
    detail_norm = np.clip(0.5 + 2.5 * detail, 0.0, 1.0)
    return detail_norm.astype(np.float32)


def apply_progressive_enhancement_numpy(
    img_01: np.ndarray,
    alpha: float = 0.5,
    base_clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Progressive enhancement operator controlled by a single scalar strength `alpha >= 0`:
    - alpha = 0.0: Raw radiograph
    - 0 < alpha <= 1.0: Linear blend between Raw and standard CLAHE(clip_limit=2.0)
    - alpha > 1.0: Pushes CLAHE clip_limit higher (2.0 * alpha) AND adds unsharp high-frequency
      amplification, modeling the over-enhancement regime where vascular markings & sensor
      noise get amplified.
    """
    if alpha <= 0.0:
        return img_01.copy()

    effective_clip = base_clip_limit * max(alpha, 1.0)
    clahe_img = apply_clahe_numpy(img_01, clip_limit=effective_clip, grid_size=grid_size)

    if alpha <= 1.0:
        return np.clip((1.0 - alpha) * img_01 + alpha * clahe_img, 0.0, 1.0).astype(np.float32)

    # Over-enhancement regime (alpha > 1.0): add high-frequency unsharp amplification
    t = torch.from_numpy(clahe_img)[None, None, :, :]
    smooth = F.avg_pool2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
    high_freq = (t - smooth).squeeze().numpy()
    over_gain = alpha - 1.0
    enhanced = clahe_img + 1.25 * over_gain * high_freq
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def build_multiview_3ch_numpy(
    img_01: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Build a 3-channel enhanced radiograph tensor `(3, H, W)`:
      - Channel 0: Raw Grayscale X-Ray
      - Channel 1: CLAHE Local Contrast Enhanced X-Ray
      - Channel 2: High-Pass Parenchymal Texture / Edge Map
    """
    ch0 = img_01
    ch1 = apply_clahe_numpy(img_01, clip_limit=clip_limit, grid_size=grid_size)
    ch2 = compute_highpass_numpy(img_01)
    return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)


class TaskGuidedEnhancer(nn.Module):
    """
    Differentiable Task-Guided Adaptive Enhancer (TGAE) — implements Idea 4 (Bus Intuition).

    Decomposes an input batch of radiographs `X in [0, 1]` into:
    1. Multi-scale Local Contrast Normalized component `C(X)` (soft differentiable CLAHE proxy
       using 15x15 local mean & standard deviation normalization).
    2. High-Frequency Parenchymal Detail component `D(X)` (5x5 residual detail).

    Combines them via classification-guided enhancement weights:
        X_hat = clip( (1 - w_c) * X + w_c * C(X) + w_d * D(X), 0, 1 )

    Modes:
    - `raw` / `fixed_clahe` / `multiview_3ch`: Identity pass-through (0 parameters).
    - `task_guided_global`: Learns global scalar logits for `(w_c, w_d)` (2 parameters total).
    - `task_guided_adaptive`: Uses a tiny 2-layer MLP gate `g_phi(stats(X))` operating on
      4 per-image exposure statistics `[mean, std, low_energy, high_energy]` to predict
      image-specific enhancement weights `(w_c(X), w_d(X))` (only 58 parameters total!).
    """

    def __init__(
        self,
        mode: str = "task_guided_adaptive",
        gate_hidden_dim: int = 8,
        init_contrast_weight: float = 0.45,
        init_detail_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.mode = mode

        # Inverse sigmoid helper for clean initialization
        init_c_logit = float(np.log(init_contrast_weight / (1.0 - init_contrast_weight)))
        init_d_logit = float(np.log(init_detail_weight / (1.0 - init_detail_weight)))

        if mode == "task_guided_global":
            self.contrast_logit = nn.Parameter(torch.tensor(init_c_logit, dtype=torch.float32))
            self.detail_logit = nn.Parameter(torch.tensor(init_d_logit, dtype=torch.float32))
        elif mode == "task_guided_adaptive":
            # 4 summary statistics -> gate_hidden_dim -> 2 enhancement gains (contrast, detail)
            self.gate_net = nn.Sequential(
                nn.Linear(4, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 2),
            )
            with torch.no_grad():
                nn.init.normal_(self.gate_net[0].weight, std=0.05)
                nn.init.zeros_(self.gate_net[0].bias)
                nn.init.normal_(self.gate_net[2].weight, std=0.05)
                self.gate_net[2].bias.copy_(torch.tensor([init_c_logit, init_d_logit]))

    @staticmethod
    def _compute_components(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute differentiable local contrast map C(x) and high-frequency detail D(x)."""
        # 1. Local contrast normalization over a 15x15 window (reflect-padded)
        pad_c = 7
        x_pad = F.pad(x, (pad_c, pad_c, pad_c, pad_c), mode="reflect")
        local_mean = F.avg_pool2d(x_pad, kernel_size=15, stride=1)
        local_sq = F.avg_pool2d(x_pad * x_pad, kernel_size=15, stride=1)
        local_std = torch.sqrt(torch.clamp(local_sq - local_mean * local_mean, min=1e-5))

        # Sigmoid-squashed standardized local contrast in [0, 1]
        z_local = (x - local_mean) / (local_std + 0.08)
        contrast_map = torch.sigmoid(1.5 * z_local)

        # 2. Fine parenchymal high-pass detail over a 5x5 window
        x_pad5 = F.pad(x, (2, 2, 2, 2), mode="reflect")
        smooth5 = F.avg_pool2d(x_pad5, kernel_size=5, stride=1)
        detail_map = torch.tanh(3.0 * (x - smooth5))

        return contrast_map, detail_map

    def get_enhancement_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(w_contrast, w_detail)` of shape `(B, 1, 1, 1)` for a batch `x`."""
        b = x.shape[0]
        if self.mode == "task_guided_global":
            wc = torch.sigmoid(self.contrast_logit).view(1, 1, 1, 1).expand(b, 1, 1, 1)
            wd = (0.5 * torch.sigmoid(self.detail_logit)).view(1, 1, 1, 1).expand(b, 1, 1, 1)
            return wc, wd
        elif self.mode == "task_guided_adaptive":
            # Compute 4 differentiable exposure statistics per image:
            # [global_mean, global_std, dark_lung_fraction, bright_bone_fraction]
            flat = x.view(b, -1)
            mean = flat.mean(dim=1, keepdim=True)
            std = flat.std(dim=1, keepdim=True, unbiased=False)
            dark_frac = torch.sigmoid(10.0 * (0.35 - flat)).mean(dim=1, keepdim=True)
            bright_frac = torch.sigmoid(10.0 * (flat - 0.75)).mean(dim=1, keepdim=True)
            stats = torch.cat([mean, std, dark_frac, bright_frac], dim=1)

            logits = self.gate_net(stats)
            wc = torch.sigmoid(logits[:, 0]).view(b, 1, 1, 1)
            wd = (0.5 * torch.sigmoid(logits[:, 1])).view(b, 1, 1, 1)
            return wc, wd
        else:
            zeros = torch.zeros((b, 1, 1, 1), device=x.device, dtype=x.dtype)
            return zeros, zeros

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhance radiograph batch `x` of shape `(B, C, H, W)`.

        Returns:
            `(x_enhanced, reg_loss)` where `reg_loss` is a mild over-enhancement penalty
            that prevents runaway noise amplification unless offset by classification gain.
        """
        if self.mode not in ("task_guided_global", "task_guided_adaptive"):
            return x, torch.tensor(0.0, device=x.device, dtype=x.dtype)

        contrast_map, detail_map = self._compute_components(x)
        wc, wd = self.get_enhancement_weights(x)

        x_enh = (1.0 - wc) * x + wc * contrast_map + wd * detail_map
        x_enh = torch.clamp(x_enh, 0.0, 1.0)

        # Mild quadratic penalty on excessive high-frequency detail gain (prevents noise blowup)
        reg_loss = torch.mean(wd ** 2 + 0.25 * torch.relu(wc - 0.75) ** 2)
        return x_enh, reg_loss
