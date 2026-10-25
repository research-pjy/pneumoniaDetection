"""
Publication-grade Matplotlib visualization utilities for PneuVision.

Generates:
1. Stage 0: Radiograph Enhancement Progression Panel (`stage0_enhancement_preview.png`)
   comparing Raw -> Optimal CLAHE -> Over-Enhanced regimes on NORMAL vs. PNEUMONIA.
2. Stage 1: The Inverted-U Enhancement Strength vs. Classification Accuracy/F1/AUROC curve.
3. Stage 3: Parameter & MACs Pareto Frontier plot against heavyweight & lightweight baselines.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pneuvision.data.enhancement import (
    apply_progressive_enhancement_numpy,
    compute_highpass_numpy,
)


def plot_enhancement_progression_panel(
    normal_img: np.ndarray,
    pneumonia_img: np.ndarray,
    out_path: str | Path,
    alphas: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0),
) -> Path:
    """
    Generate a 2-row publication figure comparing a NORMAL (row 0) and PNEUMONIA (row 1)
    chest radiograph across progressive enhancement intensities `alpha`, plus the
    High-Pass Parenchymal Detail map.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_cols = len(alphas) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.4), dpi=180)

    rows = [
        ("NORMAL (Class 0)", normal_img),
        ("PNEUMONIA (Class 1)", pneumonia_img),
    ]

    for r_idx, (row_label, base_img) in enumerate(rows):
        for c_idx, alpha in enumerate(alphas):
            ax = axes[r_idx, c_idx]
            enh = apply_progressive_enhancement_numpy(base_img, alpha=alpha)
            ax.imshow(enh, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])

            if r_idx == 0:
                if alpha == 0.0:
                    tag = "Raw (α = 0.0)"
                elif alpha <= 1.0:
                    tag = f"Sweet Spot (α = {alpha:.1f})"
                else:
                    tag = f"Over-Enh (α = {alpha:.1f})"
                ax.set_title(tag, fontsize=10, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")

        # Last column: High-Pass Parenchymal Detail channel
        ax_hp = axes[r_idx, -1]
        hp = compute_highpass_numpy(base_img)
        ax_hp.imshow(hp, cmap="gray", vmin=0.0, vmax=1.0)
        ax_hp.set_xticks([])
        ax_hp.set_yticks([])
        if r_idx == 0:
            ax_hp.set_title("High-Pass Detail", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Radiograph Enhancement Progression: From Raw to Optimal Contrast vs. Over-Enhancement Noise",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_inverted_u_enhancement_curve(
    sweep_records: List[Dict[str, Any]],
    tgae_record: Dict[str, Any],
    out_path: str | Path,
) -> Path:
    """
    Plot Validation F1-Score, AUROC, and Specificity across fixed enhancement strengths
    `alpha` alongside the horizontal benchmark line achieved by the Learnable
    Task-Guided Adaptive Enhancer (TGAE).
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    alphas = [float(r["alpha"]) for r in sweep_records]
    f1s = [float(r["f1_score"]) for r in sweep_records]
    aurocs = [float(r["auroc"]) for r in sweep_records]
    specs = [float(r["specificity"]) for r in sweep_records]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    ax.plot(alphas, f1s, marker="o", linewidth=2.2, label="Fixed Enhancement — Val F1 (%)", color="#1f77b4")
    ax.plot(alphas, aurocs, marker="s", linewidth=2.0, linestyle="--", label="Fixed Enhancement — Val AUROC (%)", color="#2ca02c")
    ax.plot(alphas, specs, marker="^", linewidth=1.8, linestyle=":", label="Fixed Enhancement — Specificity (%)", color="#d62728")

    if tgae_record:
        tgae_f1 = float(tgae_record.get("f1_score", 0.0))
        ax.axhline(
            tgae_f1,
            color="#9467bd",
            linewidth=2.3,
            linestyle="-.",
            label=f"Task-Guided Adaptive Enhancer (TGAE) F1 = {tgae_f1:.2f}%",
        )

    best_idx = int(np.argmax(f1s))
    ax.scatter([alphas[best_idx]], [f1s[best_idx]], s=110, color="#ff7f0e", zorder=5, label=f"Peak Fixed α* = {alphas[best_idx]}")

    ax.set_xlabel("Enhancement Strength Dial (α)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Metric (%)", fontsize=11, fontweight="bold")
    ax.set_title("The Over-Enhancement Inverted-U Curve vs. Classification-Guided Enhancement", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
