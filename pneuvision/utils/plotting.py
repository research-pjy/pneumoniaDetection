"""
Publication-grade Matplotlib visualization utilities for PneuVision.

Generates:
1. Stage 0: Radiograph Enhancement Progression Panel (`stage0_enhancement_preview.png`)
   comparing Raw -> Optimal CLAHE -> Over-Enhanced regimes on NORMAL vs. PNEUMONIA.
2. Stage 1:
   - `stage1_inverted_u_curve.png`: The Inverted-U Enhancement Strength vs. Classification
     F1 / Balanced Accuracy / AUROC curve + Task-Guided Adaptive Enhancer (TGAE) benchmarks.
   - `stage1_tgae_visual_comparison.png`: Side-by-side visual comparison of Raw vs.
     Learned TGAE Enhancement vs. Signed Residual vs. Over-Enhanced (α=2.0) on real validation X-rays.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    tgae_adaptive_record: Optional[Dict[str, Any]],
    tgae_global_record: Optional[Dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """
    Plot Validation F1-Score, Balanced Accuracy, and AUROC across fixed enhancement strengths
    `alpha` alongside the horizontal benchmark lines achieved by the Learnable
    Task-Guided Adaptive Enhancer (`TGAE`).
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    alphas = [float(r["alpha"]) for r in sweep_records]
    f1s = [float(r["calibrated_metrics"]["f1_score"]) for r in sweep_records]
    bal_accs = [float(r["calibrated_metrics"]["balanced_accuracy"]) for r in sweep_records]
    aurocs = [float(r["calibrated_metrics"]["auroc"]) for r in sweep_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=180)

    # Panel A: F1-Score & AUROC vs Enhancement Dial alpha
    ax1.plot(alphas, f1s, marker="o", linewidth=2.3, label="Fixed CLAHE Sweep — Val F1 (%)", color="#1f77b4")
    ax1.plot(alphas, aurocs, marker="s", linewidth=2.0, linestyle="--", label="Fixed CLAHE Sweep — Val AUROC (%)", color="#2ca02c")

    best_idx = int(np.argmax(f1s))
    ax1.scatter(
        [alphas[best_idx]],
        [f1s[best_idx]],
        s=120,
        color="#ff7f0e",
        zorder=5,
        label=f"Peak Fixed α* = {alphas[best_idx]} (F1 = {f1s[best_idx]:.2f}%)",
    )

    if tgae_global_record is not None:
        g_f1 = float(tgae_global_record["calibrated_metrics"]["f1_score"])
        ax1.axhline(
            g_f1,
            color="#e377c2",
            linewidth=2.0,
            linestyle=":",
            label=f"TGAE-Global (+2 params) F1 = {g_f1:.2f}%",
        )

    if tgae_adaptive_record is not None:
        a_f1 = float(tgae_adaptive_record["calibrated_metrics"]["f1_score"])
        ax1.axhline(
            a_f1,
            color="#9467bd",
            linewidth=2.4,
            linestyle="-.",
            label=f"TGAE-Adaptive (+58 params) F1 = {a_f1:.2f}%",
        )

    ax1.set_xlabel("Enhancement Strength Dial (α)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Validation Metric (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) F1-Score & AUROC across Enhancement Strength α", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=8.5)

    # Panel B: Balanced Accuracy & Specificity (showing false-positive drop under over-enhancement)
    specs = [float(r["calibrated_metrics"]["specificity"]) for r in sweep_records]
    sens = [float(r["calibrated_metrics"]["sensitivity_recall"]) for r in sweep_records]
    ax2.plot(alphas, bal_accs, marker="D", linewidth=2.3, label="Fixed Sweep — Balanced Acc (%)", color="#d62728")
    ax2.plot(alphas, specs, marker="^", linewidth=2.0, linestyle="--", label="Fixed Sweep — Specificity / TNR (%)", color="#8c564b")
    ax2.plot(alphas, sens, marker="v", linewidth=1.8, linestyle=":", label="Fixed Sweep — Sensitivity / TPR (%)", color="#17becf")

    if tgae_adaptive_record is not None:
        a_bal = float(tgae_adaptive_record["calibrated_metrics"]["balanced_accuracy"])
        ax2.axhline(
            a_bal,
            color="#9467bd",
            linewidth=2.4,
            linestyle="-.",
            label=f"TGAE-Adaptive Balanced Acc = {a_bal:.2f}%",
        )

    ax2.set_xlabel("Enhancement Strength Dial (α)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Validation Metric (%)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Specificity & Balanced Accuracy vs. Over-Enhancement", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=8.5)

    fig.suptitle(
        "Stage 1 (Exp 1): Empirical Verification of the Over-Enhancement Trap & Task-Guided Enhancement",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_tgae_visual_comparison(
    normal_raw: np.ndarray,
    normal_tgae: np.ndarray,
    normal_weights: Tuple[float, float],
    pneu_raw: np.ndarray,
    pneu_tgae: np.ndarray,
    pneu_weights: Tuple[float, float],
    out_path: str | Path,
) -> Path:
    """
    Generate a 2x4 visual diagnostic figure comparing:
      Col 0: Raw Radiograph (α = 0.0)
      Col 1: Learned TGAE-Adaptive Radiograph (showing adaptive w_c, w_d)
      Col 2: TGAE Signed Enhancement Residual (X_tgae - X_raw)
      Col 3: Over-Enhanced Fixed CLAHE (α = 2.5)
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.8), dpi=180)
    rows = [
        ("NORMAL (Val)", normal_raw, normal_tgae, normal_weights),
        ("PNEUMONIA (Val)", pneu_raw, pneu_tgae, pneu_weights),
    ]

    for r_idx, (label, raw_img, tgae_img, (wc, wd)) in enumerate(rows):
        over_img = apply_progressive_enhancement_numpy(raw_img, alpha=2.5)
        diff_img = tgae_img - raw_img

        axes[r_idx, 0].imshow(raw_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r_idx, 0].set_ylabel(label, fontsize=10, fontweight="bold")
        if r_idx == 0:
            axes[r_idx, 0].set_title("Raw Input (α = 0.0)", fontsize=10, fontweight="bold")

        axes[r_idx, 1].imshow(tgae_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r_idx, 1].set_xlabel(f"Learned w_c={wc:.3f}, w_d={wd:.3f}", fontsize=9)
        if r_idx == 0:
            axes[r_idx, 1].set_title("TGAE-Adaptive (Ours)", fontsize=10, fontweight="bold")

        im_diff = axes[r_idx, 2].imshow(diff_img, cmap="coolwarm", vmin=-0.25, vmax=0.25)
        if r_idx == 0:
            axes[r_idx, 2].set_title("TGAE Residual (X_enh - X)", fontsize=10, fontweight="bold")
        plt.colorbar(im_diff, ax=axes[r_idx, 2], fraction=0.046, pad=0.04)

        axes[r_idx, 3].imshow(over_img, cmap="gray", vmin=0.0, vmax=1.0)
        if r_idx == 0:
            axes[r_idx, 3].set_title("Over-Enhanced (α = 2.5)", fontsize=10, fontweight="bold")

        for c_idx in range(4):
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])

    fig.suptitle(
        "Classification-Guided Radiograph Enhancement (TGAE) vs. Fixed Over-Enhancement",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
