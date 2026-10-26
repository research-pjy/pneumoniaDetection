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


def plot_stage2_ae_reconstructions(
    sample_raw: np.ndarray,
    rec_1d: np.ndarray,
    metrics_1d: Tuple[float, float],
    rec_16: np.ndarray,
    metrics_16: Tuple[float, float],
    rec_32: np.ndarray,
    metrics_32: Tuple[float, float],
    rec_64: np.ndarray,
    metrics_64: Tuple[float, float],
    latent_32_grid: np.ndarray,
    out_path: str | Path,
) -> Path:
    """
    Generate a 1x6 publication figure comparing Autoencoder Bottleneck Reconstructions:
      Col 0: Original Radiograph (256x256)
      Col 1: 1D Vector Bottleneck (z in R^64) -> blurry ribcage/parenchyma loss
      Col 2: 2D Spatial 16x16 Bottleneck
      Col 3: 2D Spatial 32x32 Bottleneck (Core 64x area compression)
      Col 4: 2D Spatial 64x64 Bottleneck
      Col 5: Mean 2D Latent Activation Map (32x32)
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 6, figsize=(15.5, 3.3), dpi=180)

    panels = [
        ("Original (256×256)", sample_raw, "Reference", "gray"),
        ("1D Vector (R^64)", rec_1d, f"PSNR: {metrics_1d[0]:.1f} dB | SSIM: {metrics_1d[1]:.3f}", "gray"),
        ("2D Grid (16×16×16)", rec_16, f"PSNR: {metrics_16[0]:.1f} dB | SSIM: {metrics_16[1]:.3f}", "gray"),
        ("2D Grid (16×32×32)", rec_32, f"PSNR: {metrics_32[0]:.1f} dB | SSIM: {metrics_32[1]:.3f}", "gray"),
        ("2D Grid (16×64×64)", rec_64, f"PSNR: {metrics_64[0]:.1f} dB | SSIM: {metrics_64[1]:.3f}", "gray"),
        ("32×32 Latent Map", latent_32_grid, "Mean 2D Bottleneck Energy", "viridis"),
    ]

    for idx, (title, img, sub, cmap) in enumerate(panels):
        ax = axes[idx]
        if cmap == "gray":
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel(sub, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Stage 2 (Exp 2A): 1D Vector Bottleneck vs. 2D Spatial Autoencoder Reconstruction Quality & Latent Map",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_stage2_coupling_comparison(
    bottleneck_records: List[Dict[str, Any]],
    coupling_records: List[Dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """
    Generate a 2-panel publication plot:
      Panel (a): Reconstruction PSNR & Classification F1 across Bottleneck Geometries (1D vs 2D).
      Panel (b): Validation F1, Balanced Accuracy, and AUROC across the 5 AE-Classifier Coupling Regimes.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), dpi=180)

    # Panel A: Bottleneck Geometry comparison
    b_names = [r["short_label"] for r in bottleneck_records]
    b_f1s = [float(r["calibrated_metrics"]["f1_score"]) for r in bottleneck_records]
    b_bal = [float(r["calibrated_metrics"]["balanced_accuracy"]) for r in bottleneck_records]
    b_psnr = [float(r["ae_psnr_db"]) for r in bottleneck_records]

    x_pos = np.arange(len(b_names))
    w = 0.34
    ax1.bar(x_pos - w / 2, b_f1s, width=w, label="Val F1-Score (%)", color="#1f77b4")
    ax1.bar(x_pos + w / 2, b_bal, width=w, label="Balanced Acc (%)", color="#2ca02c")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(b_names, fontsize=9, fontweight="bold")
    ax1.set_ylim(max(min(b_bal) - 4.0, 70.0), 100.0)
    ax1.set_ylabel("Classification Metric (%)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) 1D Vector vs. 2D Spatial Bottleneck Geometry", fontsize=10.5, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_pos, b_psnr, color="#d62728", marker="o", linewidth=2.2, label="Recon PSNR (dB)")
    ax1_twin.set_ylabel("Reconstruction PSNR (dB)", fontsize=10, fontweight="bold", color="#d62728")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8.5)

    # Panel B: Coupling Regimes comparison
    c_names = [r["short_label"] for r in coupling_records]
    c_f1s = [float(r["calibrated_metrics"]["f1_score"]) for r in coupling_records]
    c_bal = [float(r["calibrated_metrics"]["balanced_accuracy"]) for r in coupling_records]
    c_auc = [float(r["calibrated_metrics"]["auroc"]) for r in coupling_records]

    xc = np.arange(len(c_names))
    wc = 0.26
    ax2.bar(xc - wc, c_f1s, width=wc, label="Val F1 (%)", color="#1f77b4")
    ax2.bar(xc, c_bal, width=wc, label="Balanced Acc (%)", color="#ff7f0e")
    ax2.bar(xc + wc, c_auc, width=wc, label="AUROC (%)", color="#9467bd")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(c_names, fontsize=8.5, fontweight="bold", rotation=12)
    ax2.set_ylim(max(min(c_bal) - 4.0, 75.0), 100.0)
    ax2.set_ylabel("Validation Metric (%)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Autoencoder–Classifier Coupling Regimes (32×32 Latent Grid)", fontsize=10.5, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Stage 2 (Exp 2): 2D Spatial Autoencoder Compression & Coupling Strategy Ablation",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
