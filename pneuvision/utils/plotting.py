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


def plot_stage3_attention_ablation(
    attention_records: List[Dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """
    Generate a publication figure comparing Channel/Spatial Attention mechanisms
    (`none`, `eca`, `se_r8`, `se_r4`, `se_r2`, `cbam`) on top of the 32×32 Latent Grid.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    labels = [r["short_label"] for r in attention_records]
    f1s = [float(r["calibrated_metrics"]["f1_score"]) for r in attention_records]
    bals = [float(r["calibrated_metrics"]["balanced_accuracy"]) for r in attention_records]
    specs = [float(r["calibrated_metrics"]["specificity"]) for r in attention_records]
    aucs = [float(r["calibrated_metrics"]["auroc"]) for r in attention_records]

    x = np.arange(len(labels))
    w = 0.20

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    ax.bar(x - 1.5 * w, f1s, width=w, label="Val F1 (%)", color="#1f77b4")
    ax.bar(x - 0.5 * w, bals, width=w, label="Balanced Acc (%)", color="#2ca02c")
    ax.bar(x + 0.5 * w, specs, width=w, label="Specificity / TNR (%)", color="#ff7f0e")
    ax.bar(x + 1.5 * w, aucs, width=w, label="AUROC (%)", color="#9467bd")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_ylim(max(min(specs) - 3.5, 85.0), 100.0)
    ax.set_ylabel("Validation Metric (%)", fontsize=10.5, fontweight="bold")
    ax.set_title(
        "Stage 3 (Exp 3A): Channel & Spatial Attention Ablation on 16×32×32 Latent Representation",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_stage3_pareto_frontier(
    our_records: List[Dict[str, Any]],
    baseline_records: List[Dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """
    Generate the centerpiece 2-panel Pareto Frontier publication figure:
      Panel (a): Balanced Accuracy (%) vs. Inference Parameters (Log Scale)
      Panel (b): Validation F1-Score (%) vs. Inference MACs (Log Scale)
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=180)

    # Sort our records by parameter count for clean Pareto curve connection
    ours_sorted = sorted(our_records, key=lambda r: int(r["inference_params"]))
    o_params = [int(r["inference_params"]) / 1e3 for r in ours_sorted]  # in Thousands (K)
    o_macs = [float(r["macs_m"]) for r in ours_sorted]
    o_bal = [float(r["calibrated_metrics"]["balanced_accuracy"]) for r in ours_sorted]
    o_f1 = [float(r["calibrated_metrics"]["f1_score"]) for r in ours_sorted]
    o_names = [r["short_label"] for r in ours_sorted]

    b_params = [int(r["inference_params"]) / 1e3 for r in baseline_records]
    b_macs = [float(r["macs_m"]) for r in baseline_records]
    b_bal = [float(r["calibrated_metrics"]["balanced_accuracy"]) for r in baseline_records]
    b_f1 = [float(r["calibrated_metrics"]["f1_score"]) for r in baseline_records]
    b_names = [r["short_label"] for r in baseline_records]

    # Panel A: Balanced Accuracy vs Log Parameters (K)
    ax1.plot(o_params, o_bal, marker="o", markersize=9, linewidth=2.4, color="#d62728", label="TG-SECAE Family (Ours)")
    ax1.scatter(b_params, b_bal, marker="s", s=85, color="#1f77b4", zorder=5, label="Standard Vision Backbones")

    for x_p, y_v, txt in zip(o_params, o_bal, o_names):
        ax1.annotate(txt, (x_p, y_v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#b01a1a")
    for x_p, y_v, txt in zip(b_params, b_bal, b_names):
        ax1.annotate(txt, (x_p, y_v), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color="#1f4f82")

    ax1.set_xscale("log")
    ax1.set_xlabel("Trainable Inference Parameters (Thousands, Log Scale)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Validation Balanced Accuracy (%)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) Diagnostic Balanced Accuracy vs. Parameter Count", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel B: Validation F1-Score vs Log MACs (M)
    ax2.plot(o_macs, o_f1, marker="o", markersize=9, linewidth=2.4, color="#d62728", label="TG-SECAE Family (Ours)")
    ax2.scatter(b_macs, b_f1, marker="s", s=85, color="#1f77b4", zorder=5, label="Standard Vision Backbones")

    for x_m, y_v, txt in zip(o_macs, o_f1, o_names):
        ax2.annotate(txt, (x_m, y_v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color="#b01a1a")
    for x_m, y_v, txt in zip(b_macs, b_f1, b_names):
        ax2.annotate(txt, (x_m, y_v), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color="#1f4f82")

    ax2.set_xscale("log")
    ax2.set_xlabel("Inference Multiply-Accumulates per Image (MACs in Millions, Log Scale)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Validation F1-Score (%)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Validation F1-Score vs. Computational Cost (MACs)", fontsize=11, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Stage 3 (Exp 3B): Accuracy–Efficiency Pareto Frontier Across Parameter & Compute Budgets",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_stage4_gradcam_panel(
    samples: List[Dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """
    Generate a multi-row 5-column clinical interpretability figure for representative
    validation radiographs:
      Col 0: Raw Radiograph (256x256)
      Col 1: TGAE-Adaptive Enhanced Radiograph (with learned w_c, w_d)
      Col 2: 32x32 2D Spatial Bottleneck Feature Energy
      Col 3: 2D Spatial Autoencoder Reconstructed Radiograph
      Col 4: Grad-CAM Diagnostic Saliency Overlay (with predicted probability)
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(14.0, 2.9 * n_rows), dpi=180)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "1. Raw Radiograph",
        "2. TGAE Enhanced",
        "3. 32×32 Latent Grid",
        "4. AE Reconstruction",
        "5. Grad-CAM Saliency",
    ]

    for r_idx, item in enumerate(samples):
        raw_img = item["raw"]
        enh_img = item["enhanced"]
        lat_map = item["latent_map"]
        rec_img = item["reconstructed"]
        cam_map = item["gradcam"]
        wc, wd = item["wc"], item["wd"]
        prob = item["prob"]
        row_title = item["row_label"]

        axes[r_idx, 0].imshow(raw_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r_idx, 0].set_ylabel(row_title, fontsize=9.5, fontweight="bold")

        axes[r_idx, 1].imshow(enh_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r_idx, 1].set_xlabel(f"w_c={wc:.2f}, w_d={wd:.2f}", fontsize=8.5)

        axes[r_idx, 2].imshow(lat_map, cmap="viridis")
        axes[r_idx, 2].set_xlabel("16×32×32 Bottleneck", fontsize=8.5)

        axes[r_idx, 3].imshow(rec_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r_idx, 3].set_xlabel("Zero-Cost Regularizer", fontsize=8.5)

        axes[r_idx, 4].imshow(raw_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[r_idx, 4].imshow(cam_map, cmap="jet", alpha=0.42, vmin=0.0, vmax=1.0)
        axes[r_idx, 4].set_xlabel(f"P(Pneumonia) = {prob:.3f}", fontsize=8.5, fontweight="bold")

        for c_idx in range(5):
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(col_titles[c_idx], fontsize=10, fontweight="bold")
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])

    fig.suptitle(
        "Stage 4: End-to-End Clinical Interpretability of TG-SECAE (Enhancement → 32×32 Latent → Grad-CAM)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_stage4_roc_and_confusion(
    roc_curves: List[Dict[str, Any]],
    cm_dict: Dict[str, int],
    model_title: str,
    out_path: str | Path,
) -> Path:
    """
    Generate a 2-panel clinical evaluation figure:
      Panel (a): ROC Curves comparing TG-SECAE tiers & ensemble against ResNet-18.
      Panel (b): Validation Confusion Matrix (TN, FP, FN, TP) with Sensitivity & Specificity.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=180)

    # Panel A: ROC Curves
    for item in roc_curves:
        ax1.plot(
            item["fpr"],
            item["tpr"],
            linewidth=item.get("lw", 2.0),
            linestyle=item.get("ls", "-"),
            label=f"{item['label']} (AUROC = {item['auroc']:.2f}%)",
        )
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5, fontweight="bold")
    ax1.set_title("(a) Receiver Operating Characteristic (ROC) Comparison", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Confusion Matrix
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    mat = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    im = ax2.imshow(mat, cmap="Blues")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    classes = ["NORMAL (0)", "PNEUMONIA (1)"]
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(classes, fontsize=9.5, fontweight="bold")
    ax2.set_yticklabels(classes, fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Predicted Diagnosis", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Ground-Truth Diagnosis", fontsize=10.5, fontweight="bold")
    ax2.set_title(f"(b) Validation Confusion Matrix — {model_title}", fontsize=11, fontweight="bold")

    thresh_color = mat.max() / 2.0
    cell_tags = [["True Negative (TN)", "False Positive (FP)"], ["False Negative (FN)", "True Positive (TP)"]]
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            pct = 100.0 * val / max(mat[i].sum(), 1)
            ax2.text(
                j,
                i,
                f"{cell_tags[i][j]}\n{val} ({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if val > thresh_color else "black",
            )

    fig.suptitle(
        "Stage 4: Clinical Diagnostic ROC Performance & Validation Confusion Matrix",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file
