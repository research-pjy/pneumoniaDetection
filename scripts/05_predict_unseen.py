#!/usr/bin/env python3
"""
PneuVision — Universal Inference, Evaluation & Visual Explanation on Unseen Chest X-Rays
========================================================================================

This standalone script allows you to run trained TG-SECAE models on ANY new/unseen chest
X-ray data during a live evaluation, viva demonstration, or hidden-test verification:

  Case 1 — Folder of Unlabeled X-Rays (generates `Image_ID,Label` submission CSV):
      python scripts/05_predict_unseen.py \
          --input /path/to/unseen_folder \
          --output-csv ./unseen_submission.csv

  Case 2 — Folder of Labeled X-Rays (`NORMAL/` and `PNEUMONIA/` subfolders):
      python scripts/05_predict_unseen.py \
          --input /path/to/labeled_test_folder \
          --output-csv ./labeled_predictions.csv
      (Automatically detects ground-truth folders and prints Accuracy, Balanced Accuracy,
       Sensitivity, Specificity, F1, AUROC, and Confusion Matrix!)

  Case 3 — Single Chest X-Ray Image (prints diagnosis + saves 5-panel Grad-CAM figure):
      python scripts/05_predict_unseen.py \
          --input /path/to/single_xray.jpeg \
          --explain-out ./single_xray_gradcam.png

Model Selection (`--model`):
  - `ensemble`   (Default): 5-Model Multi-Scale TG-SECAE Ensemble (189,138 total params)
  - `nano`       : TG-SECAE-Nano       (  4,993 params — 19.5 KB)
  - `micro_se`   : TG-SECAE-Micro-SE   ( 10,313 params — 40.3 KB)
  - `micro_cbam` : TG-SECAE-Micro-CBAM ( 11,848 params — 46.3 KB)
  - `tiny`       : TG-SECAE-Tiny       ( 37,279 params — 145.6 KB)
  - `small`      : TG-SECAE-Small      (124,705 params — 487.1 KB)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pneuvision.data.enhancement import apply_clahe_numpy
from pneuvision.models.classifier import TGSECAEClassifier
from pneuvision.utils.hardware import count_parameters, seed_everything
from pneuvision.utils.metrics import compute_binary_metrics


VALID_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}


def natural_sort_key(s: str):
    """Sort strings containing numbers in natural numerical order (e.g., img_2 before img_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def load_single_xray(path: Path, image_size: int = 256) -> np.ndarray:
    """
    Deterministic preprocessing pipeline identical to training:
      1. Load radiograph and convert to single-channel grayscale ('L')
      2. Bilinear resize to (256, 256)
      3. Min-max normalize intensity to [0.0, 1.0] in float32
    """
    with Image.open(path) as img:
        gray = img.convert("L")
        if gray.size != (image_size, image_size):
            gray = gray.resize((image_size, image_size), resample=Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax - vmin > 1e-6:
            arr = (arr - vmin) / (vmax - vmin)
        return arr


def discover_input_images(input_path: Path) -> Tuple[str, List[Tuple[Path, int, str]]]:
    """
    Inspect `input_path` and determine whether it is:
      - 'single'   : A single radiograph file
      - 'labeled'  : A directory with `NORMAL/` and `PNEUMONIA/` subdirectories
      - 'unlabeled': A flat or nested directory of unlabeled radiographs

    Returns:
      mode    : 'single' | 'labeled' | 'unlabeled'
      records : List of (filepath, label_int, image_id) where label_int is 0/1 or -1 if unlabeled.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return "single", [(input_path, -1, input_path.name)]

    # Check if NORMAL/ and PNEUMONIA/ subdirectories exist (case-insensitive)
    subdirs = {d.name.upper(): d for d in input_path.iterdir() if d.is_dir()}
    if "NORMAL" in subdirs and "PNEUMONIA" in subdirs:
        records: List[Tuple[Path, int, str]] = []
        for cls_name, label_int in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            cls_dir = subdirs[cls_name]
            files = sorted(
                [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS],
                key=lambda p: natural_sort_key(p.name),
            )
            for f in files:
                records.append((f, label_int, f.name))
        return "labeled", records

    # Otherwise, treat as unlabeled directory of images
    files = sorted(
        [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS],
        key=lambda p: natural_sort_key(p.name),
    )
    records = [(f, -1, f.name) for f in files]
    return "unlabeled", records


MODEL_SPECS: Dict[str, Tuple[str, str, Dict]] = {
    "nano": (
        "TG-SECAE-Nano (5.0k)",
        "stage3_tgsecae_nano_cbam_best.pt",
        dict(tier="nano", latent_channels=12, base_channels=8, attention_type="cbam", se_reduction=4),
    ),
    "micro_se": (
        "TG-SECAE-Micro-SE (10.3k)",
        "stage3_att_se_r8_best.pt",
        dict(tier="micro", latent_channels=16, base_channels=12, attention_type="se", se_reduction=8),
    ),
    "micro_cbam": (
        "TG-SECAE-Micro-CBAM (11.8k)",
        "stage3_att_cbam_best.pt",
        dict(tier="micro", latent_channels=16, base_channels=12, attention_type="cbam", se_reduction=4),
    ),
    "tiny": (
        "TG-SECAE-Tiny (37.3k)",
        "stage3_tgsecae_tiny_cbam_best.pt",
        dict(tier="tiny", latent_channels=24, base_channels=16, attention_type="cbam", se_reduction=4),
    ),
    "small": (
        "TG-SECAE-Small (124.7k)",
        "stage3_tgsecae_small_cbam_best.pt",
        dict(tier="small", latent_channels=32, base_channels=24, attention_type="cbam", se_reduction=4),
    ),
}


def resolve_checkpoint_path(ckpt_name: str, custom_ckpt_dir: Optional[Path] = None) -> Path:
    """Locate a trained checkpoint in custom_ckpt_dir, ./pretrained_models/, or ./outputs/checkpoints/."""
    search_dirs = []
    if custom_ckpt_dir is not None:
        search_dirs.append(custom_ckpt_dir)
    search_dirs.extend([
        PROJECT_ROOT / "pretrained_models",
        PROJECT_ROOT / "outputs" / "checkpoints",
    ])
    for d in search_dirs:
        candidate = d / ckpt_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find checkpoint '{ckpt_name}' in {[str(d) for d in search_dirs]}"
    )


def load_trained_models(
    model_choice: str,
    device: torch.device,
    ckpt_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[str, TGSECAEClassifier]], float]:
    """
    Load either a single TG-SECAE tier or all 5 tiers for the Multi-Scale Ensemble.
    Returns (list of (model_label, model), calibrated_threshold).
    """
    keys = list(MODEL_SPECS.keys()) if model_choice == "ensemble" else [model_choice]
    loaded: List[Tuple[str, TGSECAEClassifier]] = []
    thresholds: List[float] = []

    for k in keys:
        label, ckpt_file, kwargs = MODEL_SPECS[k]
        ckpt_path = resolve_checkpoint_path(ckpt_file, ckpt_dir)
        model = TGSECAEClassifier(
            in_channels=1,
            use_tgae=True,
            tgae_Adaptive=True,
            dropout=0.0,
            **kwargs,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        loaded.append((label, model))
        thresholds.append(float(ckpt.get("best_threshold", 0.40)))

    # Default decision threshold: calibrated 0.35 for ensemble or checkpoint optimal threshold
    default_thr = 0.35 if model_choice == "ensemble" else float(np.mean(thresholds))
    return loaded, default_thr


@torch.no_grad()
def predict_probabilities(
    models: List[Tuple[str, TGSECAEClassifier]],
    images_np: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    use_tta: bool = True,
) -> np.ndarray:
    """
    Compute pneumonia probabilities on `images_np` (N, H, W) using optional Horizontal-Flip TTA.
    If multiple models are provided (`ensemble`), averages their calibrated probabilities.
    """
    n = len(images_np)
    all_model_probs = []

    for _, model in models:
        model.eval()
        probs_list = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(images_np[start:end]).unsqueeze(1).to(device=device, dtype=torch.float32)
            p_orig = torch.sigmoid(model(x).view(-1))
            if use_tta:
                x_flip = torch.flip(x, dims=[3])
                p_flip = torch.sigmoid(model(x_flip).view(-1))
                p_batch = 0.5 * (p_orig + p_flip)
            else:
                p_batch = p_orig
            probs_list.append(p_batch.cpu().numpy())
        all_model_probs.append(np.concatenate(probs_list, axis=0))

    return np.mean(np.stack(all_model_probs, axis=0), axis=0)


def save_single_image_explanation(
    model: TGSECAEClassifier,
    img_np: np.ndarray,
    image_id: str,
    prob: float,
    pred_label: int,
    out_path: Path,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Generate a 5-panel clinical explanation figure for a single radiograph:
      [Raw X-Ray | Fixed Full CLAHE | Learned TGAE Output | 2D Latent Energy | Grad-CAM Overlay]
    Returns learned TGAE gates (w_c, w_d).
    """
    model.eval()
    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    # Extract TGAE enhanced image, gates, and 2D spatial latent map
    with torch.no_grad():
        x_tgae, gates = model.enhancer(x)
        w_c = float(gates["w_contrast"].item())
        w_d = float(gates["w_detail"].item())
        z_latent = model.extract_latent(x)
        latent_energy = z_latent.abs().mean(dim=1).squeeze(0).cpu().numpy()
        lmin, lmax = float(latent_energy.min()), float(latent_energy.max())
        if lmax - lmin > 1e-6:
            latent_energy = (latent_energy - lmin) / (lmax - lmin)
        tgae_np = x_tgae.squeeze().cpu().numpy()

    # Compute Grad-CAM on the final residual stage (stage2)
    activations = []
    gradients = []
    h_f = model.stage2.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    h_b = model.stage2.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))
    try:
        x_grad = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logit = model(x_grad).view(-1)[0]
        logit.backward()
        acts = activations[0]
        grads = gradients[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=img_np.shape, mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        cmin, cmax = float(cam_np.min()), float(cam_np.max())
        if cmax - cmin > 1e-6:
            cam_np = (cam_np - cmin) / (cmax - cmin)
    finally:
        h_f.remove()
        h_b.remove()

    clahe_full = apply_clahe_numpy(img_np, clip_limit=4.0, tile_grid_size=(8, 8))
    cls_str = "PNEUMONIA (1)" if pred_label == 1 else "NORMAL (0)"

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), dpi=250)
    fig.suptitle(
        f"PneuVision Clinical Diagnostic Explanation — {image_id} | Predicted: {cls_str} (P = {prob:.1%})",
        fontsize=12,
        fontweight="bold",
    )

    axes[0].imshow(img_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("1. Raw Radiograph", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(clahe_full, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("2. Fixed Full CLAHE (alpha=1.0)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(tgae_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"3. Learned TGAE\n(w_c={w_c:.2f}, w_d={w_d:.2f})", fontsize=10)
    axes[2].axis("off")

    axes[3].imshow(latent_energy, cmap="magma")
    axes[3].set_title("4. 2D Spatial Latent (32x32)", fontsize=10)
    axes[3].axis("off")

    axes[4].imshow(img_np, cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(cam_np, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    axes[4].set_title(f"5. Grad-CAM Localization\n({cls_str}, {prob:.1%})", fontsize=10)
    axes[4].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return w_c, w_d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PneuVision: Run inference, evaluation, or Grad-CAM explanation on any unseen chest X-rays."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a folder of unseen X-rays (unlabeled or with NORMAL/PNEUMONIA folders) OR a single X-ray image.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble", "nano", "micro_se", "micro_cbam", "tiny", "small"],
        help="Which trained TG-SECAE tier to run (default: 'ensemble' across all 5 tiers).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./unseen_predictions.csv",
        help="Output path for the `Image_ID,Label` CSV file (default: ./unseen_predictions.csv).",
    )
    parser.add_argument(
        "--detailed-csv",
        type=str,
        default="",
        help="Optional path to save detailed predictions including exact probability scores.",
    )
    parser.add_argument(
        "--explain-out",
        type=str,
        default="./outputs/figures/unseen_explanation.png",
        help="Output PNG path for the 5-panel Grad-CAM explanation (used when --input is a single image).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold in (0, 1). Defaults to the model's calibrated validation threshold.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable Horizontal-Flip Test-Time Augmentation (TTA is enabled by default).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="",
        help="Optional custom directory containing .pt checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None

    print("=" * 100)
    print("PneuVision — Universal Unseen X-Ray Inference, Clinical Evaluation & Grad-CAM Pipeline")
    print(f"Device: {device} | Model Choice: {args.model} | TTA Enabled: {not args.no_tta}")
    print("=" * 100)

    # 1. Load model(s)
    models, default_thr = load_trained_models(args.model, device=device, ckpt_dir=ckpt_dir)
    threshold = float(args.threshold) if args.threshold is not None else default_thr
    total_params = sum(count_parameters(m) for _, m in models)
    print(f"\n[1/3] Loaded {len(models)} TG-SECAE model(s) | Total Inference Parameters: {total_params:,} | Threshold: {threshold:.2f}")
    for name, m in models:
        print(f"  - {name:<30} : {count_parameters(m):>7,} params")

    # 2. Discover & preprocess input images
    input_path = Path(args.input).resolve()
    mode, records = discover_input_images(input_path)
    if len(records) == 0:
        raise RuntimeError(f"No valid radiograph images found in: {input_path}")

    print(f"\n[2/3] Discovered {len(records)} radiograph(s) in mode='{mode}' from: {input_path}")
    images_np = np.stack([load_single_xray(r[0], image_size=256) for r in records], axis=0)
    labels_np = np.array([r[1] for r in records], dtype=np.int64)
    image_ids = [r[2] for r in records]

    # 3. Run inference
    probs = predict_probabilities(
        models=models,
        images_np=images_np,
        device=device,
        batch_size=args.batch_size,
        use_tta=not args.no_tta,
    )
    preds = (probs >= threshold).astype(np.int64)

    # Handle Single-Image Mode
    if mode == "single":
        prob_val = float(probs[0])
        pred_val = int(preds[0])
        cls_name = "PNEUMONIA (Label = 1)" if pred_val == 1 else "NORMAL (Label = 0)"
        explain_path = Path(args.explain_out).resolve()
        # Use Micro-CBAM (or the loaded model) for the 5-panel Grad-CAM visualization
        vis_model = models[min(2, len(models) - 1)][1]
        w_c, w_d = save_single_image_explanation(
            model=vis_model,
            img_np=images_np[0],
            image_id=image_ids[0],
            prob=prob_val,
            pred_label=pred_val,
            out_path=explain_path,
            device=device,
        )
        print("\n[3/3] Single Radiograph Diagnostic Report:")
        print(f"  - Image_ID                : {image_ids[0]}")
        print(f"  - Predicted Diagnosis     : {cls_name}")
        print(f"  - Pneumonia Probability   : {prob_val:.4f} ({prob_val * 100:.2f}%)")
        print(f"  - Decision Threshold Used : {threshold:.2f}")
        print(f"  - Learned TGAE Gates      : w_contrast = {w_c:.4f}, w_detail = {w_d:.4f}")
        print(f"  - 5-Panel Grad-CAM Saved  : {explain_path}")

    # Handle Labeled Folder Mode (`NORMAL/` and `PNEUMONIA/` present)
    elif mode == "labeled":
        metrics = compute_binary_metrics(labels_np, probs, threshold=threshold)
        print("\n[3/3] Ground-Truth Labeled Evaluation Report on Unseen Dataset:")
        print(f"  - Total Evaluated Images  : {len(records)} (NORMAL: {int((labels_np == 0).sum())}, PNEUMONIA: {int((labels_np == 1).sum())})")
        print(f"  - Accuracy                : {metrics['accuracy'] * 100:6.2f}%")
        print(f"  - Balanced Accuracy       : {metrics['balanced_accuracy'] * 100:6.2f}%")
        print(f"  - Sensitivity (Recall)    : {metrics['sensitivity'] * 100:6.2f}%")
        print(f"  - Specificity             : {metrics['specificity'] * 100:6.2f}%")
        print(f"  - Precision (PPV)         : {metrics['precision'] * 100:6.2f}%")
        print(f"  - F1-Score                : {metrics['f1'] * 100:6.2f}%")
        print(f"  - AUROC                   : {metrics['auroc'] * 100:6.2f}%")
        print(f"  - Confusion Matrix        : TN={int(metrics['tn'])}, FP={int(metrics['fp'])}, FN={int(metrics['fn'])}, TP={int(metrics['tp'])}")

    # Handle Unlabeled Folder Mode
    else:
        n_norm = int((preds == 0).sum())
        n_pneu = int((preds == 1).sum())
        print("\n[3/3] Unlabeled Batch Prediction Summary:")
        print(f"  - Total Predicted Images  : {len(records)}")
        print(f"  - Predicted NORMAL (0)    : {n_norm} ({100.0 * n_norm / len(records):.2f}%)")
        print(f"  - Predicted PNEUMONIA (1) : {n_pneu} ({100.0 * n_pneu / len(records):.2f}%)")

    # Write `Image_ID,Label` CSV
    if args.output_csv:
        csv_path = Path(args.output_csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image_ID", "Label"])
            for img_id, pred_label in zip(image_ids, preds):
                writer.writerow([img_id, int(pred_label)])
        print(f"  - Saved Benchmark CSV     : {csv_path} ({len(records) + 1} lines)")

    # Write optional detailed CSV with probabilities
    if args.detailed_csv:
        det_path = Path(args.detailed_csv).resolve()
        det_path.parent.mkdir(parents=True, exist_ok=True)
        with open(det_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image_ID", "Label", "Probability_Pneumonia", "Predicted_Class"])
            for img_id, pred_label, p_val in zip(image_ids, preds, probs):
                writer.writerow([img_id, int(pred_label), f"{float(p_val):.6f}", "PNEUMONIA" if pred_label == 1 else "NORMAL"])
        print(f"  - Saved Detailed CSV      : {det_path}")

    print("=" * 100)


if __name__ == "__main__":
    main()
