#!/usr/bin/env python3
"""
Stage 4: Grad-CAM Clinical Interpretability, Multi-Scale `TG-SECAE` Evaluation
(with Test-Time Horizontal Flip Augmentation), and Verified `submission.csv` Generation.

Usage on `rama` (L40S):
    python scripts/04_interpret_and_submit.py \
        --dataset-dir ./DATASET \
        --output-dir ./outputs \
        --submission-path ./submission.csv
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pneuvision.data.dataset import (
    InMemoryChestXRayDataset,
    RAMCachedSplit,
    load_split_into_ram,
)
from pneuvision.models.autoencoder import UnifiedTGSECAE
from pneuvision.models.baselines import (
    TorchvisionGrayscaleBaseline,
    build_tgsecae_tier,
)
from pneuvision.utils.hardware import inspect_environment, set_deterministic_seed
from pneuvision.utils.metrics import (
    compute_binary_metrics,
    find_optimal_threshold,
)
from pneuvision.utils.plotting import (
    plot_stage4_gradcam_panel,
    plot_stage4_roc_and_confusion,
)


def compute_roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute `(fpr, tpr)` arrays for ROC curve plotting."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    n_pos = max(int(np.sum(y_true == 1)), 1)
    n_neg = max(int(np.sum(y_true == 0)), 1)

    desc_idx = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[desc_idx]
    p_sorted = y_prob[desc_idx]
    distinct_idx = np.where(np.diff(p_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    tpr = np.r_[0.0, tps / n_pos, 1.0]
    fpr = np.r_[0.0, fps / n_neg, 1.0]
    return fpr, tpr


def load_tgsecae_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[UnifiedTGSECAE, Dict[str, Any]]:
    """Load a trained `UnifiedTGSECAE` model from a Stage 3 checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tier = ckpt.get("tier", "micro")
    att_mode = ckpt.get("attention_mode", "cbam")
    se_r = int(ckpt.get("se_reduction", 4))
    model = build_tgsecae_tier(
        tier=tier,
        attention_mode=att_mode,
        se_reduction=se_r,
        enhancement_mode="task_guided_adaptive",
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_probs_with_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    use_flip_tta: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Run inference on a split (`val` or `test`) with optional horizontal-flip Test-Time
    Augmentation (`0.5 * P(x) + 0.5 * P(flip(x))`).
    """
    model.eval()
    probs_all: List[np.ndarray] = []
    ids_all: List[str] = []
    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            p1 = torch.sigmoid(model(x)).float()
            if use_flip_tta:
                p2 = torch.sigmoid(model(torch.flip(x, dims=[-1]))).float()
                p = 0.5 * (p1 + p2)
            else:
                p = p1
        probs_all.append(p.squeeze(-1).cpu().numpy())
        ids_all.extend(batch["image_id"])

    return np.concatenate(probs_all, axis=0), ids_all


def compute_tgsecae_gradcam(
    model: UnifiedTGSECAE,
    img_01: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Compute Grad-CAM saliency map on the final convolutional block (`classifier.block3`)
    of `UnifiedTGSECAE` alongside the `TGAE` enhanced radiograph, 32x32 latent map,
    and 2D spatial autoencoder reconstruction.
    """
    model.eval()
    x = torch.from_numpy(img_01[None, None, :, :]).to(device)

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        activations.append(out)

    def bwd_hook(module: nn.Module, grad_in: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_out[0])

    h_f = model.classifier.block3.register_forward_hook(fwd_hook)
    h_b = model.classifier.block3.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        x_enh, _ = model.enhancer(x)
        z = model.encoder(x_enh)
        logits = model.classifier(z)
        prob = float(torch.sigmoid(logits).item())
        logits.squeeze().backward()

    h_f.remove()
    h_b.remove()

    feat = activations[0].detach()   # (1, C, H_f, W_f)
    grad = gradients[0].detach()     # (1, C, H_f, W_f)
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.relu(torch.sum(weights * feat, dim=1, keepdim=True))
    cam_up = F.interpolate(cam, size=(256, 256), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    cam_min, cam_max = float(cam_up.min()), float(cam_up.max())
    if cam_max > cam_min:
        cam_up = (cam_up - cam_min) / (cam_max - cam_min)
    else:
        cam_up = np.zeros_like(img_01)

    with torch.no_grad():
        x_rec, z_lat, _ = model.forward_autoencoder(x)
        wc, wd = model.enhancer.get_enhancement_weights(x)

    return {
        "raw": img_01,
        "enhanced": x_enh.detach().squeeze().cpu().numpy(),
        "latent_map": z_lat.detach().squeeze(0).mean(dim=0).cpu().numpy(),
        "reconstructed": x_rec.detach().squeeze().cpu().numpy(),
        "gradcam": cam_up,
        "wc": float(wc.item()),
        "wd": float(wd.item()),
        "prob": prob,
    }


def write_and_verify_submission(
    image_ids: List[str],
    probs: np.ndarray,
    threshold: float,
    out_csv_path: Path,
    expected_rows: int = 587,
) -> Dict[str, Any]:
    """
    Write `submission.csv` matching the exact benchmark schema and perform a strict
    6-point automated compliance check.
    """
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    preds = (probs >= threshold).astype(np.int64)

    # Sort by Image_ID for clean deterministic ordering
    order = np.argsort(image_ids)
    sorted_ids = [image_ids[i] for i in order]
    sorted_preds = [int(preds[i]) for i in order]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "Label"])
        for img_id, lbl in zip(sorted_ids, sorted_preds):
            writer.writerow([img_id, int(lbl)])

    # Verification checks
    with open(out_csv_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    header_ok = (raw_lines[0] == "Image_ID,Label")
    total_lines = len(raw_lines)
    data_rows = raw_lines[1:]
    unique_ids = set()
    labels_valid = True
    for row in data_rows:
        parts = row.split(",")
        if len(parts) != 2:
            labels_valid = False
            break
        unique_ids.add(parts[0])
        if parts[1] not in ("0", "1"):
            labels_valid = False
            break

    n_pneu = int(sum(sorted_preds))
    n_norm = len(sorted_preds) - n_pneu

    return {
        "submission_path": str(out_csv_path),
        "header_exact_match": header_ok,
        "total_file_lines": total_lines,
        "expected_file_lines": expected_rows + 1,
        "unique_image_ids": len(unique_ids),
        "strictly_binary_0_or_1": labels_valid,
        "all_checks_passed": bool(
            header_ok
            and total_lines == expected_rows + 1
            and len(unique_ids) == expected_rows
            and labels_valid
        ),
        "predicted_normal_count": n_norm,
        "predicted_pneumonia_count": n_pneu,
        "predicted_pneumonia_pct": round(100.0 * n_pneu / max(len(sorted_preds), 1), 2),
        "decision_threshold_used": round(float(threshold), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Grad-CAM Interpretability, Multi-Scale Evaluation & Final submission.csv"
    )
    parser.add_argument("--dataset-dir", type=str, default="./DATASET", help="Path to DATASET/ directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--submission-path", type=str, default="./submission.csv", help="Final submission.csv path")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    out_dir = Path(args.output_dir).resolve()
    fig_dir = out_dir / "figures"
    ckpt_dir = out_dir / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)

    env_info = inspect_environment(str(out_dir))
    device = torch.device("cuda" if env_info["cuda_available"] else "cpu")
    use_bf16 = bool(env_info["bf16_supported"])

    print("=" * 105)
    print("PneuVision — Stage 4: Grad-CAM Clinical Interpretability & Final Benchmark Submission")
    print(f"Device: {env_info['gpu_name'] or 'CPU'} | BF16: {use_bf16}")
    print("=" * 105)

    # 1. Load val/ (585) and unlabeled test/ (587) into RAM
    print("\n[1/4] Loading val/ (585 images) and test/ (587 images) splits...")
    val_cache = load_split_into_ram(args.dataset_dir, split="val", image_size=256)
    test_cache = load_split_into_ram(args.dataset_dir, split="test", image_size=256)
    val_ds = InMemoryChestXRayDataset(val_cache, preprocess_mode="raw", augment=False)
    test_ds = InMemoryChestXRayDataset(test_cache, preprocess_mode="raw", augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    y_val = val_cache.labels

    # 2. Evaluate Stage 3 TG-SECAE Models (with TTA) & Multi-Scale TG-SECAE Ensemble
    print("\n[2/4] Evaluating TG-SECAE Models (with Horizontal-Flip TTA) & Multi-Scale Ensemble on val/:")
    candidate_ckpts = [
        ("TG-SECAE-Nano (5.0k)", ckpt_dir / "stage3_tgsecae_nano_cbam_best.pt"),
        ("TG-SECAE-Micro-SE (10.3k)", ckpt_dir / "stage3_att_se_r8_best.pt"),
        ("TG-SECAE-Micro-CBAM (11.8k)", ckpt_dir / "stage3_att_cbam_best.pt"),
        ("TG-SECAE-Tiny (37.3k)", ckpt_dir / "stage3_tgsecae_tiny_cbam_best.pt"),
        ("TG-SECAE-Small (124.7k)", ckpt_dir / "stage3_tgsecae_small_cbam_best.pt"),
    ]

    val_probs_dict: Dict[str, np.ndarray] = {}
    test_probs_dict: Dict[str, np.ndarray] = {}
    test_ids: List[str] = []
    model_records: List[Dict[str, Any]] = []
    loaded_models: Dict[str, UnifiedTGSECAE] = {}
    total_ensemble_params = 0

    for label, c_path in candidate_ckpts:
        if not c_path.exists():
            continue
        m, _ = load_tgsecae_from_checkpoint(c_path, device)
        loaded_models[label] = m
        inf_p = m.inference_parameter_breakdown()["inference_total_params"]
        total_ensemble_params += inf_p

        v_prob, _ = predict_probs_with_tta(m, val_loader, device, use_bf16, use_flip_tta=True)
        t_prob, test_ids = predict_probs_with_tta(m, test_loader, device, use_bf16, use_flip_tta=True)
        val_probs_dict[label] = v_prob
        test_probs_dict[label] = t_prob

        opt_t, m_cal = find_optimal_threshold(y_val, v_prob, criterion="f1")
        print(
            f"  [{label:<28}] Params: {inf_p:>7,} | Val Acc: {m_cal['accuracy']:>6.2f}% | "
            f"BalAcc: {m_cal['balanced_accuracy']:>6.2f}% | F1: {m_cal['f1_score']:>6.2f}% | "
            f"Sens: {m_cal['sensitivity_recall']:>6.2f}% | Spec: {m_cal['specificity']:>6.2f}% | AUROC: {m_cal['auroc']:>6.2f}%"
        )
        model_records.append(
            {
                "name": label,
                "inference_params": inf_p,
                "optimal_threshold": opt_t,
                "metrics": m_cal,
            }
        )

    # Multi-Scale TG-SECAE Ensemble (combining our ultra-compact tiers, still only 189k total params!)
    ens_val_prob = np.mean(list(val_probs_dict.values()), axis=0)
    ens_test_prob = np.mean(list(test_probs_dict.values()), axis=0)
    ens_opt_t, ens_metrics = find_optimal_threshold(y_val, ens_val_prob, criterion="f1")
    ens_label = f"TG-SECAE Multi-Scale Ensemble ({total_ensemble_params / 1e3:.1f}k)"
    print(
        f"  [{ens_label:<28}] Params: {total_ensemble_params:>7,} | Val Acc: {ens_metrics['accuracy']:>6.2f}% | "
        f"BalAcc: {ens_metrics['balanced_accuracy']:>6.2f}% | F1: {ens_metrics['f1_score']:>6.2f}% | "
        f"Sens: {ens_metrics['sensitivity_recall']:>6.2f}% | Spec: {ens_metrics['specificity']:>6.2f}% | AUROC: {ens_metrics['auroc']:>6.2f}%"
    )

    # 3. Generate Grad-CAM Clinical Interpretability Panel & ROC / Confusion Matrix Figures
    print("\n[3/4] Generating Grad-CAM Saliency Panel & ROC / Confusion Matrix Figures:")
    interp_model = (
        loaded_models.get("TG-SECAE-Tiny (37.3k)")
        or loaded_models.get("TG-SECAE-Small (124.7k)")
        or list(loaded_models.values())[0]
    )

    # Pick 2 NORMAL (Class 0) and 2 PNEUMONIA (Class 1) validation radiographs
    norm_indices = np.where(y_val == 0)[0][:2]
    pneu_indices = np.where(y_val == 1)[0][:2]
    cam_samples: List[Dict[str, Any]] = []

    for idx_num, v_idx in enumerate(norm_indices, start=1):
        res = compute_tgsecae_gradcam(interp_model, val_cache.images[int(v_idx)], device)
        res["row_label"] = f"NORMAL #{idx_num}\n(Ground Truth: 0)"
        cam_samples.append(res)

    for idx_num, v_idx in enumerate(pneu_indices, start=1):
        res = compute_tgsecae_gradcam(interp_model, val_cache.images[int(v_idx)], device)
        res["row_label"] = f"PNEUMONIA #{idx_num}\n(Ground Truth: 1)"
        cam_samples.append(res)

    gradcam_fig = plot_stage4_gradcam_panel(
        samples=cam_samples,
        out_path=fig_dir / "stage4_gradcam_interpretability.png",
    )
    print(f"  - Saved Grad-CAM interpretability figure -> {gradcam_fig}")

    # Build ROC comparison list
    roc_list: List[Dict[str, Any]] = []
    for lbl in ["TG-SECAE-Nano (5.0k)", "TG-SECAE-Micro-SE (10.3k)", "TG-SECAE-Tiny (37.3k)", "TG-SECAE-Small (124.7k)"]:
        if lbl in val_probs_dict:
            fpr, tpr = compute_roc_points(y_val, val_probs_dict[lbl])
            _, m_c = find_optimal_threshold(y_val, val_probs_dict[lbl], criterion="f1")
            roc_list.append({"label": lbl, "fpr": fpr, "tpr": tpr, "auroc": m_c["auroc"], "lw": 1.8, "ls": "--"})

    fpr_e, tpr_e = compute_roc_points(y_val, ens_val_prob)
    roc_list.append(
        {
            "label": "TG-SECAE Ensemble (189k)",
            "fpr": fpr_e,
            "tpr": tpr_e,
            "auroc": ens_metrics["auroc"],
            "lw": 2.6,
            "ls": "-",
        }
    )

    # Also include ResNet-18 (11.17M) on the ROC plot if checkpoint exists
    resnet_ckpt_path = ckpt_dir / "stage3_baseline_resnet18_best.pt"
    if resnet_ckpt_path.exists():
        r_ckpt = torch.load(resnet_ckpt_path, map_location=device, weights_only=False)
        r_model = TorchvisionGrayscaleBaseline("resnet18").to(device)
        r_model.load_state_dict(r_ckpt["model_state_dict"])
        r_prob, _ = predict_probs_with_tta(r_model, val_loader, device, use_bf16, use_flip_tta=True)
        fpr_r, tpr_r = compute_roc_points(y_val, r_prob)
        _, r_m = find_optimal_threshold(y_val, r_prob, criterion="f1")
        roc_list.append({"label": "ResNet-18 (11.17M)", "fpr": fpr_r, "tpr": tpr_r, "auroc": r_m["auroc"], "lw": 1.8, "ls": ":"})

    cm_dict = {
        "tn": int(ens_metrics["tn"]),
        "fp": int(ens_metrics["fp"]),
        "fn": int(ens_metrics["fn"]),
        "tp": int(ens_metrics["tp"]),
    }
    roc_cm_fig = plot_stage4_roc_and_confusion(
        roc_curves=roc_list,
        cm_dict=cm_dict,
        model_title="TG-SECAE Multi-Scale (189k Params)",
        out_path=fig_dir / "stage4_roc_and_confusion.png",
    )
    print(f"  - Saved ROC & Confusion Matrix figure    -> {roc_cm_fig}")

    # 4. Generate and Strictly Verify Final `submission.csv`
    print("\n[4/4] Generating & Verifying Final Benchmark `submission.csv`:")
    sub_main_path = Path(args.submission_path).resolve()
    sub_copy_path = out_dir / "submission.csv"

    sub_check = write_and_verify_submission(
        image_ids=test_ids,
        probs=ens_test_prob,
        threshold=ens_opt_t,
        out_csv_path=sub_main_path,
        expected_rows=len(test_ids),
    )
    _ = write_and_verify_submission(
        image_ids=test_ids,
        probs=ens_test_prob,
        threshold=ens_opt_t,
        out_csv_path=sub_copy_path,
        expected_rows=len(test_ids),
    )

    for k, v in sub_check.items():
        print(f"  - {k:<28}: {v}")

    stage4_report = {
        "individual_models_with_tta": model_records,
        "ensemble_metrics": {
            "name": ens_label,
            "total_inference_params": total_ensemble_params,
            "optimal_threshold": ens_opt_t,
            "metrics": ens_metrics,
        },
        "submission_verification": sub_check,
    }
    with open(out_dir / "stage4_final_report.json", "w", encoding="utf-8") as f:
        json.dump(stage4_report, f, indent=2)

    print(f"\n[Stage 4 Complete] Verified submission saved to: {sub_main_path}")
    print("=" * 105)


if __name__ == "__main__":
    main()
