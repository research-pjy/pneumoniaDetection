"""
Publication-grade clinical classification and image reconstruction metrics.

Computes:
- Accuracy, Balanced Accuracy
- Sensitivity (Recall / True Positive Rate) — critical for pediatric pneumonia screening
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value), Negative Predictive Value (NPV)
- F1-Score
- AUROC (Area Under the ROC Curve) and AUPRC (Area Under the Precision-Recall Curve)
- Optimal probability threshold calibration (maximizing F1 or Youden's J on `val/`)
- PSNR and SSIM for Stage 1 / Stage 2 Autoencoder radiograph reconstruction
"""

from typing import Dict, Tuple

import numpy as np
import torch


def _trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute area under a curve using the trapezoidal rule (NumPy 1.x / 2.x compatible)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_auroc_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Compute exact AUROC and AUPRC from binary ground-truth labels {0, 1} and predicted
    probabilities in [0, 1] without requiring external scikit-learn installations.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5

    # Sort descending by predicted probability
    desc_idx = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[desc_idx]
    p_sorted = y_prob[desc_idx]

    # Distinct threshold indices
    distinct_idx = np.where(np.diff(p_sorted))[0]
    threshold_idxs = np.r_[distinct_idx, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    # ROC curve endpoints (0, 0) -> (1, 1)
    tpr = np.r_[0.0, tps / n_pos, 1.0]
    fpr = np.r_[0.0, fps / n_neg, 1.0]
    auroc = _trapezoid_auc(fpr, tpr)

    # Precision-Recall curve
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / n_pos
    precision_ext = np.r_[1.0, precision]
    recall_ext = np.r_[0.0, recall]
    auprc = _trapezoid_auc(recall_ext, precision_ext)

    return float(np.clip(auroc, 0.0, 1.0)), float(np.clip(auprc, 0.0, 1.0))


def compute_binary_metrics(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    threshold: float = 0.5,
) -> Dict[str, float | int]:
    """
    Compute the full suite of clinical binary classification metrics at a given threshold.
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()
    y_hat = (y_p >= threshold).astype(np.int64)

    tp = int(np.sum((y_hat == 1) & (y_t == 1)))
    tn = int(np.sum((y_hat == 0) & (y_t == 0)))
    fp = int(np.sum((y_hat == 1) & (y_t == 0)))
    fn = int(np.sum((y_hat == 0) & (y_t == 1)))

    total = max(tp + tn + fp + fn, 1)
    accuracy = (tp + tn) / total
    sensitivity = tp / max(tp + fn, 1)       # Recall / TPR
    specificity = tn / max(tn + fp, 1)       # TNR
    precision = tp / max(tp + fp, 1)         # PPV
    npv = tn / max(tn + fn, 1)               # Negative Predictive Value
    balanced_acc = 0.5 * (sensitivity + specificity)
    f1 = (2.0 * precision * sensitivity) / max(precision + sensitivity, 1e-12)

    auroc, auprc = compute_auroc_auprc(y_t, y_p)

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy * 100.0), 2),
        "balanced_accuracy": round(float(balanced_acc * 100.0), 2),
        "f1_score": round(float(f1 * 100.0), 2),
        "sensitivity_recall": round(float(sensitivity * 100.0), 2),
        "specificity": round(float(specificity * 100.0), 2),
        "precision": round(float(precision * 100.0), 2),
        "npv": round(float(npv * 100.0), 2),
        "auroc": round(float(auroc * 100.0), 2),
        "auprc": round(float(auprc * 100.0), 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_optimal_threshold(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    criterion: str = "f1",
) -> Tuple[float, Dict[str, float | int]]:
    """
    Grid-search validation probabilities in [0.10, 0.90] to find the decision threshold
    that maximizes either F1-Score ('f1'), Balanced Accuracy ('balanced_accuracy'), or
    Accuracy ('accuracy').
    """
    best_thresh = 0.5
    best_score = -1.0
    best_metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    metric_key = {
        "f1": "f1_score",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }.get(criterion, "f1_score")

    for t in np.linspace(0.15, 0.85, 71):
        m = compute_binary_metrics(y_true, y_prob, threshold=float(t))
        score = float(m[metric_key])
        if score > best_score:
            best_score = score
            best_thresh = float(t)
            best_metrics = m

    return round(best_thresh, 4), best_metrics


@torch.no_grad()
def compute_psnr_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute batch-averaged Peak Signal-to-Noise Ratio (PSNR in dB) and Structural
    Similarity Index (SSIM) for 2D Spatial Autoencoder reconstructions in [0, 1].
    """
    pred = torch.clamp(pred.float(), 0.0, data_range)
    target = torch.clamp(target.float(), 0.0, data_range)

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10((data_range ** 2) / torch.clamp(mse, min=1e-10))

    # Global/windowed structural similarity approximation
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = torch.nn.functional.avg_pool2d(pred, kernel_size=7, stride=1, padding=3)
    mu_y = torch.nn.functional.avg_pool2d(target, kernel_size=7, stride=1, padding=3)

    sigma_x = torch.nn.functional.avg_pool2d(pred * pred, 7, 1, 3) - mu_x ** 2
    sigma_y = torch.nn.functional.avg_pool2d(target * target, 7, 1, 3) - mu_y ** 2
    sigma_xy = torch.nn.functional.avg_pool2d(pred * target, 7, 1, 3) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    ssim = torch.mean(ssim_map, dim=(1, 2, 3))

    return float(psnr.mean().item()), float(ssim.mean().item())
