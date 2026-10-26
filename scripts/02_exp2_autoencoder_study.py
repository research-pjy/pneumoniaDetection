#!/usr/bin/env python3
"""
Stage 2 (Exp 2): 2D Spatial Autoencoder Bottleneck Geometry & Coupling Regime Ablation Study.

Investigates two publication questions (Idea 1):
  - Exp 2A (Table 2A): Why 2D Spatial Bottlenecks (`16x16`, `32x32`, `64x64`) dramatically
    outperform a 1D Vector Bottleneck (`R^64`) in parameter efficiency, radiograph
    reconstruction quality (PSNR / SSIM), and diagnostic classification.
  - Exp 2B (Table 2B): How the Autoencoder and Classifier should interact across 5 coupling
    regimes:
      1. `two_stage_frozen`      (Unsupervised AE -> Frozen Encoder E -> Classifier)
      2. `two_stage_finetune`    (Unsupervised AE -> Fine-tuned E + Classifier)
      3. `enhance_ae_finetune`   (Raw-to-Enhanced AE pretraining -> Fine-tuned E + Classifier)
      4. `joint_multitask`       (End-to-End Multi-Task BCE + Reconstruction Regularizer)
      5. `transductive_ae_joint` (Unsupervised AE pretrained on `train + test` -> Joint Multi-Task)

Usage on `rama` (L40S):
    # Step 2.1 — Mandatory 2-epoch pipeline smoke test first:
    python scripts/02_exp2_autoencoder_study.py --dataset-dir ./DATASET --output-dir ./outputs --smoke-test

    # Step 2.2 — Full Stage 2 Publication Study:
    python scripts/02_exp2_autoencoder_study.py \
        --dataset-dir ./DATASET \
        --output-dir ./outputs \
        --ae-epochs 15 \
        --cls-epochs 25 \
        --batch-size 64
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pneuvision.data.dataset import (
    InMemoryChestXRayDataset,
    RAMCachedSplit,
    load_split_into_ram,
)
from pneuvision.data.enhancement import apply_progressive_enhancement_numpy
from pneuvision.models.autoencoder import (
    UnifiedTGSECAE,
    VectorAutoencoder1D,
)
from pneuvision.utils.hardware import (
    inspect_environment,
    profile_model_efficiency,
    set_deterministic_seed,
)
from pneuvision.utils.metrics import (
    compute_binary_metrics,
    compute_psnr_ssim,
    find_optimal_threshold,
)
from pneuvision.utils.plotting import (
    plot_stage2_ae_reconstructions,
    plot_stage2_coupling_comparison,
)


class PairedEnhancementAEDataset(Dataset):
    """
    In-memory dataset for Autoencoder pretraining.
    If `target_enhanced=True`, returns `(raw_xray, clahe_enhanced_xray)` so the Autoencoder
    learns to reconstruct contrast-enhanced lung structures from raw inputs.
    """

    def __init__(
        self,
        cached_split: RAMCachedSplit,
        target_enhanced: bool = False,
        alpha_target: float = 0.50,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.augment = augment
        raw = cached_split.images
        self.inputs = torch.from_numpy(raw[:, None, :, :]).contiguous()

        if target_enhanced:
            enh_list = [
                apply_progressive_enhancement_numpy(raw[i], alpha=alpha_target)[None, :, :]
                for i in range(raw.shape[0])
            ]
            self.targets = torch.from_numpy(np.stack(enh_list, axis=0)).contiguous()
        else:
            self.targets = self.inputs

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]
        t = self.targets[idx]
        if self.augment and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
            t = torch.flip(t, dims=[-1])
        return x, t


def pretrain_autoencoder_2d(
    model: UnifiedTGSECAE,
    ae_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    use_bf16: bool,
) -> Tuple[float, float]:
    """
    Pretrain `enhancer + encoder + decoder` on hybrid MSE + L1 reconstruction loss,
    and return `(val_psnr_db, val_ssim)`.
    """
    optimizer = torch.optim.AdamW(
        list(model.enhancer.parameters())
        + list(model.encoder.parameters())
        + list(model.decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.1)
    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    for _ in range(epochs):
        model.train()
        for x_in, x_tgt in ae_loader:
            x_in = x_in.to(device, non_blocking=True)
            x_tgt = x_tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                x_rec, _, reg_loss = model.forward_autoencoder(x_in)
                rec_loss = 0.5 * F.mse_loss(x_rec, x_tgt) + 0.5 * F.l1_loss(x_rec, x_tgt)
                loss = rec_loss + 0.005 * reg_loss
            loss.backward()
            optimizer.step()
        scheduler.step()

    return evaluate_ae_reconstruction(model, val_loader, device, use_bf16, is_1d=False)


@torch.no_grad()
def evaluate_ae_reconstruction(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    is_1d: bool = False,
) -> Tuple[float, float]:
    """Compute average `(PSNR_dB, SSIM)` across `val/`."""
    model.eval()
    psnrs: List[float] = []
    ssims: List[float] = []
    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            if is_1d:
                x_rec, _ = model.forward_autoencoder(x)  # type: ignore[operator]
            else:
                x_rec, _, _ = model.forward_autoencoder(x)  # type: ignore[operator]
        p, s = compute_psnr_ssim(x_rec.float(), x.float())
        psnrs.append(p)
        ssims.append(s)

    return round(float(np.mean(psnrs)), 2), round(float(np.mean(ssims)), 4)


@torch.no_grad()
def evaluate_model_classification(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate classification probabilities across `val/`."""
    model.eval()
    y_true_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []
    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].numpy()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x)
            probs = torch.sigmoid(logits).float().squeeze(-1).cpu().numpy()
        y_true_list.append(y)
        y_prob_list.append(probs)

    return np.concatenate(y_true_list, axis=0), np.concatenate(y_prob_list, axis=0)


def train_1d_bottleneck_baseline(
    train_cache: RAMCachedSplit,
    val_cache: RAMCachedSplit,
    pos_weight: float,
    ae_epochs: int,
    cls_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_bf16: bool,
    ckpt_dir: Path,
    seed: int = 42,
) -> Tuple[Dict[str, Any], VectorAutoencoder1D]:
    """
    Train the 1D Vector Bottleneck baseline (`z in R^64`) to demonstrate why flattening
    chest radiographs into a 1D vector inflates parameters and degrades reconstruction & F1.
    """
    set_deterministic_seed(seed)
    t0 = time.time()

    ae_ds = PairedEnhancementAEDataset(train_cache, target_enhanced=False, augment=True)
    ae_loader = DataLoader(ae_ds, batch_size=batch_size, shuffle=True)
    train_ds = InMemoryChestXRayDataset(train_cache, preprocess_mode="raw", augment=True)
    val_ds = InMemoryChestXRayDataset(val_cache, preprocess_mode="raw", augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VectorAutoencoder1D(in_channels=1, base_channels=12, vec_dim=64).to(device)
    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    # Stage 1: AE Pretraining
    opt_ae = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(ae_epochs):
        model.train()
        for x_in, x_tgt in ae_loader:
            x_in = x_in.to(device, non_blocking=True)
            x_tgt = x_tgt.to(device, non_blocking=True)
            opt_ae.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                x_rec, _ = model.forward_autoencoder(x_in)
                loss = 0.5 * F.mse_loss(x_rec, x_tgt) + 0.5 * F.l1_loss(x_rec, x_tgt)
            loss.backward()
            opt_ae.step()

    psnr_1d, ssim_1d = evaluate_ae_reconstruction(model, val_loader, device, use_bf16, is_1d=True)

    # Stage 2: Joint Classification + Reconstruction
    pos_w = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cls_epochs, eta_min=lr * 0.05)

    best_score = -1.0
    best_state = None
    best_epoch = 1

    for epoch in range(1, cls_epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                z_vec = model.encode(x)
                logits = model.cls_head(z_vec)
                x_rec = model.decode(z_vec)
                cls_loss = criterion(logits, y)
                rec_loss = 0.5 * F.mse_loss(x_rec, x) + 0.5 * F.l1_loss(x_rec, x)
                loss = cls_loss + 0.25 * rec_loss
            loss.backward()
            optimizer.step()
        scheduler.step()

        y_true, y_prob = evaluate_model_classification(model, val_loader, device, use_bf16)
        _, m_cal = find_optimal_threshold(y_true, y_prob, criterion="f1")
        score = 0.40 * float(m_cal["f1_score"]) + 0.35 * float(m_cal["balanced_accuracy"]) + 0.25 * float(m_cal["auroc"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    psnr_1d, ssim_1d = evaluate_ae_reconstruction(model, val_loader, device, use_bf16, is_1d=True)
    y_true, y_prob = evaluate_model_classification(model, val_loader, device, use_bf16)
    opt_t, m_cal = find_optimal_threshold(y_true, y_prob, criterion="f1")

    eff = profile_model_efficiency(model, input_shape=(1, 1, 256, 256), device=device)
    inf_params = (
        sum(p.numel() for p in model.spatial_enc.parameters())
        + sum(p.numel() for p in model.to_vec.parameters())
        + sum(p.numel() for p in model.cls_head.parameters())
    )
    elapsed = round(time.time() - t0, 2)

    ckpt_path = ckpt_dir / "stage2_ae_1d_vec64_best.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    print(
        f"  [{'ae_1d_vec64':<24}] BestEp {best_epoch:02d}/{cls_epochs:02d} ({elapsed:>5.1f}s) | "
        f"InfParams: {inf_params:>7,} | PSNR: {psnr_1d:>5.2f} dB | SSIM: {ssim_1d:.3f} | "
        f"F1: {m_cal['f1_score']:>6.2f}% | BalAcc: {m_cal['balanced_accuracy']:>6.2f}% | AUROC: {m_cal['auroc']:>6.2f}%"
    )

    rec = {
        "run_name": "ae_1d_vec64",
        "short_label": "1D Vec (R^64)",
        "bottleneck_shape": "64 (1D)",
        "coupling_mode": "joint_multitask",
        "inference_params": inf_params,
        "macs_m": eff["macs_m"],
        "ae_psnr_db": psnr_1d,
        "ae_ssim": ssim_1d,
        "optimal_threshold": opt_t,
        "calibrated_metrics": m_cal,
        "time_sec": elapsed,
        "checkpoint_path": str(ckpt_path),
    }
    return rec, model


def train_2d_spatial_condition(
    run_name: str,
    short_label: str,
    downsample_factor: int,
    latent_channels: int,
    coupling_mode: str,
    train_cache: RAMCachedSplit,
    val_cache: RAMCachedSplit,
    unlabeled_test_cache: Optional[RAMCachedSplit],
    pos_weight: float,
    ae_epochs: int,
    cls_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_bf16: bool,
    ckpt_dir: Path,
    seed: int = 42,
) -> Tuple[Dict[str, Any], UnifiedTGSECAE]:
    """
    Train `UnifiedTGSECAE` under the specified 2D spatial bottleneck geometry and
    AE-Classifier coupling regime.
    """
    set_deterministic_seed(seed)
    t0 = time.time()

    train_ds = InMemoryChestXRayDataset(train_cache, preprocess_mode="raw", augment=True)
    val_ds = InMemoryChestXRayDataset(val_cache, preprocess_mode="raw", augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UnifiedTGSECAE(
        in_channels=1,
        enhancement_mode="task_guided_adaptive",
        base_channels=12,
        latent_channels=latent_channels,
        downsample_factor=downsample_factor,
        cls_stage_channels=(24, 36, 48),
        attention_mode="none",
        dropout=0.2,
    ).to(device)

    # 1. Optional Stage 1 Unsupervised Autoencoder Pretraining
    if coupling_mode in ("two_stage_frozen", "two_stage_finetune", "enhance_ae_finetune", "transductive_ae_joint"):
        if coupling_mode == "transductive_ae_joint" and unlabeled_test_cache is not None:
            combined_imgs = np.concatenate([train_cache.images, unlabeled_test_cache.images], axis=0)
            combined_lbls = np.concatenate([train_cache.labels, unlabeled_test_cache.labels], axis=0)
            combined_ids = train_cache.image_ids + unlabeled_test_cache.image_ids
            ae_split = RAMCachedSplit(images=combined_imgs, labels=combined_lbls, image_ids=combined_ids)
        else:
            ae_split = train_cache

        target_enh = (coupling_mode == "enhance_ae_finetune")
        ae_ds = PairedEnhancementAEDataset(ae_split, target_enhanced=target_enh, alpha_target=0.50, augment=True)
        ae_loader = DataLoader(ae_ds, batch_size=batch_size, shuffle=True)
        pretrain_autoencoder_2d(model, ae_loader, val_loader, epochs=ae_epochs, lr=lr, device=device, use_bf16=use_bf16)

    # 2. Configure Freeze / Fine-Tune / Multi-Task for Stage 2 Classification
    if coupling_mode == "two_stage_frozen":
        model.set_encoder_trainable(False)
        use_joint_recon = False
    elif coupling_mode in ("two_stage_finetune", "enhance_ae_finetune"):
        model.set_encoder_trainable(True)
        use_joint_recon = False
    else:
        # 'joint_multitask' or 'transductive_ae_joint'
        model.set_encoder_trainable(True)
        use_joint_recon = True

    pos_w = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    enh_params = [p for p in model.enhancer.parameters() if p.requires_grad]
    enh_ids = {id(p) for p in enh_params}
    other_trainable = [p for p in model.parameters() if p.requires_grad and id(p) not in enh_ids]

    param_groups = [{"params": other_trainable, "lr": lr, "weight_decay": 1e-4}]
    if enh_params:
        param_groups.append({"params": enh_params, "lr": lr * 3.0, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cls_epochs, eta_min=lr * 0.05)

    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    best_score = -1.0
    best_state = None
    best_epoch = 1

    for epoch in range(1, cls_epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model.forward_full(x, compute_reconstruction=use_joint_recon)
                logits = out["logits"]
                assert logits is not None
                cls_loss = criterion(logits, y)
                loss = cls_loss + 0.01 * out["enh_reg_loss"]

                if use_joint_recon and out["x_reconstructed"] is not None:
                    x_rec = out["x_reconstructed"]
                    x_enh = out["x_enhanced"]
                    rec_loss = 0.5 * F.mse_loss(x_rec, x_enh.detach()) + 0.5 * F.l1_loss(x_rec, x_enh.detach())
                    loss = loss + 0.25 * rec_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        y_true, y_prob = evaluate_model_classification(model, val_loader, device, use_bf16)
        _, m_cal = find_optimal_threshold(y_true, y_prob, criterion="f1")
        sel_score = (
            0.40 * float(m_cal["f1_score"])
            + 0.35 * float(m_cal["balanced_accuracy"])
            + 0.25 * float(m_cal["auroc"])
        )
        if sel_score > best_score:
            best_score = sel_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # If the decoder wasn't trained during joint_multitask (e.g. if coupling_mode started from scratch without AE),
    # ensure AE reconstruction metrics are evaluated properly
    psnr_db, ssim_val = evaluate_ae_reconstruction(model, val_loader, device, use_bf16, is_1d=False)
    y_true, y_prob = evaluate_model_classification(model, val_loader, device, use_bf16)
    opt_t, m_cal = find_optimal_threshold(y_true, y_prob, criterion="f1")

    eff = profile_model_efficiency(model, input_shape=(1, 1, 256, 256), device=device)
    param_bd = model.inference_parameter_breakdown()
    elapsed = round(time.time() - t0, 2)

    ckpt_path = ckpt_dir / f"stage2_{run_name}_best.pt"
    torch.save(
        {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "downsample_factor": downsample_factor,
            "latent_channels": latent_channels,
            "coupling_mode": coupling_mode,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    h_lat = 256 // downsample_factor
    b_shape = f"{latent_channels}×{h_lat}×{h_lat}"

    print(
        f"  [{run_name:<24}] BestEp {best_epoch:02d}/{cls_epochs:02d} ({elapsed:>5.1f}s) | "
        f"InfParams: {param_bd['inference_total_params']:>7,} | PSNR: {psnr_db:>5.2f} dB | SSIM: {ssim_val:.3f} | "
        f"F1: {m_cal['f1_score']:>6.2f}% | BalAcc: {m_cal['balanced_accuracy']:>6.2f}% | AUROC: {m_cal['auroc']:>6.2f}%"
    )

    rec = {
        "run_name": run_name,
        "short_label": short_label,
        "bottleneck_shape": b_shape,
        "downsample_factor": downsample_factor,
        "latent_channels": latent_channels,
        "coupling_mode": coupling_mode,
        "parameter_breakdown": param_bd,
        "inference_params": param_bd["inference_total_params"],
        "macs_m": eff["macs_m"],
        "ae_psnr_db": psnr_db,
        "ae_ssim": ssim_val,
        "optimal_threshold": opt_t,
        "calibrated_metrics": m_cal,
        "time_sec": elapsed,
        "checkpoint_path": str(ckpt_path),
    }
    return rec, model


def format_stage2_markdown(
    bottleneck_records: List[Dict[str, Any]],
    coupling_records: List[Dict[str, Any]],
) -> str:
    """Format publication Markdown tables for Stage 2 (Table 2A and Table 2B)."""
    lines = [
        "# Stage 2 (Exp 2): 2D Spatial Autoencoder Compression & Coupling Regimes",
        "",
        "## Table 2A: Spatial Bottleneck Geometry vs. 1D Vector Bottleneck (`joint_multitask`)",
        "",
        "| Model / Bottleneck | Latent Shape | Inference Params | Inference MACs (M) | Recon PSNR (dB) | Recon SSIM | **Balanced Acc (%)** | Sens (%) | **Spec (%)** | **Val F1 (%)** | **AUROC (%)** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in bottleneck_records:
        m = r["calibrated_metrics"]
        lines.append(
            f"| `{r['run_name']}` | `{r['bottleneck_shape']}` | {r['inference_params']:,} | {r['macs_m']:.2f} | "
            f"{r['ae_psnr_db']:.2f} | {r['ae_ssim']:.3f} | **{m['balanced_accuracy']:.2f}** | "
            f"{m['sensitivity_recall']:.2f} | **{m['specificity']:.2f}** | **{m['f1_score']:.2f}** | **{m['auroc']:.2f}** |"
        )

    lines.extend(
        [
            "",
            "## Table 2B: Autoencoder–Classifier Coupling Regimes (`16×32×32` 2D Spatial Bottleneck + `TGAE`)",
            "",
            "| Coupling Regime | Description | Inference Params | Recon PSNR (dB) | **Balanced Acc (%)** | Sens (%) | **Spec (%)** | **Val F1 (%)** | **AUROC (%)** |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in coupling_records:
        m = r["calibrated_metrics"]
        lines.append(
            f"| `{r['run_name']}` | {r['short_label']} | {r['inference_params']:,} | {r['ae_psnr_db']:.2f} | "
            f"**{m['balanced_accuracy']:.2f}** | {m['sensitivity_recall']:.2f} | **{m['specificity']:.2f}** | "
            f"**{m['f1_score']:.2f}** | **{m['auroc']:.2f}** |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 (Exp 2): 2D Spatial Autoencoder Bottleneck & Coupling Regime Ablation"
    )
    parser.add_argument("--dataset-dir", type=str, default="./DATASET", help="Path to DATASET/ directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--ae-epochs", type=int, default=15, help="Epochs for Stage 1 AE pretraining")
    parser.add_argument("--cls-epochs", type=int, default=25, help="Epochs for Stage 2 classification")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for L40S")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Base learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 2-epoch smoke test first")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    fig_dir = out_dir / "figures"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env_info = inspect_environment(str(out_dir))
    device = torch.device("cuda" if env_info["cuda_available"] else "cpu")
    use_bf16 = bool(env_info["bf16_supported"])

    print("=" * 105)
    print("PneuVision — Stage 2 (Exp 2): 2D Spatial Autoencoder Bottleneck & Coupling Regime Study")
    print(f"Device: {env_info['gpu_name'] or 'CPU'} | BF16: {use_bf16} | Free VRAM: {env_info['vram_free_gib']} GiB")
    print("=" * 105)

    if args.smoke_test:
        print("\n[Smoke Test] Loading 128-image balanced subsample from DATASET...")
        train_sub = load_split_into_ram(args.dataset_dir, split="train", image_size=256, max_samples=128)
        val_sub = load_split_into_ram(args.dataset_dir, split="val", image_size=256, max_samples=64)
        rec, model = train_2d_spatial_condition(
            run_name="smoke_test_2d_ae",
            short_label="Smoke Test 32x32",
            downsample_factor=8,
            latent_channels=16,
            coupling_mode="two_stage_finetune",
            train_cache=train_sub,
            val_cache=val_sub,
            unlabeled_test_cache=None,
            pos_weight=0.3704,
            ae_epochs=2,
            cls_epochs=2,
            batch_size=32,
            lr=1.5e-3,
            device=device,
            use_bf16=use_bf16,
            ckpt_dir=ckpt_dir,
            seed=args.seed,
        )
        print(f"\n[Smoke Test PASSED] Inference Params: {rec['inference_params']:,} | Recon PSNR: {rec['ae_psnr_db']} dB")
        return

    # 1. Load train, val, and unlabeled test splits into RAM once
    t0 = time.time()
    print("\n[1/4] Decoding & caching train/, val/, and unlabeled test/ radiographs in RAM...")
    train_cache = load_split_into_ram(args.dataset_dir, split="train", image_size=256)
    val_cache = load_split_into_ram(args.dataset_dir, split="val", image_size=256)
    test_cache = load_split_into_ram(args.dataset_dir, split="test", image_size=256)
    n_norm = int(np.sum(train_cache.labels == 0))
    n_pneu = int(np.sum(train_cache.labels == 1))
    pos_weight = round(n_norm / max(n_pneu, 1), 4)
    print(
        f"  - Cached {len(train_cache.labels)} train, {len(val_cache.labels)} val, and "
        f"{len(test_cache.labels)} unlabeled test images in {time.time() - t0:.2f}s"
    )

    # 2. Exp 2A: Bottleneck Geometry Ablation (1D Vec-64 vs 2D 16x16 vs 2D 32x32 vs 2D 64x64)
    print("\n[2/4] Running Exp 2A — 1D Vector Bottleneck vs. 2D Spatial Bottleneck Resolutions:")
    bottleneck_records: List[Dict[str, Any]] = []

    rec_1d, model_1d = train_1d_bottleneck_baseline(
        train_cache=train_cache,
        val_cache=val_cache,
        pos_weight=pos_weight,
        ae_epochs=args.ae_epochs,
        cls_epochs=args.cls_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
    )
    bottleneck_records.append(rec_1d)

    rec_16, model_16 = train_2d_spatial_condition(
        run_name="ae_2d_16x16_c16",
        short_label="2D (16×16)",
        downsample_factor=16,
        latent_channels=16,
        coupling_mode="joint_multitask",
        train_cache=train_cache,
        val_cache=val_cache,
        unlabeled_test_cache=None,
        pos_weight=pos_weight,
        ae_epochs=args.ae_epochs,
        cls_epochs=args.cls_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
    )
    bottleneck_records.append(rec_16)

    rec_32_joint, model_32_joint = train_2d_spatial_condition(
        run_name="ae_2d_32x32_c16",
        short_label="2D (32×32)",
        downsample_factor=8,
        latent_channels=16,
        coupling_mode="joint_multitask",
        train_cache=train_cache,
        val_cache=val_cache,
        unlabeled_test_cache=None,
        pos_weight=pos_weight,
        ae_epochs=args.ae_epochs,
        cls_epochs=args.cls_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
    )
    bottleneck_records.append(rec_32_joint)

    rec_64, model_64 = train_2d_spatial_condition(
        run_name="ae_2d_64x64_c16",
        short_label="2D (64×64)",
        downsample_factor=4,
        latent_channels=16,
        coupling_mode="joint_multitask",
        train_cache=train_cache,
        val_cache=val_cache,
        unlabeled_test_cache=None,
        pos_weight=pos_weight,
        ae_epochs=args.ae_epochs,
        cls_epochs=args.cls_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
    )
    bottleneck_records.append(rec_64)

    # 3. Exp 2B: Autoencoder-Classifier Coupling Regimes on the 16x32x32 Bottleneck
    print("\n[3/4] Running Exp 2B — Autoencoder–Classifier Coupling Regimes (on 16×32×32 Spatial Grid):")
    coupling_records: List[Dict[str, Any]] = []

    c_specs = [
        ("1_two_stage_frozen", "Two-Stage Frozen", "two_stage_frozen"),
        ("2_two_stage_finetune", "Two-Stage Fine-Tune", "two_stage_finetune"),
        ("3_enhance_ae_finetune", "Enhance-AE Fine-Tune", "enhance_ae_finetune"),
    ]
    for r_name, s_lbl, c_mode in c_specs:
        rec_c, _ = train_2d_spatial_condition(
            run_name=r_name,
            short_label=s_lbl,
            downsample_factor=8,
            latent_channels=16,
            coupling_mode=c_mode,
            train_cache=train_cache,
            val_cache=val_cache,
            unlabeled_test_cache=None,
            pos_weight=pos_weight,
            ae_epochs=args.ae_epochs,
            cls_epochs=args.cls_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            use_bf16=use_bf16,
            ckpt_dir=ckpt_dir,
            seed=args.seed,
        )
        coupling_records.append(rec_c)

    # Reuse the 32x32 Joint Multi-Task record as Regime 4
    rec_joint_copy = dict(rec_32_joint)
    rec_joint_copy["run_name"] = "4_joint_multitask"
    rec_joint_copy["short_label"] = "Joint Multi-Task"
    coupling_records.append(rec_joint_copy)

    # Regime 5: Transductive Unsupervised AE Pretraining (train + unlabeled test) + Joint Multi-Task
    rec_trans, _ = train_2d_spatial_condition(
        run_name="5_transductive_ae_joint",
        short_label="Transductive AE + Joint",
        downsample_factor=8,
        latent_channels=16,
        coupling_mode="transductive_ae_joint",
        train_cache=train_cache,
        val_cache=val_cache,
        unlabeled_test_cache=test_cache,
        pos_weight=pos_weight,
        ae_epochs=args.ae_epochs,
        cls_epochs=args.cls_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
    )
    coupling_records.append(rec_trans)

    # 4. Generate Stage 2 Publication Figures & Markdown Report
    print("\n[4/4] Generating Stage 2 Publication Figures & Markdown Summary:")
    pneu_idx = int(np.where(val_cache.labels == 1)[0][0])
    sample_raw = val_cache.images[pneu_idx]
    x_sample = torch.from_numpy(sample_raw[None, None, :, :]).to(device)

    with torch.no_grad():
        rec1d_t, _ = model_1d.forward_autoencoder(x_sample)
        rec16_t, _, _ = model_16.forward_autoencoder(x_sample)
        rec32_t, z32_t, _ = model_32_joint.forward_autoencoder(x_sample)
        rec64_t, _, _ = model_64.forward_autoencoder(x_sample)

        rec_1d_np = rec1d_t.squeeze().cpu().numpy()
        rec_16_np = rec16_t.squeeze().cpu().numpy()
        rec_32_np = rec32_t.squeeze().cpu().numpy()
        rec_64_np = rec64_t.squeeze().cpu().numpy()
        lat_32_np = z32_t.squeeze(0).mean(dim=0).cpu().numpy()

    fig_recon = plot_stage2_ae_reconstructions(
        sample_raw=sample_raw,
        rec_1d=rec_1d_np,
        metrics_1d=(rec_1d["ae_psnr_db"], rec_1d["ae_ssim"]),
        rec_16=rec_16_np,
        metrics_16=(rec_16["ae_psnr_db"], rec_16["ae_ssim"]),
        rec_32=rec_32_np,
        metrics_32=(rec_32_joint["ae_psnr_db"], rec_32_joint["ae_ssim"]),
        rec_64=rec_64_np,
        metrics_64=(rec_64["ae_psnr_db"], rec_64["ae_ssim"]),
        latent_32_grid=lat_32_np,
        out_path=fig_dir / "stage2_ae_reconstructions.png",
    )
    print(f"  - Saved AE reconstruction figure -> {fig_recon}")

    fig_coup = plot_stage2_coupling_comparison(
        bottleneck_records=bottleneck_records,
        coupling_records=coupling_records,
        out_path=fig_dir / "stage2_coupling_comparison.png",
    )
    print(f"  - Saved Coupling comparison plot -> {fig_coup}")

    md_text = format_stage2_markdown(bottleneck_records, coupling_records)
    md_path = out_dir / "stage2_summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    json_path = out_dir / "stage2_autoencoder_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "environment": env_info,
                "config": vars(args),
                "bottleneck_records": bottleneck_records,
                "coupling_records": coupling_records,
            },
            f,
            indent=2,
        )

    print(f"  - Saved Markdown tables          -> {md_path}")
    print(f"  - Saved JSON results             -> {json_path}")
    print("\n" + md_text)
    print("=" * 105)


if __name__ == "__main__":
    main()
