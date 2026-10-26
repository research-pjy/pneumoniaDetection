#!/usr/bin/env python3
"""
Stage 3 (Exp 3): Channel/Spatial Attention Ablation (`SE`, `ECA`, `CBAM`) and
Parameter-Budget Pareto Frontier Study (`TG-SECAE` Nano/Micro/Tiny/Small vs.
`ShuffleNetV2_x0.5`, `SqueezeNet1_1`, `MobileNetV2`, and `ResNet-18`).

Investigates two publication questions (Idea 2 & Overall Pareto Efficiency):
  - Exp 3A (Table 3A): How much diagnostic accuracy and specificity gain do ultra-lightweight
    channel attention blocks (`ECA` [+18 params], `SE` [r=8,4,2], `CBAM`) unlock on top of
    our `16×32×32` 2D spatial latent bottleneck (`transductive_ae_joint`)?
  - Exp 3B (Table 3B & Figure 3): Scaling `TG-SECAE` across 4 strict parameter budgets
    (`Nano` ~4k, `Micro` ~10k, `Tiny` ~35k, `Small` ~130k) against standard vision backbones
    spanning 340k to 11.17M parameters.

Usage on `rama` (L40S):
    # Step 3.1 — Mandatory 2-epoch smoke test first:
    python scripts/03_exp3_attention_pareto.py --dataset-dir ./DATASET --output-dir ./outputs --smoke-test

    # Step 3.2 — Full Stage 3 Publication Study:
    python scripts/03_exp3_attention_pareto.py \
        --dataset-dir ./DATASET \
        --output-dir ./outputs \
        --ae-epochs 12 \
        --cls-epochs 25 \
        --batch-size 64
"""

import argparse
import json
from pathlib import Path
import sys
import time
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
    plot_stage3_attention_ablation,
    plot_stage3_pareto_frontier,
)


@torch.no_grad()
def evaluate_classifier_probs(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(y_true, y_prob)` over `val/`."""
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


@torch.no_grad()
def evaluate_ae_psnr_ssim(
    model: UnifiedTGSECAE,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> Tuple[float, float]:
    """Compute average `(PSNR_dB, SSIM)` for a `UnifiedTGSECAE` model across `val/`."""
    model.eval()
    psnrs, ssims = [], []
    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            x_rec, _, _ = model.forward_autoencoder(x)
        p, s = compute_psnr_ssim(x_rec.float(), x.float())
        psnrs.append(p)
        ssims.append(s)
    return round(float(np.mean(psnrs)), 2), round(float(np.mean(ssims)), 4)


def train_tgsecae_model(
    run_name: str,
    short_label: str,
    tier: str,
    attention_mode: str,
    se_reduction: int,
    train_cache: RAMCachedSplit,
    val_cache: RAMCachedSplit,
    test_cache: RAMCachedSplit,
    pos_weight: float,
    ae_epochs: int,
    cls_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_bf16: bool,
    ckpt_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a `UnifiedTGSECAE` model using our best Stage 2 coupling strategy
    (`transductive_ae_joint`: Unsupervised AE pretraining on `train + unlabeled test`
    followed by Joint Multi-Task fine-tuning).
    """
    set_deterministic_seed(seed)
    t0 = time.time()

    # Build combined unsupervised split (train + unlabeled test) for Stage 1 AE pretraining
    combined_imgs = np.concatenate([train_cache.images, test_cache.images], axis=0)
    combined_lbls = np.concatenate([train_cache.labels, test_cache.labels], axis=0)
    combined_ids = train_cache.image_ids + test_cache.image_ids
    ae_split = RAMCachedSplit(images=combined_imgs, labels=combined_lbls, image_ids=combined_ids)

    ae_ds = InMemoryChestXRayDataset(ae_split, preprocess_mode="raw", augment=True)
    train_ds = InMemoryChestXRayDataset(train_cache, preprocess_mode="raw", augment=True)
    val_ds = InMemoryChestXRayDataset(val_cache, preprocess_mode="raw", augment=False)

    ae_loader = DataLoader(ae_ds, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_tgsecae_tier(
        tier=tier,
        attention_mode=attention_mode,
        se_reduction=se_reduction,
        enhancement_mode="task_guided_adaptive",
        dropout=0.2,
    ).to(device)

    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    # Phase 1: Transductive Unsupervised AE Pretraining
    opt_ae = torch.optim.AdamW(
        list(model.enhancer.parameters())
        + list(model.encoder.parameters())
        + list(model.decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    sched_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=max(ae_epochs, 1), eta_min=lr * 0.1)
    for _ in range(ae_epochs):
        model.train()
        for batch in ae_loader:
            x = batch["image"].to(device, non_blocking=True)
            opt_ae.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                x_rec, _, reg_loss = model.forward_autoencoder(x)
                rec_loss = 0.5 * F.mse_loss(x_rec, x) + 0.5 * F.l1_loss(x_rec, x)
                loss = rec_loss + 0.005 * reg_loss
            loss.backward()
            opt_ae.step()
        sched_ae.step()

    # Phase 2: Joint Multi-Task Classification + Reconstruction Regularization
    pos_w = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    enh_params = list(model.enhancer.parameters())
    enh_ids = {id(p) for p in enh_params}
    other_params = [p for p in model.parameters() if id(p) not in enh_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": lr, "weight_decay": 1e-4},
            {"params": enh_params, "lr": lr * 3.0, "weight_decay": 0.0},
        ]
    )
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
                out = model.forward_full(x, compute_reconstruction=True)
                logits = out["logits"]
                x_rec = out["x_reconstructed"]
                x_enh = out["x_enhanced"]
                assert logits is not None and x_rec is not None and x_enh is not None
                cls_loss = criterion(logits, y)
                rec_loss = 0.5 * F.mse_loss(x_rec, x_enh.detach()) + 0.5 * F.l1_loss(x_rec, x_enh.detach())
                loss = cls_loss + 0.25 * rec_loss + 0.01 * out["enh_reg_loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()

        y_true, y_prob = evaluate_classifier_probs(model, val_loader, device, use_bf16)
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

    psnr_db, ssim_val = evaluate_ae_psnr_ssim(model, val_loader, device, use_bf16)
    y_true, y_prob = evaluate_classifier_probs(model, val_loader, device, use_bf16)
    opt_t, m_cal = find_optimal_threshold(y_true, y_prob, criterion="f1")
    m_def = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    eff = profile_model_efficiency(model, input_shape=(1, 1, 256, 256), device=device)
    param_bd = model.inference_parameter_breakdown()
    inf_params = param_bd["inference_total_params"]
    size_fp32_kb = round((inf_params * 4) / 1024.0, 1)
    elapsed = round(time.time() - t0, 2)

    ckpt_path = ckpt_dir / f"stage3_{run_name}_best.pt"
    torch.save(
        {
            "run_name": run_name,
            "tier": tier,
            "attention_mode": attention_mode,
            "se_reduction": se_reduction,
            "optimal_threshold": opt_t,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    print(
        f"  [{run_name:<24}] BestEp {best_epoch:02d}/{cls_epochs:02d} ({elapsed:>5.1f}s) | "
        f"InfParams: {inf_params:>8,} | MACs: {eff['macs_m']:>6.2f}M | "
        f"F1: {m_cal['f1_score']:>6.2f}% | BalAcc: {m_cal['balanced_accuracy']:>6.2f}% | "
        f"Sens: {m_cal['sensitivity_recall']:>6.2f}% | Spec: {m_cal['specificity']:>6.2f}% | AUROC: {m_cal['auroc']:>6.2f}%"
    )

    return {
        "run_name": run_name,
        "short_label": short_label,
        "family": "TG-SECAE (Ours)",
        "tier": tier,
        "attention_mode": attention_mode,
        "se_reduction": se_reduction,
        "inference_params": inf_params,
        "size_fp32_kb": size_fp32_kb,
        "macs_m": eff["macs_m"],
        "ae_psnr_db": psnr_db,
        "ae_ssim": ssim_val,
        "optimal_threshold": opt_t,
        "default_metrics": m_def,
        "calibrated_metrics": m_cal,
        "time_sec": elapsed,
        "checkpoint_path": str(ckpt_path),
    }


def train_standard_baseline(
    arch: str,
    short_label: str,
    train_cache: RAMCachedSplit,
    val_cache: RAMCachedSplit,
    pos_weight: float,
    cls_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_bf16: bool,
    ckpt_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a standard `torchvision` vision backbone (`shufflenet_v2_x0_5`, `squeezenet1_1`,
    `mobilenet_v2`, `resnet18`) from scratch under identical training conditions.
    """
    set_deterministic_seed(seed)
    t0 = time.time()

    train_ds = InMemoryChestXRayDataset(train_cache, preprocess_mode="raw", augment=True)
    val_ds = InMemoryChestXRayDataset(val_cache, preprocess_mode="raw", augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TorchvisionGrayscaleBaseline(arch=arch).to(device)
    eff = profile_model_efficiency(model, input_shape=(1, 1, 256, 256), device=device)
    inf_params = eff["trainable_params"]
    size_fp32_kb = round((inf_params * 4) / 1024.0, 1)

    pos_w = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    base_lr = lr * 0.5 if arch in ("squeezenet1_1", "resnet18") else lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cls_epochs, eta_min=base_lr * 0.05)

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
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()

        y_true, y_prob = evaluate_classifier_probs(model, val_loader, device, use_bf16)
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

    y_true, y_prob = evaluate_classifier_probs(model, val_loader, device, use_bf16)
    opt_t, m_cal = find_optimal_threshold(y_true, y_prob, criterion="f1")
    m_def = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    elapsed = round(time.time() - t0, 2)

    ckpt_path = ckpt_dir / f"stage3_baseline_{arch}_best.pt"
    torch.save(
        {
            "arch": arch,
            "optimal_threshold": opt_t,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    print(
        f"  [{arch:<24}] BestEp {best_epoch:02d}/{cls_epochs:02d} ({elapsed:>5.1f}s) | "
        f"InfParams: {inf_params:>8,} | MACs: {eff['macs_m']:>6.2f}M | "
        f"F1: {m_cal['f1_score']:>6.2f}% | BalAcc: {m_cal['balanced_accuracy']:>6.2f}% | "
        f"Sens: {m_cal['sensitivity_recall']:>6.2f}% | Spec: {m_cal['specificity']:>6.2f}% | AUROC: {m_cal['auroc']:>6.2f}%"
    )

    return {
        "run_name": f"baseline_{arch}",
        "short_label": short_label,
        "family": "Standard Baselines",
        "arch": arch,
        "inference_params": inf_params,
        "size_fp32_kb": size_fp32_kb,
        "macs_m": eff["macs_m"],
        "optimal_threshold": opt_t,
        "default_metrics": m_def,
        "calibrated_metrics": m_cal,
        "time_sec": elapsed,
        "checkpoint_path": str(ckpt_path),
    }


def format_stage3_markdown(
    attention_records: List[Dict[str, Any]],
    pareto_our_records: List[Dict[str, Any]],
    baseline_records: List[Dict[str, Any]],
) -> str:
    """Format publication Markdown tables for Stage 3 (Table 3A and Table 3B)."""
    lines = [
        "# Stage 3 (Exp 3): Channel Attention Ablation & Parameter-Budget Pareto Frontier",
        "",
        "## Table 3A: Channel & Spatial Attention Ablation (`TG-SECAE-Micro`, `16×32×32` Latent Grid)",
        "",
        "| Attention Mechanism | Extra Params | Total Inf. Params | MACs (M) | **Balanced Acc (%)** | Sens / Recall (%) | **Spec / TNR (%)** | **Val F1 (%)** | **AUROC (%)** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    base_params = attention_records[0]["inference_params"]
    for r in attention_records:
        m = r["calibrated_metrics"]
        extra = int(r["inference_params"]) - int(base_params)
        lines.append(
            f"| `{r['run_name']}` ({r['short_label']}) | +{extra:,} | {r['inference_params']:,} | {r['macs_m']:.2f} | "
            f"**{m['balanced_accuracy']:.2f}** | {m['sensitivity_recall']:.2f} | **{m['specificity']:.2f}** | "
            f"**{m['f1_score']:.2f}** | **{m['auroc']:.2f}** |"
        )

    lines.extend(
        [
            "",
            "## Table 3B: Accuracy–Efficiency Pareto Frontier (`TG-SECAE` Tiers vs. Standard Vision Backbones)",
            "",
            "| Model Architecture | Family | Inference Params | Param Ratio vs. ResNet-18 | Size (FP32 KB) | MACs (M) | Val Acc (%) | **Balanced Acc (%)** | Sens (%) | **Spec (%)** | **Val F1 (%)** | **AUROC (%)** |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    resnet_params = max(
        [int(r["inference_params"]) for r in baseline_records if "resnet18" in r["run_name"]] + [11170000]
    )
    for r in pareto_our_records + baseline_records:
        m = r["calibrated_metrics"]
        ratio = resnet_params / max(int(r["inference_params"]), 1)
        lines.append(
            f"| **{r['short_label']}** (`{r['run_name']}`) | {r['family']} | **{r['inference_params']:,}** | "
            f"**{ratio:.1f}× smaller** | {r['size_fp32_kb']:,.1f} KB | {r['macs_m']:.2f} | {m['accuracy']:.2f} | "
            f"**{m['balanced_accuracy']:.2f}** | {m['sensitivity_recall']:.2f} | **{m['specificity']:.2f}** | "
            f"**{m['f1_score']:.2f}** | **{m['auroc']:.2f}** |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 (Exp 3): Channel Attention Ablation & Parameter-Budget Pareto Frontier"
    )
    parser.add_argument("--dataset-dir", type=str, default="./DATASET", help="Path to DATASET/ directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--ae-epochs", type=int, default=12, help="Unsupervised AE pretraining epochs")
    parser.add_argument("--cls-epochs", type=int, default=25, help="Classification training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for L40S")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Base learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 2-epoch smoke test across models")
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

    print("=" * 110)
    print("PneuVision — Stage 3 (Exp 3): Channel Attention Ablation & Parameter-Budget Pareto Frontier")
    print(f"Device: {env_info['gpu_name'] or 'CPU'} | BF16: {use_bf16} | Free VRAM: {env_info['vram_free_gib']} GiB")
    print("=" * 110)

    if args.smoke_test:
        print("\n[Smoke Test] Verifying SE/ECA/CBAM, all 4 TG-SECAE tiers, and Torchvision baselines on 128 images...")
        tr_sub = load_split_into_ram(args.dataset_dir, split="train", image_size=256, max_samples=128)
        va_sub = load_split_into_ram(args.dataset_dir, split="val", image_size=256, max_samples=64)
        te_sub = load_split_into_ram(args.dataset_dir, split="test", image_size=256, max_samples=64)
        _ = train_tgsecae_model(
            run_name="smoke_tgsecae_micro_se",
            short_label="TG-SECAE-Micro (SE)",
            tier="micro",
            attention_mode="se",
            se_reduction=4,
            train_cache=tr_sub,
            val_cache=va_sub,
            test_cache=te_sub,
            pos_weight=0.3704,
            ae_epochs=1,
            cls_epochs=2,
            batch_size=32,
            lr=1.5e-3,
            device=device,
            use_bf16=use_bf16,
            ckpt_dir=ckpt_dir,
            seed=args.seed,
        )
        _ = train_standard_baseline(
            arch="shufflenet_v2_x0_5",
            short_label="ShuffleNetV2-0.5x",
            train_cache=tr_sub,
            val_cache=va_sub,
            pos_weight=0.3704,
            cls_epochs=2,
            batch_size=32,
            lr=1.5e-3,
            device=device,
            use_bf16=use_bf16,
            ckpt_dir=ckpt_dir,
            seed=args.seed,
        )
        print("\n[Smoke Test PASSED] Ready for full Stage 3 Pareto study!")
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
    print(f"  - Cached splits in {time.time() - t0:.2f}s (pos_weight = {pos_weight})")

    # 2. Exp 3A: Channel & Spatial Attention Ablation on `TG-SECAE-Micro`
    print("\n[2/4] Running Exp 3A — Channel & Spatial Attention Ablation (`TG-SECAE-Micro` 16×32×32):")
    att_specs = [
        ("att_none", "No Attention", "none", 4),
        ("att_eca_k3", "ECA (1D-Conv k=3)", "eca", 4),
        ("att_se_r8", "SE (r=8)", "se", 8),
        ("att_se_r4", "SE (r=4)", "se", 4),
        ("att_se_r2", "SE (r=2)", "se", 2),
        ("att_cbam", "CBAM (Ch+Spatial)", "cbam", 4),
    ]
    attention_records: List[Dict[str, Any]] = []
    for r_name, s_lbl, a_mode, r_ratio in att_specs:
        rec = train_tgsecae_model(
            run_name=r_name,
            short_label=s_lbl,
            tier="micro",
            attention_mode=a_mode,
            se_reduction=r_ratio,
            train_cache=train_cache,
            val_cache=val_cache,
            test_cache=test_cache,
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
        attention_records.append(rec)

    # Pick the highest-scoring attention variant (among SE / ECA / CBAM) to scale across Nano/Micro/Tiny/Small
    att_candidates = [r for r in attention_records if r["attention_mode"] != "none"]
    best_att_rec = max(
        att_candidates,
        key=lambda r: (
            0.40 * float(r["calibrated_metrics"]["f1_score"])
            + 0.35 * float(r["calibrated_metrics"]["balanced_accuracy"])
            + 0.25 * float(r["calibrated_metrics"]["auroc"])
        ),
    )
    win_att_mode = best_att_rec["attention_mode"]
    win_se_r = int(best_att_rec["se_reduction"])
    print(
        f"\n  -> Best Attention Mechanism on Micro: {best_att_rec['short_label']} "
        f"(F1={best_att_rec['calibrated_metrics']['f1_score']:.2f}%, "
        f"BalAcc={best_att_rec['calibrated_metrics']['balanced_accuracy']:.2f}%, "
        f"AUROC={best_att_rec['calibrated_metrics']['auroc']:.2f}%)"
    )

    # 3. Exp 3B: Parameter-Budget Pareto Sweep (`TG-SECAE` Tiers vs. Standard Vision Backbones)
    print(f"\n[3/4] Running Exp 3B — Pareto Frontier (`TG-SECAE` Nano/Micro/Tiny/Small with {best_att_rec['short_label']} vs. Baselines):")
    pareto_our_records: List[Dict[str, Any]] = []

    for tier_name, tier_label in [
        ("nano", "TG-SECAE-Nano"),
        ("micro", "TG-SECAE-Micro"),
        ("tiny", "TG-SECAE-Tiny"),
        ("small", "TG-SECAE-Small"),
    ]:
        if tier_name == "micro":
            micro_copy = dict(best_att_rec)
            micro_copy["run_name"] = f"tgsecae_micro_{win_att_mode}"
            micro_copy["short_label"] = tier_label
            pareto_our_records.append(micro_copy)
        else:
            rec_tier = train_tgsecae_model(
                run_name=f"tgsecae_{tier_name}_{win_att_mode}",
                short_label=tier_label,
                tier=tier_name,
                attention_mode=win_att_mode,
                se_reduction=win_se_r,
                train_cache=train_cache,
                val_cache=val_cache,
                test_cache=test_cache,
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
            pareto_our_records.append(rec_tier)

    baseline_specs = [
        ("shufflenet_v2_x0_5", "ShuffleNetV2-0.5x"),
        ("squeezenet1_1", "SqueezeNet-1.1"),
        ("mobilenet_v2", "MobileNetV2"),
        ("resnet18", "ResNet-18"),
    ]
    baseline_records: List[Dict[str, Any]] = []
    for b_arch, b_label in baseline_specs:
        b_rec = train_standard_baseline(
            arch=b_arch,
            short_label=b_label,
            train_cache=train_cache,
            val_cache=val_cache,
            pos_weight=pos_weight,
            cls_epochs=args.cls_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            use_bf16=use_bf16,
            ckpt_dir=ckpt_dir,
            seed=args.seed,
        )
        baseline_records.append(b_rec)

    # 4. Generate Stage 3 Publication Figures & Summary Report
    print("\n[4/4] Generating Stage 3 Publication Figures & Markdown Summary:")
    fig_att = plot_stage3_attention_ablation(
        attention_records=attention_records,
        out_path=fig_dir / "stage3_attention_ablation.png",
    )
    print(f"  - Saved Attention Ablation figure -> {fig_att}")

    fig_pareto = plot_stage3_pareto_frontier(
        our_records=pareto_our_records,
        baseline_records=baseline_records,
        out_path=fig_dir / "stage3_pareto_frontier.png",
    )
    print(f"  - Saved Pareto Frontier figure    -> {fig_pareto}")

    md_text = format_stage3_markdown(attention_records, pareto_our_records, baseline_records)
    md_path = out_dir / "stage3_summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    json_path = out_dir / "stage3_attention_pareto_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "environment": env_info,
                "config": vars(args),
                "winning_attention_on_micro": {
                    "attention_mode": win_att_mode,
                    "se_reduction": win_se_r,
                    "short_label": best_att_rec["short_label"],
                },
                "attention_records": attention_records,
                "pareto_our_records": pareto_our_records,
                "baseline_records": baseline_records,
            },
            f,
            indent=2,
        )

    print(f"  - Saved Markdown tables           -> {md_path}")
    print(f"  - Saved JSON results              -> {json_path}")
    print("\n" + md_text)
    print("=" * 110)


if __name__ == "__main__":
    main()
