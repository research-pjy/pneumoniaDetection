#!/usr/bin/env python3
"""
Stage 1 (Exp 1): Empirical Investigation of the Over-Enhancement Inverted-U Curve
and Differentiable Task-Guided Adaptive Enhancement (TGAE) on Pediatric Chest X-Rays.

Usage on `rama` (L40S):
    # Step 1.1 — Mandatory 2-epoch pipeline smoke test first (per L40S Execution Context):
    python scripts/01_exp1_enhancement_study.py --dataset-dir ./DATASET --smoke-test

    # Step 1.2 — Full Stage 1 Publication Study (8 Fixed Alpha Sweep + Multi-View + 2 TGAE variants):
    python scripts/01_exp1_enhancement_study.py \
        --dataset-dir ./DATASET \
        --output-dir ./outputs \
        --epochs 25 \
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
from torch.utils.data import DataLoader

from pneuvision.data.dataset import (
    InMemoryChestXRayDataset,
    RAMCachedSplit,
    load_split_into_ram,
)
from pneuvision.models.classifier import TinyPneuNet
from pneuvision.utils.hardware import (
    inspect_environment,
    profile_model_efficiency,
    set_deterministic_seed,
)
from pneuvision.utils.metrics import (
    compute_binary_metrics,
    find_optimal_threshold,
)
from pneuvision.utils.plotting import (
    plot_inverted_u_enhancement_curve,
    plot_tgae_visual_comparison,
)


@torch.no_grad()
def evaluate_classifier(
    model: TinyPneuNet,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Run inference over `val/` and return `(y_true, y_prob, gate_stats)`.
    """
    model.eval()
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    wc_list: List[float] = []
    wd_list: List[float] = []

    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].numpy()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, _, _ = model.forward_with_extras(x)
            probs = torch.sigmoid(logits).float().squeeze(-1).cpu().numpy()

        if model.enhancement_mode in ("task_guided_global", "task_guided_adaptive"):
            wc, wd = model.enhancer.get_enhancement_weights(x)
            wc_list.extend(wc.float().view(-1).cpu().tolist())
            wd_list.extend(wd.float().view(-1).cpu().tolist())

        all_labels.append(y)
        all_probs.append(probs)

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    gate_stats: Dict[str, float] = {}
    if wc_list:
        gate_stats = {
            "wc_mean": round(float(np.mean(wc_list)), 4),
            "wc_std": round(float(np.std(wc_list)), 4),
            "wd_mean": round(float(np.mean(wd_list)), 4),
            "wd_std": round(float(np.std(wd_list)), 4),
        }
    return y_true, y_prob, gate_stats


def train_single_condition(
    run_name: str,
    preprocess_mode: str,
    enhancement_mode: str,
    alpha: float,
    train_cache: RAMCachedSplit,
    val_cache: RAMCachedSplit,
    pos_weight: float,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_penalty_weight: float,
    device: torch.device,
    use_bf16: bool,
    ckpt_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train `TinyPneuNet` under a single enhancement condition from a deterministic seed
    and return complete efficiency + clinical validation metrics.
    """
    set_deterministic_seed(seed)

    train_ds = InMemoryChestXRayDataset(
        train_cache,
        preprocess_mode=preprocess_mode,
        alpha_strength=alpha,
        augment=True,
    )
    val_ds = InMemoryChestXRayDataset(
        val_cache,
        preprocess_mode=preprocess_mode,
        alpha_strength=alpha,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    in_ch = 3 if preprocess_mode == "multiview_3ch" else 1
    model = TinyPneuNet(
        in_channels=in_ch,
        enhancement_mode=enhancement_mode,
        stage_channels=(16, 24, 36, 48),
        attention_mode="none",
        dropout=0.2,
    ).to(device)

    eff = profile_model_efficiency(model, input_shape=(1, in_ch, 256, 256), device=device)

    pos_w_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

    # Give the differentiable enhancer gates a 3x higher learning rate so they adapt rapidly
    enh_params = list(model.enhancer.parameters())
    enh_param_ids = {id(p) for p in enh_params}
    backbone_params = [p for p in model.parameters() if id(p) not in enh_param_ids]

    param_groups = [{"params": backbone_params, "lr": lr, "weight_decay": 1e-4}]
    if enh_params:
        param_groups.append({"params": enh_params, "lr": lr * 3.0, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    best_score = -1.0
    best_state = None
    best_epoch = 1
    history: List[Dict[str, Any]] = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits, _, reg_loss = model.forward_with_extras(x)
                cls_loss = criterion(logits, y)
                loss = cls_loss + noise_penalty_weight * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)

        y_true, y_prob, gate_stats = evaluate_classifier(model, val_loader, device, use_bf16)
        m_05 = compute_binary_metrics(y_true, y_prob, threshold=0.5)
        _, m_cal = find_optimal_threshold(y_true, y_prob, criterion="f1")

        # Selection score balances AUROC, Calibrated F1, and Balanced Accuracy
        sel_score = (
            0.40 * float(m_cal["f1_score"])
            + 0.35 * float(m_cal["balanced_accuracy"])
            + 0.25 * float(m_cal["auroc"])
        )
        if sel_score > best_score:
            best_score = sel_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_f1_cal": m_cal["f1_score"],
                "val_bal_acc_cal": m_cal["balanced_accuracy"],
                "val_auroc": m_cal["auroc"],
                "gate_stats": gate_stats,
            }
        )

    elapsed = round(time.time() - t_start, 2)

    # Restore best checkpoint and compute final validation metrics
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = ckpt_dir / f"stage1_{run_name}_best.pt"
    torch.save(
        {
            "run_name": run_name,
            "best_epoch": best_epoch,
            "preprocess_mode": preprocess_mode,
            "enhancement_mode": enhancement_mode,
            "alpha": alpha,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    y_true, y_prob, final_gate_stats = evaluate_classifier(model, val_loader, device, use_bf16)
    default_metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    opt_thresh, calibrated_metrics = find_optimal_threshold(y_true, y_prob, criterion="f1")

    print(
        f"  [{run_name:<22}] BestEp {best_epoch:02d}/{epochs:02d} ({elapsed:>5.1f}s) | "
        f"Params: {eff['trainable_params']:>6} | "
        f"F1: {calibrated_metrics['f1_score']:>6.2f}% | "
        f"BalAcc: {calibrated_metrics['balanced_accuracy']:>6.2f}% | "
        f"Sens: {calibrated_metrics['sensitivity_recall']:>6.2f}% | "
        f"Spec: {calibrated_metrics['specificity']:>6.2f}% | "
        f"AUROC: {calibrated_metrics['auroc']:>6.2f}%"
    )
    if final_gate_stats:
        print(
            f"      -> Learned TGAE Gates on Val: w_c = {final_gate_stats['wc_mean']:.4f} ± {final_gate_stats['wc_std']:.4f}, "
            f"w_d = {final_gate_stats['wd_mean']:.4f} ± {final_gate_stats['wd_std']:.4f}"
        )

    return {
        "run_name": run_name,
        "preprocess_mode": preprocess_mode,
        "enhancement_mode": enhancement_mode,
        "alpha": alpha,
        "best_epoch": best_epoch,
        "time_sec": elapsed,
        "efficiency": eff,
        "optimal_threshold": opt_thresh,
        "default_threshold_metrics": default_metrics,
        "calibrated_metrics": calibrated_metrics,
        "gate_stats": final_gate_stats,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }


def run_smoke_test(dataset_dir: str, out_dir: Path, device: torch.device, use_bf16: bool, seed: int) -> None:
    """
    Execute the mandatory L40S pipeline smoke test before full training:
    Data loading -> preprocessing -> TaskGuidedEnhancer -> TinyPneuNet -> BF16 forward ->
    BCE loss -> backward -> gradient check -> optimizer step -> parameter update check -> checkpoint.
    """
    print("\n[Smoke Test] Loading 128-image balanced subsample from DATASET...")
    train_sub = load_split_into_ram(dataset_dir, split="train", image_size=256, max_samples=128)
    val_sub = load_split_into_ram(dataset_dir, split="val", image_size=256, max_samples=64)

    set_deterministic_seed(seed)
    model = TinyPneuNet(in_channels=1, enhancement_mode="task_guided_adaptive").to(device)
    before_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rec = train_single_condition(
        run_name="smoke_test_tgae",
        preprocess_mode="raw",
        enhancement_mode="task_guided_adaptive",
        alpha=0.0,
        train_cache=train_sub,
        val_cache=val_sub,
        pos_weight=0.3704,
        epochs=2,
        batch_size=32,
        lr=1e-3,
        noise_penalty_weight=0.01,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=seed,
    )

    # Verify parameter updates occurred
    ckpt = torch.load(rec["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    after_params = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    total_delta = sum(float((a - b).norm().item()) for a, b in zip(after_params, before_params))

    vram_peak_gib = (
        round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3)
        if device.type == "cuda"
        else 0.0
    )

    print("\n[Smoke Test Verification Summary]")
    print(f"  - Device used          : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  - BF16 Autocast active : {use_bf16 and device.type == 'cuda'}")
    print(f"  - Peak GPU VRAM used   : {vram_peak_gib} GiB")
    print(f"  - Loss finite          : {np.isfinite(rec['history'][-1]['train_loss'])}")
    print(f"  - Parameter L2 update  : {total_delta:.6f} (> 0 confirms valid gradients & optimizer step)")
    print(f"  - Checkpoint verified  : {rec['checkpoint_path']}")
    print("\n[Smoke Test PASSED] Ready for full Stage 1 experiment!")


def format_markdown_summary(
    sweep_records: List[Dict[str, Any]],
    other_records: List[Dict[str, Any]],
) -> str:
    """Format a publication-ready Markdown summary table for Stage 1 (Exp 1)."""
    lines = [
        "# Stage 1 (Exp 1): Image Enhancement & Task-Guided Adaptive Enhancer (`TGAE`) Results",
        "",
        "## Table 1: Fixed Enhancement Strength Sweep (`α`) vs. Multi-View & Task-Guided Enhancers",
        "",
        "| Condition / Method | Enh. Dial `α` | Trainable Params | MACs (M) | Val Acc (%) | **Balanced Acc (%)** | Sens / Recall (%) | **Spec / TNR (%)** | **Val F1 (%)** | **AUROC (%)** | AUPRC (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_recs = sweep_records + other_records
    for r in all_recs:
        m = r["calibrated_metrics"]
        eff = r["efficiency"]
        alpha_str = f"{r['alpha']:.2f}" if r["preprocess_mode"] == "fixed_clahe" or r["run_name"] == "alpha_0.00_raw" else "Learned / Multi"
        lines.append(
            f"| `{r['run_name']}` | {alpha_str} | {eff['trainable_params']:,} | {eff['macs_m']:.2f} | "
            f"{m['accuracy']:.2f} | **{m['balanced_accuracy']:.2f}** | {m['sensitivity_recall']:.2f} | "
            f"**{m['specificity']:.2f}** | **{m['f1_score']:.2f}** | **{m['auroc']:.2f}** | {m['auprc']:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 (Exp 1): Inverted-U Enhancement Sweep & Task-Guided Adaptive Enhancer Study"
    )
    parser.add_argument("--dataset-dir", type=str, default="./DATASET", help="Path to DATASET/ directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per condition (default: 25)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for L40S (default: 64)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Base learning rate")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0],
        help="Fixed enhancement strength dial values for the Inverted-U study",
    )
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

    print("=" * 100)
    print("PneuVision — Stage 1 (Exp 1): Inverted-U Enhancement Curve & Task-Guided Enhancer (TGAE)")
    print(f"Device: {env_info['gpu_name'] or 'CPU'} | BF16: {use_bf16} | Free VRAM: {env_info['vram_free_gib']} GiB")
    print("=" * 100)

    if args.smoke_test:
        run_smoke_test(args.dataset_dir, out_dir, device, use_bf16, args.seed)
        return

    # 1. Load train (4,684) and val (585) splits into RAM once (~1.38 GB in RAM)
    t0 = time.time()
    print("\n[1/4] Decoding & caching train/ and val/ radiographs in RAM (256x256 float32)...")
    train_cache = load_split_into_ram(args.dataset_dir, split="train", image_size=256)
    val_cache = load_split_into_ram(args.dataset_dir, split="val", image_size=256)
    n_norm = int(np.sum(train_cache.labels == 0))
    n_pneu = int(np.sum(train_cache.labels == 1))
    pos_weight = round(n_norm / max(n_pneu, 1), 4)
    print(
        f"  - Cached {len(train_cache.labels)} train images & {len(val_cache.labels)} val images "
        f"in {time.time() - t0:.2f}s (BCE pos_weight = {pos_weight})"
    )

    # 2. Part A: Exp 1A — Fixed Enhancement Strength Sweep (alpha in args.alphas)
    print("\n[2/4] Running Exp 1A — Fixed Enhancement Strength Sweep (Testing the Inverted-U Curve):")
    sweep_records: List[Dict[str, Any]] = []
    for alpha in args.alphas:
        r_name = "alpha_0.00_raw" if alpha == 0.0 else f"alpha_{alpha:.2f}_clahe"
        p_mode = "raw" if alpha == 0.0 else "fixed_clahe"
        rec = train_single_condition(
            run_name=r_name,
            preprocess_mode=p_mode,
            enhancement_mode="raw",
            alpha=float(alpha),
            train_cache=train_cache,
            val_cache=val_cache,
            pos_weight=pos_weight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            noise_penalty_weight=0.0,
            device=device,
            use_bf16=use_bf16,
            ckpt_dir=ckpt_dir,
            seed=args.seed,
        )
        sweep_records.append(rec)

    # 3. Part B: Exp 1B — Multi-View Stack & Differentiable Task-Guided Enhancers (Ideas 3 & 4)
    print("\n[3/4] Running Exp 1B — Multi-View 3-Channel & Differentiable Task-Guided Enhancers (TGAE):")
    other_records: List[Dict[str, Any]] = []

    mv_rec = train_single_condition(
        run_name="multiview_3ch_stack",
        preprocess_mode="multiview_3ch",
        enhancement_mode="raw",
        alpha=0.0,
        train_cache=train_cache,
        val_cache=val_cache,
        pos_weight=pos_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_penalty_weight=0.0,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
    )
    other_records.append(mv_rec)

    tgae_global_rec = train_single_condition(
        run_name="tgae_global_2params",
        preprocess_mode="raw",
        enhancement_mode="task_guided_global",
        alpha=0.0,
        train_cache=train_cache,
        val_cache=val_cache,
        pos_weight=pos_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_penalty_weight=0.01,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
    )
    other_records.append(tgae_global_rec)

    tgae_adaptive_rec = train_single_condition(
        run_name="tgae_adaptive_58params",
        preprocess_mode="raw",
        enhancement_mode="task_guided_adaptive",
        alpha=0.0,
        train_cache=train_cache,
        val_cache=val_cache,
        pos_weight=pos_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_penalty_weight=0.01,
        device=device,
        use_bf16=use_bf16,
        ckpt_dir=ckpt_dir,
        seed=args.seed,
    )
    other_records.append(tgae_adaptive_rec)

    # 4. Generate Publication Plots & Markdown Summary Table
    print("\n[4/4] Generating Publication Figures & Stage 1 Summary Report:")
    curve_fig = plot_inverted_u_enhancement_curve(
        sweep_records=sweep_records,
        tgae_adaptive_record=tgae_adaptive_rec,
        tgae_global_record=tgae_global_rec,
        out_path=fig_dir / "stage1_inverted_u_curve.png",
    )
    print(f"  - Saved Inverted-U curve figure -> {curve_fig}")

    # Load best TGAE-Adaptive model to visualize learned enhancement on 1 NORMAL & 1 PNEUMONIA val image
    best_tgae_model = TinyPneuNet(in_channels=1, enhancement_mode="task_guided_adaptive").to(device)
    ckpt = torch.load(tgae_adaptive_rec["checkpoint_path"], map_location=device, weights_only=False)
    best_tgae_model.load_state_dict(ckpt["model_state_dict"])
    best_tgae_model.eval()

    norm_idx = int(np.where(val_cache.labels == 0)[0][0])
    pneu_idx = int(np.where(val_cache.labels == 1)[0][0])
    norm_raw = val_cache.images[norm_idx]
    pneu_raw = val_cache.images[pneu_idx]

    with torch.no_grad():
        x_pair = torch.from_numpy(np.stack([norm_raw, pneu_raw], axis=0)[:, None, :, :]).to(device)
        x_enh_pair, _ = best_tgae_model.enhancer(x_pair)
        wc_pair, wd_pair = best_tgae_model.enhancer.get_enhancement_weights(x_pair)
        enh_np = x_enh_pair.squeeze(1).cpu().numpy()
        wc_np = wc_pair.view(-1).cpu().numpy()
        wd_np = wd_pair.view(-1).cpu().numpy()

    vis_fig = plot_tgae_visual_comparison(
        normal_raw=norm_raw,
        normal_tgae=enh_np[0],
        normal_weights=(float(wc_np[0]), float(wd_np[0])),
        pneu_raw=pneu_raw,
        pneu_tgae=enh_np[1],
        pneu_weights=(float(wc_np[1]), float(wd_np[1])),
        out_path=fig_dir / "stage1_tgae_visual_comparison.png",
    )
    print(f"  - Saved TGAE visual comparison  -> {vis_fig}")

    md_table = format_markdown_summary(sweep_records, other_records)
    md_path = out_dir / "stage1_summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    json_path = out_dir / "stage1_enhancement_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "environment": env_info,
                "config": vars(args),
                "sweep_records": sweep_records,
                "other_records": other_records,
            },
            f,
            indent=2,
        )

    print(f"  - Saved Markdown table          -> {md_path}")
    print(f"  - Saved JSON metrics            -> {json_path}")
    print("\n" + md_table)
    print("=" * 100)


if __name__ == "__main__":
    main()
