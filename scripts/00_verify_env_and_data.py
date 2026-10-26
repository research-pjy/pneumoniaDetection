#!/usr/bin/env python3
"""
Stage 0: Verify L40S (`rama`) Environment, Audit Chest X-Ray Dataset, and Smoke-Test
the Differentiable Task-Guided Adaptive Enhancer (TGAE).

Usage on `rama` (L40S) or locally:
    # With real dataset directory (or --tar-path to auto-extract dataset.tar.gz):
    python scripts/00_verify_env_and_data.py --dataset-dir ./dataset --tar-path ~/datasets/dataset.tar.gz

    # Instant dry-run / smoke test (even before transferring dataset.tar.gz):
    python scripts/00_verify_env_and_data.py --dataset-dir ./dataset
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np
import torch
import torch.nn as nn

from pneuvision.data.dataset import (
    _list_images_in_dir,
    audit_dataset_directory,
    ensure_dataset_extracted,
    generate_synthetic_chest_xray,
)
from pneuvision.data.enhancement import TaskGuidedEnhancer
from pneuvision.utils.hardware import (
    count_parameters,
    inspect_environment,
    set_deterministic_seed,
)
from pneuvision.utils.plotting import plot_enhancement_progression_panel


def run_tgae_smoke_test(device: torch.device, use_bf16: bool) -> dict:
    """
    Verify that the differentiable Task-Guided Adaptive Enhancer (`TaskGuidedEnhancer`)
    executes on the target device, supports BF16 autocast, and receives valid gradients
    directly from a binary classification loss (Idea 4 / Bus Intuition).
    """
    enhancer = TaskGuidedEnhancer(mode="task_guided_adaptive", gate_hidden_dim=8).to(device)
    tiny_probe = nn.Sequential(
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(64, 1),
    ).to(device)

    x = torch.rand(4, 1, 256, 256, device=device)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)
    criterion = nn.BCEWithLogitsLoss()

    amp_enabled = (device.type == "cuda") and use_bf16
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        x_enh, reg_loss = enhancer(x)
        logits = tiny_probe(x_enh)
        cls_loss = criterion(logits, y)
        total_loss = cls_loss + 0.01 * reg_loss

    total_loss.backward()

    wc, wd = enhancer.get_enhancement_weights(x)
    grad_norms = [
        float(p.grad.norm().item())
        for p in enhancer.parameters()
        if p.grad is not None
    ]

    return {
        "tgae_trainable_params": count_parameters(enhancer),
        "forward_output_shape": list(x_enh.shape),
        "cls_loss": round(float(cls_loss.item()), 5),
        "reg_loss": round(float(reg_loss.item()), 5),
        "mean_contrast_weight_wc": round(float(wc.mean().item()), 4),
        "mean_detail_weight_wd": round(float(wd.mean().item()), 4),
        "gradients_valid": len(grad_norms) > 0 and all(np.isfinite(g) and g > 0 for g in grad_norms),
        "grad_norm_sum": round(float(sum(grad_norms)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 0: Verify L40S GPU environment, audit Chest X-Ray dataset, and preview enhancement."
    )
    parser.add_argument("--dataset-dir", type=str, default="./DATASET", help="Path to DATASET/ directory")
    parser.add_argument("--tar-path", type=str, default=None, help="Optional path to dataset.tar.gz to auto-extract")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory for JSON audit and figures")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    out_dir = Path(args.output_dir).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("PneuVision — Stage 0: Environment Verification, Dataset Audit & Enhancement Smoke Test")
    print("=" * 85)

    # 1. Hardware & Environment Check
    env_info = inspect_environment(str(out_dir))
    print("\n[1/4] Hardware & Software Environment:")
    for k, v in env_info.items():
        print(f"  - {k:<18}: {v}")

    device = torch.device("cuda" if env_info["cuda_available"] else "cpu")
    use_bf16 = bool(env_info["bf16_supported"])

    # 2. Dataset Extraction & Audit
    print("\n[2/4] Auditing Pediatric Chest X-Ray Dataset:")
    ds_path = ensure_dataset_extracted(args.dataset_dir, args.tar_path)
    audit = audit_dataset_directory(ds_path)

    if audit["exists"]:
        print(f"  - Dataset found at  : {audit['dataset_dir']}")
        print(f"  - Total radiographs : {audit['total_images']}")
        for split_name, s_info in audit["splits"].items():
            if "NORMAL" in s_info:
                print(
                    f"  - {split_name:<15}: {s_info['total']:>5} images "
                    f"(NORMAL: {s_info['NORMAL']}, PNEUMONIA: {s_info['PNEUMONIA']}, "
                    f"Prevalence: {s_info['pneumonia_prevalence_pct']}%, "
                    f"BCE pos_weight: {s_info['pos_weight_for_bce']})"
                )
            else:
                print(f"  - {split_name:<15}: {s_info['total']:>5} unlabeled images")
        res = audit["resolution_sample_stats"]
        print(f"  - Color modes       : {res['modes_observed']}")
        print(f"  - Width (min/med/max): {res['width_min_median_max']} px")
        print(f"  - Height(min/med/max): {res['height_min_median_max']} px")

        # Load one real NORMAL and one real PNEUMONIA radiograph for the preview panel
        norm_p = _list_images_in_dir(ds_path / "train" / "NORMAL")[0]
        pneu_p = _list_images_in_dir(ds_path / "train" / "PNEUMONIA")[0]
        with Image.open(norm_p) as im:
            norm_img = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
        with Image.open(pneu_p) as im:
            pneu_img = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
        sample_source = "real_dataset"
    else:
        print(f"  - [Notice] Real dataset not yet present at {ds_path}.")
        print("  - Generating synthetic pediatric NORMAL & PNEUMONIA radiographs for Stage 0 smoke test.")
        norm_img = generate_synthetic_chest_xray(256, is_pneumonia=False, seed=args.seed)
        pneu_img = generate_synthetic_chest_xray(256, is_pneumonia=True, seed=args.seed + 1)
        sample_source = "synthetic_smoke_test"

    # 3. Generate Stage 0 Enhancement Progression Visual Panel
    print("\n[3/4] Generating Radiograph Enhancement Progression Figure:")
    fig_path = plot_enhancement_progression_panel(
        normal_img=norm_img,
        pneumonia_img=pneu_img,
        out_path=fig_dir / "stage0_enhancement_preview.png",
        alphas=(0.0, 0.5, 1.0, 2.0, 3.0),
    )
    print(f"  - Saved visual panel ({sample_source}) -> {fig_path}")

    # 4. Smoke-Test Differentiable Task-Guided Adaptive Enhancer (Bus Intuition)
    print("\n[4/4] Smoke-Testing Differentiable Task-Guided Adaptive Enhancer (TGAE):")
    tgae_check = run_tgae_smoke_test(device=device, use_bf16=use_bf16)
    for k, v in tgae_check.items():
        print(f"  - {k:<26}: {v}")

    report = {
        "environment": env_info,
        "dataset_audit": audit,
        "sample_source_for_preview": sample_source,
        "tgae_smoke_test": tgae_check,
    }
    report_path = out_dir / "stage0_dataset_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Stage 0 Complete] Saved audit report -> {report_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
