# PneuVision (`pneuvision`): Task-Guided Enhancement, 2D Spatial Autoencoder Compression & Squeeze-and-Excitation for Ultra-Lightweight Pediatric Pneumonia Detection

A clean, modular, publication-oriented PyTorch research framework for **Pediatric Chest X-Ray Pneumonia Classification** under a strict **< 1M parameter budget (targeting 25k–300k parameters)**, engineered for execution on the **NVIDIA L40S (`rama`)**.

---

## Core Scientific Contributions & Architecture (`TG-SECAE`)

1. **Module 1 — Task-Guided Adaptive Enhancer (`TGAE`):**
   - Addresses the **Over-Enhancement Inverted-U Trap** in chest radiography (where excessive contrast enhancement amplifies normal bronchovascular markings and quantum noise into false positives).
   - Uses a differentiable, 58-parameter content-adaptive gating network driven directly by the downstream binary classification loss to tune local contrast and high-frequency parenchymal detail per radiograph.
2. **Module 2 — Reconstruction-Regularized 2D Spatial Autoencoder Bottleneck:**
   - Compresses $1 \times 256 \times 256$ radiographs into a compact $C \times 32 \times 32$ 2D spatial latent grid ($64\times$ area reduction) using depthwise-separable convolutions.
   - Regularized by a reconstruction Decoder $D$ during training (including optional transductive pretraining on `test_unlabeled/`) that is discarded at inference (**0 extra parameters at test time**).
3. **Module 3 — Channel-Recalibrated Latent Classifier (`SE` / `ECA`):**
   - Operates strictly on the $32 \times 32$ spatial latent representation using ultra-narrow residual blocks with **Squeeze-and-Excitation (SE)** and **Efficient Channel Attention (ECA)**.

---

## Stage-by-Stage Publication Roadmap

- [x] **Stage 0: Environment Verification, Dataset Audit & Enhancement Smoke Test**
  - L40S (`rama`) GPU/VRAM/BF16 verification & MACs/Parameter profiler ([`pneuvision/utils/hardware.py`](pneuvision/utils/hardware.py))
  - Clinical evaluation metrics: Accuracy, Balanced Accuracy, Sensitivity (Recall), Specificity, Precision, NPV, F1-Score, AUROC, AUPRC, and optimal threshold calibration ([`pneuvision/utils/metrics.py`](pneuvision/utils/metrics.py))
  - Deterministic Tile-Interpolated CLAHE, Multi-View 3-Channel Stack, and Differentiable `TaskGuidedEnhancer` ([`pneuvision/data/enhancement.py`](pneuvision/data/enhancement.py))
  - Dataset auditor & Stage 0 verification script ([`scripts/00_verify_env_and_data.py`](scripts/00_verify_env_and_data.py))
- [x] **Stage 1: Exp 1 — The "Inverted-U" Enhancement Study & Task-Guided Adaptive Enhancer**
  - Empirically confirmed the **Inverted-U Over-Enhancement Curve** (Fixed $\alpha = 0.25\text{--}0.50$ improves F1 from `97.20% -> 97.55%` and Balanced Accuracy from `94.40% -> 95.55%`, whereas full unblended CLAHE $\alpha = 1.00$ drops to `93.66%` Balanced Accuracy / `96.23%` F1).
  - **`TGAE-Adaptive` (+58 params, 8,303 total params)** achieved the **highest Balanced Accuracy (`96.31%`), highest Specificity (`96.84%`, +5.70% over Raw), and highest AUROC (`98.60%`)** of all 11 conditions.
- [x] **Stage 2: Exp 2 — 2D Spatial Autoencoder Compression & Coupling Regimes**
  - Empirically proved that **2D Spatial Bottlenecks (`16×32×32`)** reduce inference parameters by **15.2× (`8,839` vs. `134,361`)** compared to a **1D Vector Bottleneck (`R^64`)** while improving reconstruction structural similarity (`SSIM 0.699–0.746` vs. `0.631`) and diagnostic accuracy.
  - Established that **Transductive Unsupervised Autoencoder Pretraining (`train + unlabeled test`) + Joint Multi-Task Fine-Tuning (`5_transductive_ae_joint`)** achieves **`95.95%` Balanced Accuracy, `94.94%` Specificity, `97.53%` F1, and `98.72%` AUROC** at just **8,839 inference parameters** (while `3_enhance_ae_finetune` reaches **`97.56%` F1** and **`98.13%` Sensitivity**).
- [x] **Stage 3: Exp 3 — Channel Attention (`SE` / `ECA` / `CBAM`) & Parameter-Budget Pareto Frontier vs. Baselines**
  - Channel & Spatial Attention Ablation (`none`, `eca_k3`, `se_r8`, `se_r4`, `se_r2`, `cbam`) and 4-tier `TG-SECAE` (`Nano`, `Micro`, `Tiny`, `Small`) Pareto Frontier against `ShuffleNetV2-0.5x`, `SqueezeNet-1.1`, `MobileNetV2`, and `ResNet-18` ([`pneuvision/models/baselines.py`](pneuvision/models/baselines.py), [`scripts/03_exp3_attention_pareto.py`](scripts/03_exp3_attention_pareto.py))
- [ ] **Stage 4: Grad-CAM Clinical Interpretability & Final `submission.csv` Generation**
