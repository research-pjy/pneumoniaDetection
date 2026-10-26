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
  - Ultra-lightweight depthwise-separable classifier probe [`TinyPneuNet`](pneuvision/models/classifier.py) (~7.3k parameters) & attention blocks ([`pneuvision/models/attention.py`](pneuvision/models/attention.py))
  - Fast RAM-cached radiograph loader ([`load_split_into_ram`](pneuvision/data/dataset.py)) for multi-run ablation sweeps on `rama`
  - Fixed Enhancement Strength Sweep ($\alpha \in \{0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0\}$) vs. Multi-View 3-Channel vs. `TGAE-Global` (+2 params) and `TGAE-Adaptive` (+58 params) ([`scripts/01_exp1_enhancement_study.py`](scripts/01_exp1_enhancement_study.py))
- [ ] **Stage 2: Exp 2 — 2D Spatial Autoencoder Compression & Coupling Regimes**
- [ ] **Stage 3: Exp 3 — Channel Attention (`SE` / `ECA` / `CBAM`) & Parameter-Budget Pareto Frontier vs. Baselines**
- [ ] **Stage 4: Grad-CAM Clinical Interpretability & Final `submission.csv` Generation**
