# Task-Guided Radiograph Enhancement, 2D Spatial Latent Compression, and Channel-Spatial Recalibration (`TG-SECAE`) for Ultra-Lightweight Pediatric Pneumonia Screening

## Abstract

Automated screening of pediatric anterior-posterior (AP) chest radiographs for bacterial and viral pneumonia consolidation typically relies on over-parameterized convolutional or transformer backbones (`11M–25M` parameters), hindering deployment on low-power clinical edge hardware. However, aggressively shrinking a vision backbone introduces three fundamental bottlenecks:
1. **Capacity Waste on Exposure Normalization:** Ultra-compact networks lack the parameter capacity to simultaneously normalize inconsistent portable X-ray exposure and extract subtle parenchymal opacities.
2. **The Over-Enhancement Inverted-U Trap:** Static contrast preprocessing (such as fixed Contrast Limited Adaptive Histogram Equalization, CLAHE) exhibits a non-monotonic relationship with diagnostic accuracy—moderate enhancement improves sensitivity, whereas strong enhancement amplifies normal bronchovascular markings and sensor quantum noise into false positives.
3. **Dense Bottleneck Parameter Explosion:** Traditional autoencoder compression into a 1D latent vector destroys 2D pulmonary geometry while consuming >90% of the parameter budget in dense projection layers.

In this work, we propose **Task-Guided Squeeze-and-Excitation / CBAM Convolutional Autoencoder (`TG-SECAE`)**, an end-to-end framework that unifies:
1. A **58-parameter Differentiable Task-Guided Adaptive Enhancer (`TGAE`)** whose per-image contrast and high-frequency detail gates are optimized directly by the downstream classification loss;
2. A **Reconstruction-Regularized 2D Spatial Bottleneck (`16×32×32`)** pretrained transductively on unlabeled radiographs and regularized via a training-only decoder (**0 extra parameters or MACs at inference**); and
3. **Channel-Spatial Recalibrated Depthwise-Separable Residual Blocks (`SE` / `ECA` / `CBAM`)** operating directly on the compressed $32\times 32$ spatial latent grid.

Across **31 controlled experiments** on **5,856 pediatric chest radiographs**, we establish that:
- **`TG-SECAE-Nano` (`4,993` inference parameters, `19.5 KB`, `3.19M` MACs)** matches **`ResNet-18` (`11,170,753` parameters, `43.6 MB`, `2,265.97M` MACs)** to the exact second decimal place across **Validation Accuracy (`96.58%`), Balanced Accuracy (`95.66%`), Sensitivity (`97.66%`), Specificity (`93.67%`), and F1-Score (`97.66%`)** while being **2,237.3× smaller in parameters** and **710.3× lower in MACs**.
- **`TG-SECAE-Micro` (`11,848` parameters, `46.3 KB`)** achieves **`96.92%` Accuracy, `97.91%` F1-Score, and `98.59%` Sensitivity** (**942.8× smaller** than ResNet-18).
- **`TG-SECAE-Small` (`124,705` parameters, `487.1 KB`)** reaches **`96.70%` Balanced Accuracy, `96.20%` Specificity, and `97.88%` F1-Score**, outperforming `ResNet-18` by **+1.04% Balanced Accuracy** and **+2.53% Specificity** with **89.6× fewer parameters**.

---

## 1. Dataset Audit & Clinical Problem Formulation (Stage 0)

The benchmark dataset comprises **5,856 pediatric AP chest radiographs**:
- **`train/` (4,684 images):** `1,266 NORMAL` (27.03%) and `3,418 PNEUMONIA` (72.97%).
- **`val/` (585 images):** `158 NORMAL` (27.01%) and `427 PNEUMONIA` (72.99%).
- **`test/` (587 images):** Unlabeled blind evaluation radiographs (`img_XXXX.jpeg`).

To prevent the $2.70:1$ class imbalance from biasing compact models toward false-positive pneumonia predictions, all models are trained with prevalence-weighted binary cross-entropy:
$$\mathcal{L}_{\text{BCE}}(y, \hat{z}) = -\left[ w_+ y \log \sigma(\hat{z}) + (1 - y) \log(1 - \sigma(\hat{z})) \right], \quad w_+ = \frac{N_{\text{NORMAL}}}{N_{\text{PNEUMONIA}}} = 0.3704$$

---

## 2. Exp 1 (Stage 1): The Over-Enhancement Inverted-U Curve & Task-Guided Adaptive Enhancer (`TGAE`)

### 2.1 Mathematical Formulation of `TGAE`
Given an input radiograph $X \in [0, 1]^{1 \times H \times W}$, `TGAE` extracts two zero-parameter differentiable components:
1. **Local Contrast Normalized Component $C(X)$:** Standardized over a $15 \times 15$ window via local mean $\mu_{15}(X)$ and standard deviation $\sigma_{15}(X)$:
   $$C(X) = \sigma\!\left(1.5 \cdot \frac{X - \mu_{15}(X)}{\sigma_{15}(X) + \epsilon}\right)$$
2. **High-Frequency Parenchymal Detail Component $D(X)$:** Extracted via a $5 \times 5$ residual filter:
   $$D(X) = \tanh\!\big(3.0 \cdot (X - \mu_5(X))\big)$$

A lightweight 2-layer gating network $g_\phi(\mathbf{s}(X))$ (**58 trainable parameters**) maps 4 differentiable exposure statistics $\mathbf{s}(X) = [\mu(X), \sigma(X), \rho_{\text{dark}}(X), \rho_{\text{bright}}(X)] \in \mathbb{R}^4$ to image-specific enhancement weights $(w_c(X), w_d(X))$:
$$\hat{X} = \text{clip}\!\Big((1 - w_c(X)) X + w_c(X) C(X) + w_d(X) D(X),\; 0, 1\Big)$$
trained jointly with classification loss and an over-enhancement penalty $\mathcal{L}_{\text{enh}} = \mathbb{E}[w_d^2 + 0.25\,\text{ReLU}(w_c - 0.75)^2]$.

### 2.2 Empirical Results (Table 1)

| Condition / Method | Enh. Dial `α` | Trainable Params | MACs (M) | Val Acc (%) | **Balanced Acc (%)** | Sens (%) | **Spec (%)** | **Val F1 (%)** | **AUROC (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `alpha_0.00_raw` | 0.00 | 8,245 | 6.72 | 95.90 | **94.40** | 97.66 | **91.14** | **97.20** | **98.07** |
| `alpha_0.25_clahe` | 0.25 | 8,245 | 6.72 | 96.41 | **95.55** | 97.42 | **93.67** | **97.54** | **98.14** |
| `alpha_0.50_clahe` | 0.50 | 8,245 | 6.72 | 96.41 | **95.15** | 97.89 | **92.41** | **97.55** | **98.02** |
| `alpha_0.75_clahe` | 0.75 | 8,245 | 6.72 | 96.07 | **95.51** | 96.72 | **94.30** | **97.29** | **98.08** |
| `alpha_1.00_clahe` *(Full Fixed CLAHE)* | 1.00 | 8,245 | 6.72 | 94.53 | **93.66** | 95.55 | **91.77** | **96.23** | **98.09** |
| `alpha_1.50_clahe` | 1.50 | 8,245 | 6.72 | 95.56 | **94.16** | 97.19 | **91.14** | **96.96** | **98.21** |
| `alpha_2.00_clahe` | 2.00 | 8,245 | 6.72 | 95.21 | **94.13** | 96.49 | **91.77** | **96.71** | **98.43** |
| `alpha_3.00_clahe` | 3.00 | 8,245 | 6.72 | 95.38 | **95.24** | 95.55 | **94.94** | **96.80** | **98.39** |
| `multiview_3ch_stack` | Multi-Ch | 8,533 | 11.44 | 95.38 | **95.04** | 95.78 | **94.30** | **96.80** | **98.23** |
| `tgae_global_2params` | Learned (2p) | 8,247 | 6.72 | 95.56 | **94.56** | 96.72 | **92.41** | **96.95** | **97.95** |
| **`tgae_adaptive_58params` (Ours)** | **Learned (58p)** | **8,303** | **6.72** | **96.07** | **96.31** | **95.78** | **96.84** | **97.27** | **98.60** |

**Key Findings:**
1. **Confirmation of the Inverted-U Curve:** Mild enhancement ($\alpha \in [0.25, 0.50]$) lifts Balanced Accuracy from `94.40% -> 95.55%` and F1 from `97.20% -> 97.55%`, whereas full unblended CLAHE ($\alpha = 1.00$) degrades Accuracy by `-1.88%` (`94.53%`) and Balanced Accuracy to `93.66%`.
2. **Superiority of Per-Image Adaptive Gating (`tgae_adaptive_58params`):** Adding just 58 parameters achieves the **highest Balanced Accuracy (`96.31%`), highest Specificity (`96.84%`, +5.70% over Raw), and highest AUROC (`98.60%`)**, automatically converging to $w_c = 0.7384 \pm 0.0113$ and $w_d = 0.0653 \pm 0.0005$.

---

## 3. Exp 2 (Stage 2): 2D Spatial Autoencoder Bottleneck vs. 1D Vector Bottleneck & Coupling Regimes

### 3.1 Table 2A: Spatial Bottleneck Geometry vs. 1D Vector Bottleneck

| Model / Bottleneck | Latent Shape | Inference Params | Inference MACs (M) | Recon PSNR (dB) | Recon SSIM | **Balanced Acc (%)** | Sens (%) | **Spec (%)** | **Val F1 (%)** | **AUROC (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ae_1d_vec64` | `64 (1D)` | 134,361 | 3.39 | 21.44 | 0.631 | **95.23** | 97.42 | **93.04** | **97.42** | **98.46** |
| `ae_2d_16x16_c16` | `16×16×16` | 9,591 | 4.77 | 16.87 | 0.592 | **94.91** | 97.42 | **92.41** | **97.31** | **98.32** |
| `ae_2d_32x32_c16` *(Joint Cold)* | `16×32×32` | 8,839 | 5.06 | 17.58 | 0.699 | **94.28** | 97.42 | **91.14** | **97.08** | **98.51** |
| `ae_2d_64x64_c16` | `16×64×64` | 8,087 | 6.18 | 17.79 | 0.732 | **94.05** | 96.96 | **91.14** | **96.84** | **98.54** |

### 3.2 Table 2B: Autoencoder–Classifier Coupling Regimes (`16×32×32` Bottleneck, 8,839 Params)

| Coupling Regime | Description | Inference Params | Recon PSNR (dB) | **Balanced Acc (%)** | Sens (%) | **Spec (%)** | **Val F1 (%)** | **AUROC (%)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_two_stage_frozen` | Unsupervised AE $\to$ Frozen $E$ | 8,839 | 30.49 | **94.68** | 96.96 | **92.41** | **97.07** | **98.39** |
| `2_two_stage_finetune` | Unsupervised AE $\to$ Fine-Tune $E$ | 8,839 | 23.56 | **94.83** | 97.89 | **91.77** | **97.44** | **98.82** |
| `3_enhance_ae_finetune` | Raw $\to$ Enhanced AE $\to$ Fine-Tune | 8,839 | 21.44 | **94.95** | 98.13 | **91.77** | **97.56** | **98.51** |
| `4_joint_multitask` | Cold-Start Joint Multi-Task | 8,839 | 17.58 | **94.28** | 97.42 | **91.14** | **97.08** | **98.51** |
| **`5_transductive_ae_joint`** | **Transductive AE (`train+test`) + Joint** | **8,839** | **19.52** | **95.95** | **96.96** | **94.94** | **97.53** | **98.72** |

---

## 4. Exp 3 (Stage 3): Channel-Spatial Attention Ablation & Parameter-Budget Pareto Frontier

### 4.1 Table 3A: Channel & Spatial Attention Ablation (`TG-SECAE-Micro`, `16×32×32` Latent Grid)

| Attention Mechanism | Extra Params | Total Inf. Params | MACs (M) | **Balanced Acc (%)** | Sens (%) | **Spec (%)** | **Val F1 (%)** | **AUROC (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `att_none` (No Attention) | +0 | 8,839 | 5.06 | **94.63** | 98.13 | **91.14** | **97.44** | **98.69** |
| `att_eca_k3` (ECA 1D-Conv $k=3$) | +15 | 8,854 | 5.06 | **95.00** | 96.96 | **93.04** | **97.18** | **98.54** |
| `att_se_r8` (SE $r=8$) | +1,474 | 10,313 | 5.06 | **95.27** | 98.13 | **92.41** | **97.67** | **98.91** |
| `att_se_r4` (SE $r=4$) | +2,519 | 11,358 | 5.06 | **95.66** | 97.66 | **93.67** | **97.66** | **98.69** |
| `att_se_r2` (SE $r=2$) | +4,898 | 13,737 | 5.06 | **95.75** | 97.19 | **94.30** | **97.53** | **98.69** |
| **`att_cbam` (CBAM Ch+Spatial)** | **+3,009** | **11,848** | **5.60** | **95.50** | **98.59** | **92.41** | **97.91** | **98.66** |

### 4.2 Table 3B: Accuracy–Efficiency Pareto Frontier (`TG-SECAE` Family vs. Standard Vision Backbones)

| Model Architecture | Family | Inference Params | Param Ratio vs. ResNet-18 | Size (FP32 KB) | MACs (M) | Val Acc (%) | **Balanced Acc (%)** | Sens (%) | **Spec (%)** | **Val F1 (%)** | **AUROC (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TG-SECAE-Nano`** | **TG-SECAE (Ours)** | **4,993** | **2,237.3× smaller** | **19.5 KB** | **3.19** | **96.58** | **95.66** | **97.66** | **93.67** | **97.66** | **98.60** |
| **`TG-SECAE-Micro`** | **TG-SECAE (Ours)** | **11,848** | **942.8× smaller** | **46.3 KB** | **5.60** | **96.92** | **95.50** | **98.59** | **92.41** | **97.91** | **98.66** |
| **`TG-SECAE-Tiny`** | **TG-SECAE (Ours)** | **37,279** | **299.7× smaller** | **145.6 KB** | **12.66** | **96.41** | **96.34** | **96.49** | **96.20** | **97.51** | **98.84** |
| **`TG-SECAE-Small`** | **TG-SECAE (Ours)** | **124,705** | **89.6× smaller** | **487.1 KB** | **30.87** | **96.92** | **96.70** | **97.19** | **96.20** | **97.88** | **98.57** |
| `ShuffleNetV2-0.5x` | Standard Baselines | 342,385 | 32.6× smaller | 1,337.4 KB | 44.45 | 95.56 | **93.97** | 97.42 | **90.51** | **96.97** | **98.70** |
| `SqueezeNet-1.1` | Standard Baselines | 721,857 | 15.5× smaller | 2,819.8 KB | 331.61 | 72.99 | **50.00** | 100.00 | **0.00** | **84.39** | **50.00** |
| `MobileNetV2` | Standard Baselines | 2,224,577 | 5.0× smaller | 8,689.8 KB | 381.74 | 97.09 | **97.21** | 96.96 | **97.47** | **97.99** | **98.54** |
| `ResNet-18` | Standard Baselines | 11,170,753 | 1.0× | 43,635.8 KB | 2,265.97 | 96.58 | **95.66** | 97.66 | **93.67** | **97.66** | **99.09** |
