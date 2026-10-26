# PneuVision — Complete Viva Defense, Code Walkthrough & Live Hidden-Data Testing Guide

This guide is designed so that you know **every file, equation, tensor shape, design choice, and command** in this project inside and out. Remember: **all four core scientific ideas in this project came directly from your own intuitions** (the 2D spatial autoencoder bottleneck, channel attention for <1M parameter efficiency, CLAHE contrast enhancement, and your "bus intuition" linking image enhancement to classification loss).

---

## Part 1: Live Testing Cheat Sheet (If the Professor Gives You New Unseen Images)

All 5 trained `TG-SECAE` models (**1.14 MB total combined size**) are saved inside [`pretrained_models/`](pretrained_models/) and tracked in Git. You **do not need to retrain** to test on unseen images — inference runs in **~1 second** on GPU (or a few seconds on CPU).

### Scenario A: Professor gives you a folder of UNLABEled hidden X-rays
*(e.g., `/path/to/sir_hidden_test/` containing `.jpeg`, `.jpg`, or `.png` images, and asks for a `submission.csv`)*

```bash
python scripts/05_predict_unseen.py \
    --input /path/to/sir_hidden_test \
    --output-csv ./sir_hidden_submission.csv \
    --detailed-csv ./sir_hidden_with_probabilities.csv
```
- **What it produces:**
  1. `sir_hidden_submission.csv`: Exact benchmark format (`Image_ID,Label` where `0 = NORMAL`, `1 = PNEUMONIA`).
  2. `sir_hidden_with_probabilities.csv`: Includes the exact probability score `P(Pneumonia)` for every image.

---

### Scenario B: Professor gives you a folder of LABELED hidden X-rays (`NORMAL/` and `PNEUMONIA/` subfolders)
*(e.g., `/path/to/sir_labeled_data/` with `NORMAL/` and `PNEUMONIA/` inside it, or testing on `./DATASET/val`)*

```bash
python scripts/05_predict_unseen.py \
    --input /path/to/sir_labeled_data \
    --output-csv ./sir_labeled_predictions.csv
```
- **What happens:** [`scripts/05_predict_unseen.py`](scripts/05_predict_unseen.py) **automatically detects** the `NORMAL/` and `PNEUMONIA/` folders and prints the full clinical evaluation report right on screen:
  - `Accuracy`, `Balanced Accuracy`, `Sensitivity (Recall)`, `Specificity`, `Precision`, `F1-Score`, `AUROC`, and the `Confusion Matrix (TN, FP, FN, TP)`.
- **Tip:** You can also choose which specific model tier to run using `--model`:
  - `--model ensemble` *(Default: 5-model multi-scale ensemble, 189,138 params)*
  - `--model nano` *(Ultra-compact 4,993-parameter model — 19.5 KB!)*
  - `--model micro_se` *(10,313-parameter SE model)*
  - `--model micro_cbam` *(11,848-parameter CBAM model)*
  - `--model tiny` *(37,279-parameter model)*
  - `--model small` *(124,705-parameter model)*

---

### Scenario C: Professor gives you a SINGLE X-ray image and asks "Show me what your model predicts and where it looks"

```bash
python scripts/05_predict_unseen.py \
    --input /path/to/single_xray.jpeg \
    --explain-out ./single_xray_explanation.png
```
- **What happens:**
  1. Prints the predicted label (`NORMAL (0)` or `PNEUMONIA (1)`), the exact probability `P(Pneumonia)`, and the learned enhancement weights ($w_{\text{contrast}}, w_{\text{detail}}$) for that radiograph.
  2. Saves a 5-panel clinical visualization (`single_xray_explanation.png`) showing:
     `[1. Raw X-Ray | 2. Fixed Full CLAHE | 3. Learned TGAE Output | 4. 32×32 2D Latent Feature Map | 5. Grad-CAM Heatmap Overlay]`.

---

### Scenario D: Professor says "Can you run the entire training + evaluation pipeline in a single go?"

1. **Rapid 15-Second Live Verification (Smoke Test of all stages in one go):**
   ```bash
   bash run_full_pipeline.sh ./DATASET ./outputs --smoke-test
   ```
2. **Complete Training & Evaluation from Scratch in One Go (~6 minutes on L40S):**
   ```bash
   bash run_full_pipeline.sh ./DATASET ./outputs
   ```

---

## Part 2: Step-by-Step Walkthrough of How One Image Flows Through Our Code

If the professor asks **"Explain your pipeline step by step from raw image to final prediction"**, walk through these 4 exact stages:

### Step 1: Deterministic Preprocessing ([`pneuvision/data/dataset.py`](pneuvision/data/dataset.py))
- **Input:** Raw chest radiograph (`.jpeg`/`.png`) of arbitrary resolution.
- **Operations:**
  1. Convert to single-channel grayscale (`L`).
  2. Bilinear resize to $256 \times 256$ (`1 × 256 × 256`).
  3. Per-image min-max intensity normalization to $[0.0, 1.0]$ so scanner exposure offsets are standardized.

### Step 2: Module 1 — Task-Guided Adaptive Enhancer (`TGAE`, [`pneuvision/data/enhancement.py`](pneuvision/data/enhancement.py))
- **Your Intuition:** Fixed contrast enhancement (like full CLAHE) helps up to a point, but over-enhancing amplifies normal lung markings (bronchovascular trees, rib edges) into false-positive pneumonia predictions.
- **How `TaskGuidedEnhancer` Works (**only 58 learnable parameters**):**
  1. Given input image $I \in [0, 1]^{1 \times 256 \times 256}$, it computes two differentiable branches on GPU:
     - **Local Contrast Branch $B_{\text{contrast}}(I)$:** Normalizes local intensity using a $31 \times 31$ local mean $\mu$ and local standard deviation $\sigma$:
       $$B_{\text{contrast}}(I) = \sigma_{\text{sigmoid}}\!\left(\gamma \cdot \frac{I - \mu_{31\times 31}(I)}{\sigma_{31\times 31}(I) + \epsilon} + \beta\right)$$
     - **High-Frequency Detail Branch $B_{\text{detail}}(I)$:** Extracts fine alveolar/interstitial edges via Laplacian high-pass residual against a $5 \times 5$ local mean:
       $$B_{\text{detail}}(I) = I - \mu_{5\times 5}(I)$$
  2. **Content-Adaptive Gating Network:** Computes 4 global image statistics $[\mu(I), \sigma(I), \mu(|B_{\text{detail}}|), \sigma(B_{\text{detail}})]$ and passes them through a tiny two-layer MLP ($4 \to 8 \to 2$) to predict two gates per radiograph:
     - $w_c(I) \in [0, 1]$ (how much local contrast enhancement to apply)
     - $w_d(I) \in [0, 0.5]$ (how much high-frequency sharpening to apply)
  3. **Output:**
     $$\tilde{I} = \text{clamp}\!\big((1 - w_c(I))\,I + w_c(I)\,B_{\text{contrast}}(I) + w_d(I)\,B_{\text{detail}}(I),\; 0,\, 1\big)$$
  4. **Why it works:** Because $w_c$ and $w_d$ receive gradients directly from the classification loss $\mathcal{L}_{\text{BCE}}$, they automatically stop at the sweet spot ($w_c \approx 0.74, w_d \approx 0.065$), boosting Specificity on normal lungs from `91.14%` to **`96.84%`** (`+5.70%` fewer false positives!).

### Step 3: Module 2 — 2D Spatial Autoencoder Encoder ([`pneuvision/models/autoencoder.py`](pneuvision/models/autoencoder.py))
- **Your Intuition:** Standard autoencoders flatten the bottleneck into a 1D vector (e.g., $\mathbb{R}^{64}$), which destroys 2D spatial layout (where the opacity is located in the left/right lung lobes) and requires massive fully-connected matrices ($16,384 \times 64 = 1.05\text{M}$ parameters).
- **How `SpatialEncoder2D` Works:**
  - Uses 3 strided **Depthwise-Separable Convolution (`DSConvBlock`)** stages:
    $$(1 \times 256 \times 256) \xrightarrow{\text{Stage 1}} (C_1 \times 128 \times 128) \xrightarrow{\text{Stage 2}} (C_2 \times 64 \times 64) \xrightarrow{\text{Stage 3}} (C_{\text{lat}} \times 32 \times 32)$$
  - Spatial area is compressed by **$64\times$** ($256^2 = 65,536 \to 32^2 = 1,024$), while preserving the 2D grid topology!
  - During training (`Stage 2` & `Stage 3`), a symmetric decoder reconstructs the $256 \times 256$ X-ray (pretrained transductively on `train + unlabeled test` images and jointly regularized during classification). **At test time, the decoder is discarded (`0` extra inference parameters).**

### Step 4: Module 3 — Channel/Spatial Attention (`SE` / `ECA` / `CBAM`) & Latent Classifier ([`pneuvision/models/attention.py`](pneuvision/models/attention.py) & [`pneuvision/models/classifier.py`](pneuvision/models/classifier.py))
- **How it Works on the $32 \times 32$ Latent Feature Map $Z$:**
  1. **Latent Bottleneck Attention:** Applies `SE` (Squeeze-and-Excitation channel recalibration), `ECA` (1D conv across channels, only `+15` parameters!), or `CBAM` (Channel Attention + $7 \times 7$ Spatial Attention mask) directly on $Z \in \mathbb{R}^{C_{\text{lat}} \times 32 \times 32}$.
  2. **Two Lightweight Residual Depthwise-Separable Stages:**
     $$(C_{\text{lat}} \times 32 \times 32) \xrightarrow{\text{ResDSBlock 1}} (2C_{\text{lat}} \times 16 \times 16) \xrightarrow{\text{ResDSBlock 2}} (4C_{\text{lat}} \times 8 \times 8)$$
  3. **Dual Global Pooling + Linear Head:** Concatenates Global Average Pooling (overall parenchymal haziness) and Global Max Pooling (focal lobar consolidation) into a vector of size $8 C_{\text{lat}}$, followed by a single linear layer (`Linear(8*C_lat, 1)`) producing the pneumonia logit $z$.
  4. **Final Prediction:** $p = \sigma(z) \in (0, 1)$, thresholded to `1` (`PNEUMONIA`) or `0` (`NORMAL`).

---

## Part 3: How Well Does the Model Generalize to Unseen Data? (5 Points for Your Viva)

If the professor asks: **"How do you know your model hasn't overfitted and will generalize well to my hidden test set?"**, give these 5 concrete scientific reasons:

1. **Extreme Parameter Capacity Constraint (Strongest Structural Regularizer):**
   - Our training set has `4,684` images. A standard `ResNet-18` has `11,170,753` parameters (`2,385` parameters per training image!), making it easy to memorize training quirks.
   - Our **`TG-SECAE-Nano` has only `4,993` parameters** (~`1` parameter per training image!), and **`TG-SECAE-Micro` has only `10,313–11,848` parameters**. By information bottleneck theory, a 5k–12k parameter network physically lacks the capacity to memorize individual image noise and is forced to learn generalizable anatomical features.
2. **Reconstruction-Regularized Latent Bottleneck (`L_cls + λ L_rec`):**
   - Because the $32 \times 32$ encoder is trained jointly to reconstruct the entire chest radiograph (`PSNR ~ 23.2 dB`, `SSIM ~ 0.72`), the encoder cannot collapse onto spurious background shortcuts (like corner text markers or border artifacts) — it must preserve true lung/rib/heart structure.
3. **Adaptive Exposure & Contrast Normalization (`TGAE`):**
   - Unseen radiographs often come from different X-ray machines with slightly darker or brighter contrast. Because `TGAE` dynamically measures each unseen image's mean and standard deviation $(\mu(I), \sigma(I))$ at test time to compute custom gates $(w_c, w_d)$, it automatically adapts to unseen contrast levels.
4. **Class-Imbalance Counter-Weighting (`pos_weight = 0.3704`):**
   - Pediatric chest X-ray datasets are heavily skewed toward Pneumonia (`73% Pneumonia` vs. `27% Normal`). Naive models over-predict Pneumonia and fail on Normal images (`MobileNetV3-Small` only got `88.61%` Specificity).
   - We weighted the BCE loss by $N_{\text{neg}} / N_{\text{pos}} = 1266 / 3418 = 0.3704$, achieving **`96.20%–96.84%` Specificity** on unseen validation radiographs.
5. **Multi-Scale Ensembling + Horizontal-Flip Test-Time Augmentation (TTA):**
   - For the final submission (`scripts/04_interpret_and_submit.py` and `scripts/05_predict_unseen.py`), every unseen radiograph is evaluated both in its original orientation and horizontally flipped, averaged across 5 spatial scales (`Nano 5k`, `Micro-SE 10k`, `Micro-CBAM 12k`, `Tiny 37k`, `Small 125k`), reaching **`99.04%` AUROC** and **`99.06%` Sensitivity**.

---

## Part 4: Directory Structure Quick Reference

```text
pneumoniaDetection/
├── pneuvision/                     # Core modular Python package
│   ├── config.py                   # Experiment & dataset configuration dataclasses
│   ├── data/
│   │   ├── dataset.py              # Fast parallel RAM dataset loader & augmentations
│   │   └── enhancement.py          # CLAHE, Unsharp Masking, and 58-param TaskGuidedEnhancer (TGAE)
│   ├── models/
│   │   ├── attention.py            # Squeeze-and-Excitation (SE), ECA, and CBAM blocks
│   │   ├── autoencoder.py          # 1D Vector vs. 2D Spatial Autoencoder (Encoder + Decoder)
│   │   ├── classifier.py           # TGSECAEClassifier (Nano 5k, Micro 11k, Tiny 37k, Small 125k)
│   │   └── baselines.py            # Torchvision baselines (ResNet-18, MobileNetV2, ShuffleNetV2, etc.)
│   └── utils/
│       ├── hardware.py             # Seed reproducibility, GPU/VRAM audit, parameter & MACs counter
│       ├── metrics.py              # Clinical metrics (Acc, BalAcc, Sens, Spec, F1, AUROC, threshold)
│       └── plotting.py             # 300-DPI Matplotlib publication figure generators
├── scripts/
│   ├── 00_verify_env_and_data.py   # Stage 0: Hardware check, dataset audit & TGAE smoke test
│   ├── 01_exp1_enhancement_study.py# Stage 1: Inverted-U CLAHE curve vs. TGAE study
│   ├── 02_exp2_autoencoder_study.py# Stage 2: 1D vs. 2D bottleneck & 5 coupling regimes study
│   ├── 03_exp3_attention_pareto.py # Stage 3: SE/ECA/CBAM ablation & Pareto frontier vs. ResNet-18
│   ├── 04_interpret_and_submit.py  # Stage 4: Grad-CAM panel, ROC/Confusion plots & submission.csv
│   └── 05_predict_unseen.py        # Universal inference & evaluation script for ANY new unseen images
├── pretrained_models/              # All 5 trained TG-SECAE checkpoints (< 1.15 MB total, Git-tracked)
├── outputs/                        # All generated JSON metrics, Markdown tables, and 300-DPI figures
├── run_full_pipeline.sh            # Single-command end-to-end bash runner
├── submission.csv                  # Final verified 588-line benchmark submission (Image_ID,Label)
├── RESEARCH_REPORT.md              # Complete paper-style research manuscript with all tables
└── VIVA_AND_DEMO_GUIDE.md          # This viva defense & live demo guide
```
