# PneuVision Execution Runbook (`rama` L40S 44 GiB)

Follow these commands sequentially on `rama`.

---

## Stage 0: Environment Verification, Dataset Audit & Enhancement Smoke Test

### Step 0.1: Activate Conda Environment & Pull Repository on `rama`

```bash
conda activate qwen-vl
cd ~/projects/pneumoniaDetection   # (or wherever you clone this repository on rama)
git pull
```

### Step 0.2: Verify L40S GPU, Audit Dataset & Generate Enhancement Progression Figure

If `dataset.tar.gz` is located on `rama` (e.g., `~/datasets/dataset.tar.gz`), pass `--tar-path` to extract it automatically into `./DATASET` on first run:

```bash
python scripts/00_verify_env_and_data.py \
    --dataset-dir ./DATASET \
    --tar-path ~/datasets/dataset.tar.gz \
    --output-dir ./outputs
```

*(Note: Even if you pass `--dataset-dir ./dataset`, the script now automatically detects uppercase `./DATASET` and `DATASET/test/` on Linux!)*

**What to check after Step 0.2:**
1. Terminal output confirms `NVIDIA L40S` (~44 GiB VRAM), `bf16_supported: True`, and `gradients_valid: True` on `TaskGuidedEnhancer` (58 trainable parameters).
2. `outputs/stage0_dataset_audit.json` records exact counts (`train`: 4,684, `val`: 585, `test`: 587), class prevalence, and the positive class weight for `BCEWithLogitsLoss`.
3. `outputs/figures/stage0_enhancement_preview.png` visualizes `NORMAL` vs. `PNEUMONIA` radiographs across $\alpha \in \{0.0, 0.5, 1.0, 2.0, 3.0\}$ and the High-Pass Parenchymal Detail channel.

---

## Stage 1 (Exp 1): The "Inverted-U" Enhancement Study & Task-Guided Enhancer (`TGAE`)

### Step 1.1: Run Mandatory 2-Epoch Training Smoke Test on `rama`

Per the L40S execution workflow, verify the end-to-end training pipeline (`DataLoader -> TaskGuidedEnhancer -> TinyPneuNet -> BF16 Forward -> BCE Loss -> Backward -> Optimizer Step -> Checkpoint`) on a 128-image real subset first:

```bash
python scripts/01_exp1_enhancement_study.py \
    --dataset-dir ./DATASET \
    --output-dir ./outputs \
    --smoke-test
```

### Step 1.2: Run Full Stage 1 Publication Study (11 Conditions)

Once Step 1.1 passes, run the complete Stage 1 ablation study across:
- **Exp 1A (Fixed Enhancement Sweep):** $\alpha \in \{0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0\}$
- **Exp 1B (Multi-View & Task-Guided Enhancers):** `multiview_3ch_stack`, `tgae_global_2params`, and `tgae_adaptive_58params`

```bash
python scripts/01_exp1_enhancement_study.py \
    --dataset-dir ./DATASET \
    --output-dir ./outputs \
    --epochs 25 \
    --batch-size 64
```

**Generated Publication Outputs (`./outputs`):**
- `outputs/stage1_summary_table.md`: Complete Markdown table (Trainable Params, MACs, Val Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1-Score, AUROC, AUPRC).
- `outputs/stage1_enhancement_results.json`: Full per-epoch training/validation histories and learned `TGAE` gate statistics (`w_c`, `w_d`).
- `outputs/figures/stage1_inverted_u_curve.png`: Publication figure plotting the Inverted-U Over-Enhancement Curve vs. `TGAE`.
- `outputs/figures/stage1_tgae_visual_comparison.png`: Visual comparison of Raw vs. Learned `TGAE-Adaptive` vs. Residual Map vs. Over-Enhanced ($\alpha = 2.5$) on real validation radiographs.

---

## Stage 2 (Exp 2): 2D Spatial Autoencoder Compression & Coupling Regimes

### Step 2.1: Run Mandatory 2-Epoch Autoencoder + Classifier Smoke Test on `rama`

```bash
git pull
python scripts/02_exp2_autoencoder_study.py \
    --dataset-dir ./DATASET \
    --output-dir ./outputs \
    --smoke-test
```

### Step 2.2: Run Full Stage 2 Publication Study (Exp 2A + Exp 2B)

Runs:
- **Exp 2A (Table 2A — Bottleneck Geometry):** `ae_1d_vec64` (1D Vector $\mathbb{R}^{64}$) vs. `ae_2d_16x16_c16`, `ae_2d_32x32_c16`, and `ae_2d_64x64_c16` (measuring Inference Params, MACs, Reconstruction PSNR/SSIM, and Classification Metrics).
- **Exp 2B (Table 2B — 5 Coupling Regimes on `16×32×32` + `TGAE`):**
  1. `1_two_stage_frozen`
  2. `2_two_stage_finetune`
  3. `3_enhance_ae_finetune` (Raw $\to$ Enhanced AE pretraining + fine-tuning)
  4. `4_joint_multitask` (End-to-End Multi-Task BCE + Reconstruction Regularizer)
  5. `5_transductive_ae_joint` (Unsupervised AE pretrained on `train/` + unlabeled `test/` + Joint Multi-Task)

```bash
python scripts/02_exp2_autoencoder_study.py \
    --dataset-dir ./DATASET \
    --output-dir ./outputs \
    --ae-epochs 15 \
    --cls-epochs 25 \
    --batch-size 64
```

**Generated Publication Outputs (`./outputs`):**
- `outputs/stage2_summary_table.md`: Publication Markdown tables (`Table 2A` & `Table 2B`).
- `outputs/stage2_autoencoder_results.json`: Complete metrics and PSNR/SSIM evaluations.
- `outputs/figures/stage2_ae_reconstructions.png`: Visual comparison of Original vs. 1D Vector vs. 2D 16×16 / 32×32 / 64×64 reconstructions + 32×32 Latent Energy Map.
- `outputs/figures/stage2_coupling_comparison.png`: Bar/line plots comparing bottleneck geometries and the 5 coupling regimes.
