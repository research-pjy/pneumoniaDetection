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

If `dataset.tar.gz` is located on `rama` (e.g., `~/datasets/dataset.tar.gz`), pass `--tar-path` to extract it automatically into `./dataset` on first run:

```bash
python scripts/00_verify_env_and_data.py \
    --dataset-dir ./dataset \
    --tar-path ~/datasets/dataset.tar.gz \
    --output-dir ./outputs
```

*(Note: You can also run `python scripts/00_verify_env_and_data.py` before copying `dataset.tar.gz`—it will automatically run a synthetic radiograph smoke test to verify PyTorch, BF16, and the differentiable `TaskGuidedEnhancer`!)*

**What to check after Step 0.2:**
1. Terminal output confirms `NVIDIA L40S` (~44 GiB VRAM), `bf16_supported: True`, and `gradients_valid: True` on `TaskGuidedEnhancer` (58 trainable parameters).
2. `outputs/stage0_dataset_audit.json` records exact counts (`train`: 4,684, `val`: 585, `test_unlabeled`: 587), class prevalence, and the positive class weight for `BCEWithLogitsLoss`.
3. `outputs/figures/stage0_enhancement_preview.png` visualizes `NORMAL` vs. `PNEUMONIA` radiographs across $\alpha \in \{0.0, 0.5, 1.0, 2.0, 3.0\}$ and the High-Pass Parenchymal Detail channel.
