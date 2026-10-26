#!/usr/bin/env bash
# ==============================================================================
# PneuVision — Single-Command End-to-End Pipeline Runner (Stages 0 to 4)
# ==============================================================================
# Usage:
#   1. Full Training & Evaluation + submission.csv (~6 minutes on L40S):
#        bash run_full_pipeline.sh ./DATASET ./outputs
#
#   2. Rapid 15-Second Smoke Test of All Stages (for live demo / verification):
#        bash run_full_pipeline.sh ./DATASET ./outputs --smoke-test
# ==============================================================================

set -euo pipefail

DATASET_DIR="${1:-./DATASET}"
OUTPUT_DIR="${2:-./outputs}"
MODE="${3:-full}"

echo "=============================================================================="
echo "PneuVision End-to-End Pipeline | Dataset: ${DATASET_DIR} | Output: ${OUTPUT_DIR}"
echo "=============================================================================="

# Stage 0: Environment Verification, Dataset Audit & Enhancement Smoke Test
python scripts/00_verify_env_and_data.py \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${OUTPUT_DIR}"

if [[ "${MODE}" == "--smoke-test" ]]; then
    echo "[Smoke-Test Mode] Running rapid 2-epoch verification across Stages 1, 2, and 3..."
    python scripts/01_exp1_enhancement_study.py --dataset-dir "${DATASET_DIR}" --output-dir "${OUTPUT_DIR}" --smoke-test
    python scripts/02_exp2_autoencoder_study.py --dataset-dir "${DATASET_DIR}" --output-dir "${OUTPUT_DIR}" --smoke-test
    python scripts/03_exp3_attention_pareto.py  --dataset-dir "${DATASET_DIR}" --output-dir "${OUTPUT_DIR}" --smoke-test
    echo "[Smoke-Test Complete] All stages verified!"
else
    # Stage 1: Inverted-U Enhancement Curve & Task-Guided Enhancer (TGAE)
    python scripts/01_exp1_enhancement_study.py \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --epochs 25 \
        --batch-size 64

    # Stage 2: 2D Spatial Autoencoder Bottleneck & Coupling Regimes
    python scripts/02_exp2_autoencoder_study.py \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --ae-epochs 15 \
        --cls-epochs 25 \
        --batch-size 64

    # Stage 3: Channel Attention Ablation & Parameter-Budget Pareto Frontier
    python scripts/03_exp3_attention_pareto.py \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --ae-epochs 12 \
        --cls-epochs 25 \
        --batch-size 64

    # Stage 4: Grad-CAM Clinical Interpretability & Final Benchmark Submission
    python scripts/04_interpret_and_submit.py \
        --dataset-dir "${DATASET_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --submission-path ./submission.csv
fi
