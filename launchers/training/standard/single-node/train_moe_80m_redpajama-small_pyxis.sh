#!/bin/bash
# =============================================================================
# Standard Megatron MoE GPT 80M: TEGroupedMLP Backend (Pyxis/Enroot)
# =============================================================================
# Standard (non-hyperbolic) MoE training with TEGroupedMLP.
# Baseline for comparison with Lorentz MoE.
# For use on Slurm clusters with Pyxis/Enroot (no Docker/Apptainer needed).
#
# Usage (within an salloc allocation):
#   salloc --account=mrs_2 --qos=h200_mrs_2_high --time=1:00:00 \
#          --nodes=1 --ntasks-per-node=1 --gpus-per-node=8 \
#          --cpus-per-task=80 --mem=0
#   bash launchers/training/standard/single-node/train_moe_80m_redpajama-small_pyxis.sh
#
# Expert Type: TEGroupedMLP
#   - TransformerEngine's efficient grouped linear
#   - Euclidean geometry (standard)
#
# Model: ~80M total parameters
#   - 4 layers, hidden_size 256, ffn 512
#   - 4 routed experts
#   - Top-2 routing, MoE every 2nd layer
# =============================================================================

set -e

# =============================================================================
# Data Configuration (RedPajama-Tiny)
# =============================================================================
# Uses the tiny dataset prepared by: bash launchers/data_processing/prepare_tiny_pyxis.sh
HOST_DATA_DIR="${HOST_DATA_DIR:-/fsx/karish/data}"
HOST_DATA_PATH="${HOST_DATA_DIR}/processed_data/redpajama_tiny/tiny_text_document"

export DATA_PATH="/workspace/data/processed_data/redpajama_tiny/tiny_text_document"
export HOST_DATA_DIR

# Validate data exists (skip if using mock data)
if [ ! -f "${HOST_DATA_PATH}.bin" ]; then
    echo "WARNING: Data file not found: ${HOST_DATA_PATH}.bin"
    echo "Run first: bash launchers/data_processing/prepare_tiny_pyxis.sh"
    echo "Falling back to mock data."
    unset DATA_PATH
    unset HOST_DATA_DIR
fi

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="standard-moe-80M-te"
export EXPERT_TYPE="TEGroupedMLP"

# Model Architecture (~80M params)
export MODEL_ARGS=(
    --num-layers 4
    --hidden-size 256
    --num-attention-heads 4
    --group-query-attention
    --num-query-groups 2
    --ffn-hidden-size 512
    --seq-length 256
    --max-position-embeddings 256
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --no-bias-dropout-fusion
    --no-persist-layer-norm
    --untie-embeddings-and-output-weights
    --tokenizer-type NullTokenizer
    --vocab-size 151936
)

# MoE Configuration (Megatron-style)
export MOE_ARGS=(
    --num-experts 4
    --moe-router-topk 2
    --moe-layer-freq 2
    --moe-aux-loss-coeff 0.01
    --moe-token-dispatcher-type allgather
    --moe-grouped-gemm
)

# Export settings for display
export NUM_EXPERTS=4
export MOE_ROUTER_TOPK=2
export MOE_LAYER_FREQ=2
export MOE_AUX_LOSS_COEFF=0.01

# Training Hyperparameters
export BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export LR=3e-4
export MIN_LR=3e-5
export MAX_STEPS=1000
export WARMUP_STEPS=100
export WEIGHT_DECAY=0.1
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=500

# Parallelism
export TP=1
export PP=1
export EP=1
export GPUS_PER_NODE=8

# Container
# export CONTAINER_NAME="lorentz-moe"  # Uncomment if using saved container

# =============================================================================
# Run
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base/_train_moe_base_pyxis.sh"
