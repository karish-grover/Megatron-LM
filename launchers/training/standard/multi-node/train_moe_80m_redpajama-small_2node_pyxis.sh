#!/bin/bash
#SBATCH --job-name=std-moe-80M-2node
#SBATCH --account=mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --output=logs/std-moe-80M-2node-%j.out
#SBATCH --error=logs/std-moe-80M-2node-%j.err
# =============================================================================
# Standard MoE GPT 80M: TEGroupedMLP Backend (Pyxis Multi-Node)
# =============================================================================
# Multi-node training with 2 nodes (16 H200 GPUs total).
# Baseline for comparison with Lorentz MoE.
#
# Usage:
#   sbatch launchers/training/standard/multi-node/train_moe_80m_redpajama-small_2node_pyxis.sh
#
# Expert Type: TEGroupedMLP (Standard Euclidean - NO Hyperbolic)
#   - TransformerEngine's efficient grouped linear operations
#
# Model: ~80M total parameters
#   - 4 layers, hidden_size 256, ffn 512
#   - 4 routed experts + 1 shared expert
#   - Top-2 routing, MoE every 2nd layer
# =============================================================================

set -e

# Create logs directory
mkdir -p logs

# =============================================================================
# Data Configuration (RedPajama-Tiny)
# =============================================================================
# Uses the tiny dataset prepared by: bash launchers/data_processing/prepare_tiny_pyxis.sh
export HOST_DATA_DIR="${HOST_DATA_DIR:-/fsx/karish/data}"
HOST_DATA_PATH="${HOST_DATA_DIR}/processed_data/redpajama_tiny/tiny_text_document"

export DATA_PATH="/workspace/data/processed_data/redpajama_tiny/tiny_text_document"

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
export MODEL_NAME="std-moe-80M-te-2node"
export EXPERT_TYPE="TEGroupedMLP"

# Model Architecture (~80M params)
export MODEL_ARGS_STR="--num-layers 4 --hidden-size 256 --num-attention-heads 4 --group-query-attention --num-query-groups 2 --ffn-hidden-size 512 --seq-length 256 --max-position-embeddings 256 --position-embedding-type rope --normalization RMSNorm --swiglu --disable-bias-linear --no-bias-dropout-fusion --no-persist-layer-norm --untie-embeddings-and-output-weights --tokenizer-type NullTokenizer --vocab-size 151936"

# MoE Configuration (Megatron-style)
# --sequence-parallel required when using MoE + TP together
export MOE_ARGS_STR="--num-experts 4 --moe-shared-expert-intermediate-size 512 --moe-router-topk 2 --moe-layer-freq 2 --moe-aux-loss-coeff 0.01 --moe-token-dispatcher-type allgather --moe-grouped-gemm --sequence-parallel"

# NO Hyperbolic Configuration (Standard Euclidean baseline)

# Export settings for display
export NUM_EXPERTS=4
export NUM_SHARED_EXPERTS=1
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
export SAVE_INTERVAL=100

# Parallelism (TP=2, EP=2 across 16 GPUs)
export TP=2
export PP=1
export EP=2
export GPUS_PER_NODE=8

# Container
# export CONTAINER_NAME="lorentz-moe"  # Uncomment if using saved container

# =============================================================================
# Run with Pyxis
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base/_train_moe_base_pyxis.sh"
