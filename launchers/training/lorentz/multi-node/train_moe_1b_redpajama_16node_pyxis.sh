#!/bin/bash
#SBATCH --job-name=lorentz-moe-1B-helm
#SBATCH --account=mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/lorentz-moe-1B-helm-%j.out
#SBATCH --error=logs/lorentz-moe-1B-helm-%j.err
# =============================================================================
# Lorentz MoE GPT ~1B: HELM-MiCE (Pyxis 16-Node Production)
# =============================================================================
# 1B parameter HELM-MiCE model matching the HELM paper's best hyperparameters.
# Production training on 16 nodes (128 H200 GPUs).
#
# Usage:
#   sbatch launchers/training/lorentz/multi-node/train_moe_1b_redpajama_16node_pyxis.sh
#
# Architecture (from HELM paper best hyperparameters):
#   - 16 layers (1 dense + 15 MoE), hidden_size 910, ffn 3640
#   - 14 attention heads, MLA (kv_lora_rank=257)
#   - 8 routed experts + 1 shared expert, top-2 routing
#   - Per-expert FFN: 1820
#   - Sequence length: 2048
#
# Parallelism (128 GPUs):
#   - TP=2, PP=1, EP=8, DP=64
#   - Global batch size: 256 sequences (grad_accum = 256/64 = 4)
# =============================================================================

set -e

# Create logs directory
mkdir -p logs

# =============================================================================
# Data Configuration (Full RedPajama)
# =============================================================================
export HOST_DATA_DIR="${HOST_DATA_DIR:-/fsx/karish/data}"

# Use full RedPajama blend
if [ -f "${HOST_DATA_DIR}/processed_data/redpajama/data_blend_container.txt" ]; then
    export DATA_BLEND_PATH="/workspace/data/processed_data/redpajama/data_blend_container.txt"
elif [ -f "${HOST_DATA_DIR}/processed_data/redpajama_tiny/tiny_text_document.bin" ]; then
    echo "WARNING: Full RedPajama not processed. Falling back to tiny dataset."
    export DATA_PATH="/workspace/data/processed_data/redpajama_tiny/tiny_text_document"
else
    echo "WARNING: No processed data found. Using mock data."
    echo "Run: bash launchers/data_processing/prepare_redpajama_pyxis.sh"
fi

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="lorentz-moe-1B-helm-mice-16node"
export EXPERT_TYPE="LorentzTEGroupedMLP"

# Model Architecture (~1B params, HELM paper best hyperparameters)
# Note: Use _STR strings (not arrays) for srun compatibility across nodes
export MODEL_ARGS_STR="--num-layers 16 --hidden-size 910 --num-attention-heads 14 --ffn-hidden-size 3640 --seq-length 2048 --max-position-embeddings 2048 --position-embedding-type rope --normalization RMSNorm --swiglu --disable-bias-linear --no-bias-dropout-fusion --no-persist-layer-norm --untie-embeddings-and-output-weights --tokenizer-type HuggingFaceTokenizer --tokenizer-model meta-llama/Llama-3.1-8B --kv-lora-rank 257 --qk-head-dim 65 --qk-pos-emb-head-dim 65 --v-head-dim 65"

# MoE Configuration
# n_dense_layers=1: first layer dense, rest MoE
# --sequence-parallel required with TP > 1
export MOE_ARGS_STR="--num-experts 8 --moe-ffn-hidden-size 1820 --moe-shared-expert-intermediate-size 1820 --moe-router-topk 2 --moe-layer-freq ([0]+[1]*15) --moe-aux-loss-coeff 0.01 --moe-token-dispatcher-type alltoall --moe-grouped-gemm --sequence-parallel"

# Hyperbolic Configuration
# train_curv=False: curvature is not learnable (default)
export HYPERBOLIC_ARGS_STR="--use-lorentz-moe --use-hyperbolic --hyperbolic-curvature 1.0 --expert-curvature-min 0.1 --expert-curvature-max 2.0"

# Export settings for display
export NUM_EXPERTS=8
export NUM_SHARED_EXPERTS=1
export MOE_ROUTER_TOPK=2
export MOE_AUX_LOSS_COEFF=0.01
export HYPERBOLIC_CURVATURE=1.0
export EXPERT_CURVATURE_MIN=0.1
export EXPERT_CURVATURE_MAX=2.0

# Training Hyperparameters (HELM paper best)
# Global batch = 256 sequences. With DP=64: grad_accum = 256/64 = 4
export MICRO_BATCH_SIZE=1          # max_batch_size=1 from HELM
export GLOBAL_BATCH_SIZE=256       # gradient_accumulation_steps=256 from HELM
export LR=4e-4                     # lr from HELM
export MIN_LR=4e-5                 # 10x reduction
export MAX_STEPS=100000            # Long production run
export WARMUP_STEPS=2000
export WEIGHT_DECAY=0.01           # weight_decay from HELM
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=1000

# Checkpoint on shared filesystem
export CHECKPOINT_DIR="/fsx/karish/checkpoints/${MODEL_NAME}"

# Parallelism (16 nodes, 128 GPUs)
# TP=2: hidden_size=910/2=455 per rank; 14 heads / 2 = 7 per rank
# EP=8: 8 experts / 8 = 1 expert per EP rank
# DP=64: 128 / (TP=2 * PP=1) = 64
export TP=2
export PP=1
export EP=8
export GPUS_PER_NODE=8

# Container
# export CONTAINER_NAME="lorentz-moe"  # Uncomment if using saved container

# =============================================================================
# Run with Pyxis
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base/_train_moe_base_pyxis.sh"
