#!/bin/bash
# =============================================================================
# Lorentz MoE GPT ~1B: HELM-MiCE (Pyxis/Enroot) — Single-Node Testing
# =============================================================================
# 1B parameter HELM-MiCE model matching the HELM paper's best hyperparameters.
# Single-node config for testing before scaling to multi-node.
#
# Usage (within an salloc allocation):
#   salloc --account=mrs_2 --qos=h200_mrs_2_high --time=4:00:00 \
#          --nodes=1 --ntasks-per-node=1 --gpus-per-node=8 \
#          --cpus-per-task=80 --mem=0
#   bash launchers/training/lorentz/single-node/train_moe_1b_redpajama_pyxis.sh
#
# Expert Type: LorentzTEGroupedMLP
#   - Lorentz geometry via tangent space approximation
#   - TransformerEngine's efficient grouped linear operations
#   - Per-expert curvatures distributed across range
#
# Architecture (from HELM paper best hyperparameters):
#   - 16 layers (1 dense + 15 MoE), hidden_size 910, ffn 3640
#   - 14 attention heads, MLA (kv_lora_rank=257)
#   - 8 routed experts + 1 shared expert, top-2 routing
#   - Per-expert FFN: 1820
#   - Sequence length: 2048
# =============================================================================

set -e

# =============================================================================
# Data Configuration
# =============================================================================
# For full RedPajama blend (after running prepare_redpajama_pyxis.sh):
#   export DATA_BLEND_PATH="/workspace/data/processed_data/redpajama/data_blend_container.txt"
# For tiny dataset (testing):
#   export DATA_PATH="/workspace/data/processed_data/redpajama_tiny/tiny_text_document"

HOST_DATA_DIR="${HOST_DATA_DIR:-/fsx/karish/data}"
export HOST_DATA_DIR

# Check for blend file first, then single dataset, then fall back to mock
if [ -f "${HOST_DATA_DIR}/processed_data/redpajama/data_blend_container.txt" ]; then
    export DATA_BLEND_PATH="${DATA_BLEND_PATH:-/workspace/data/processed_data/redpajama/data_blend_container.txt}"
    echo "Using full RedPajama blend: ${DATA_BLEND_PATH}"
elif [ -f "${HOST_DATA_DIR}/processed_data/redpajama_tiny/tiny_text_document.bin" ]; then
    export DATA_PATH="${DATA_PATH:-/workspace/data/processed_data/redpajama_tiny/tiny_text_document}"
    echo "Using tiny dataset: ${DATA_PATH}"
else
    echo "WARNING: No processed data found. Using mock data."
    echo "Run: bash launchers/data_processing/prepare_redpajama_pyxis.sh"
fi

# =============================================================================
# Export Config for Base Script
# =============================================================================
export MODEL_NAME="lorentz-moe-1B-helm-mice"
export EXPERT_TYPE="LorentzTEGroupedMLP"

# Model Architecture (~1B params, HELM paper best hyperparameters)
export MODEL_ARGS=(
    --num-layers 16
    --hidden-size 910
    --num-attention-heads 14
    --ffn-hidden-size 3640
    --seq-length 2048
    --max-position-embeddings 2048
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --no-bias-dropout-fusion
    --no-persist-layer-norm
    --untie-embeddings-and-output-weights
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model meta-llama/Llama-3.1-8B
    # MLA (Multi-head Latent Attention) settings
    --kv-lora-rank 257
    --qk-head-dim 65
    --qk-pos-emb-head-dim 65
    --v-head-dim 65
)

# MoE Configuration (Megatron-style)
# n_dense_layers=1: first layer dense, rest MoE -> moe-layer-freq pattern
export MOE_ARGS=(
    --num-experts 8
    --moe-ffn-hidden-size 1820
    --moe-shared-expert-intermediate-size 1820
    --moe-router-topk 2
    --moe-layer-freq "([0]+[1]*15)"
    --moe-aux-loss-coeff 0.01
    --moe-token-dispatcher-type allgather
    --moe-grouped-gemm
)

# Hyperbolic Configuration (Lorentz-specific)
# train_curv=False: curvature is NOT learnable (default)
export HYPERBOLIC_ARGS=(
    --use-lorentz-moe
    --use-hyperbolic
    --hyperbolic-curvature 1.0
    --expert-curvature-min 0.1
    --expert-curvature-max 2.0
)

# Export settings for display
export NUM_EXPERTS=8
export NUM_SHARED_EXPERTS=1
export MOE_ROUTER_TOPK=2
export MOE_AUX_LOSS_COEFF=0.01
export HYPERBOLIC_CURVATURE=1.0
export EXPERT_CURVATURE_MIN=0.1
export EXPERT_CURVATURE_MAX=2.0

# Training Hyperparameters (HELM paper best)
export BATCH_SIZE=4                # Per-node contribution (unused, overridden by GLOBAL_BATCH_SIZE)
export MICRO_BATCH_SIZE=1          # max_batch_size=1 from HELM
export GLOBAL_BATCH_SIZE=256       # gradient_accumulation_steps=256 from HELM
export LR=4e-4                     # lr from HELM
export MIN_LR=4e-5                 # 10x reduction
export MAX_STEPS=10000             # Adjust based on dataset size
export WARMUP_STEPS=500
export WEIGHT_DECAY=0.01           # weight_decay from HELM
export GRAD_CLIP=1.0
export LOG_INTERVAL=10
export SAVE_INTERVAL=500

# Parallelism (single-node, 8 GPUs)
# TP=2 for hidden_size=910 (910/2=455); 14 heads / 2 = 7 per rank
export TP=2
export PP=1
export EP=4                        # 8 experts / 4 = 2 experts per EP rank
export GPUS_PER_NODE=8

# Container
# export CONTAINER_NAME="lorentz-moe"  # Uncomment if using saved container

# =============================================================================
# Run
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../base/_train_moe_base_pyxis.sh"
