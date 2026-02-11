#!/bin/bash
# =============================================================================
# Base Training Script: Standard Megatron MoE GPT - Pyxis/Enroot
# =============================================================================
# This is the base script - do not run directly.
# Use the config-specific launcher scripts.
#
# Uses Megatron's full pretrain infrastructure with:
# - Megatron's DistributedDataParallel (NOT PyTorch DDP)
# - Megatron's distributed optimizer
# - Expert parallelism via Megatron's MoE infrastructure
# - Pyxis/Enroot for containerized execution on Slurm
#
# Expected environment variables from launcher:
#   MODEL_NAME          - Name for checkpoints
#   MODEL_ARGS / MODEL_ARGS_STR   - Model architecture arguments
#   MOE_ARGS / MOE_ARGS_STR       - MoE configuration arguments
#   DATA_PATH           - Path to data inside container
#   HOST_DATA_DIR       - Host path to data directory
#   BATCH_SIZE, MICRO_BATCH_SIZE, LR, MAX_STEPS, etc.
#
# Container variables:
#   CONTAINER_IMAGE     - Docker/NGC image URI (default: NGC PyTorch 25.04)
#   CONTAINER_NAME      - Saved enroot container name (overrides CONTAINER_IMAGE)
# =============================================================================

set -e

# =============================================================================
# Validate Required Variables
# =============================================================================

if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: This is a base script. Use a config-specific launcher."
    exit 1
fi

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${MEGATRON_DIR}/checkpoints/${MODEL_NAME}"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${MEGATRON_DIR}/tensorboard/${MODEL_NAME}"}
mkdir -p "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

# =============================================================================
# Slurm/GPU Configuration
# =============================================================================

NUM_NODES=${SLURM_NNODES:-${NUM_NODES:-1}}

if [ -n "$SLURM_JOB_NODELIST" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
    MASTER_ADDR=${MASTER_ADDR:-localhost}
fi
MASTER_PORT=${MASTER_PORT:-29500}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))

# =============================================================================
# Container Configuration (Pyxis/Enroot)
# =============================================================================

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"nvcr.io#nvidia/pytorch:25.04-py3"}

# Enroot workaround: /tmp is overlay (can't create device nodes for whiteouts)
# and Lustre is too slow for extraction. Use /dev/shm (RAM-backed tmpfs).
export TMPDIR="${TMPDIR:-/dev/shm}"

if [ -n "$CONTAINER_NAME" ]; then
    CONTAINER_FLAGS="--container-name=${CONTAINER_NAME}"
else
    CONTAINER_FLAGS="--container-image=${CONTAINER_IMAGE}"
fi

# =============================================================================
# Parallelism Configuration (Megatron-style)
# =============================================================================

TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-1}

# =============================================================================
# Training Arguments (Megatron-style)
# =============================================================================

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((BATCH_SIZE * WORLD_SIZE))}

TRAINING_ARGS=(
    # Batch configuration
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE:-1}"

    # Optimizer
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr "${LR:-3e-4}"
    --min-lr "${MIN_LR:-3e-5}"
    --lr-decay-style cosine
    --lr-warmup-iters "${WARMUP_STEPS:-100}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --clip-grad "${GRAD_CLIP:-1.0}"

    # Training duration
    --train-iters "${MAX_STEPS:-1000}"
    --eval-interval "${EVAL_INTERVAL:-1000}"
    --eval-iters "${EVAL_ITERS:-10}"

    # Precision
    --bf16

    # Parallelism
    --tensor-model-parallel-size "${TP}"
    --pipeline-model-parallel-size "${PP}"
    --expert-model-parallel-size "${EP}"

    # Checkpointing
    --save "${CHECKPOINT_DIR}"
    --save-interval "${SAVE_INTERVAL:-500}"

    # Logging
    --log-interval "${LOG_INTERVAL:-10}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --log-throughput

    # Use Megatron's distributed optimizer
    --use-distributed-optimizer
)

# Add data path if provided, otherwise use mock data
if [ -n "$DATA_PATH" ]; then
    DATA_CACHE_DIR="${MEGATRON_DIR}/data_cache/${MODEL_NAME}"
    mkdir -p "$DATA_CACHE_DIR"
    TRAINING_ARGS+=(
        --data-path "$DATA_PATH"
        --split "949,50,1"
        --data-cache-path "$DATA_CACHE_DIR"
    )
else
    TRAINING_ARGS+=(
        --mock-data
    )
fi

# =============================================================================
# Build Combined Arguments
# =============================================================================

if [ -n "$MODEL_ARGS_STR" ]; then
    ALL_ARGS="${MODEL_ARGS_STR} ${MOE_ARGS_STR} ${TRAINING_ARGS[*]}"
else
    ALL_ARGS="${MODEL_ARGS[*]} ${MOE_ARGS[*]} ${TRAINING_ARGS[*]}"
fi

# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================="
echo "Standard Megatron MoE GPT Training"
echo "Pyxis/Enroot Mode"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Data: ${DATA_PATH:-'mock data'}"
echo "Nodes: $NUM_NODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $WORLD_SIZE"
echo "Parallelism: TP=$TP, PP=$PP, EP=$EP"
echo "Batch: ${GLOBAL_BATCH_SIZE} global (micro: ${MICRO_BATCH_SIZE:-1})"
echo "Steps: ${MAX_STEPS:-1000} (warmup: ${WARMUP_STEPS:-100})"
echo "LR: ${LR:-3e-4} -> ${MIN_LR:-3e-5}"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Container: ${CONTAINER_NAME:-$CONTAINER_IMAGE}"
echo "=============================================="
echo "MoE Settings:"
echo "  Experts: ${NUM_EXPERTS:-8}"
echo "  Activated: ${MOE_ROUTER_TOPK:-2}"
echo "  Layer freq: ${MOE_LAYER_FREQ:-2}"
echo "  Aux loss coeff: ${MOE_AUX_LOSS_COEFF:-0.01}"
echo "  Expert type: ${EXPERT_TYPE:-TEGroupedMLP}"
echo "=============================================="

# =============================================================================
# Build Container Mounts
# =============================================================================

NOOP_PROLOG="/fsx/karish/enroot/noop_prolog.sh"
CONTAINER_MOUNTS="${MEGATRON_DIR}:/workspace/megatron"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${NOOP_PROLOG}:/etc/slurm/other-scripts/task_prolog.sh:ro"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${CHECKPOINT_DIR}:${CHECKPOINT_DIR}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${TENSORBOARD_DIR}:${TENSORBOARD_DIR}"

if [ -n "$HOST_DATA_DIR" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${HOST_DATA_DIR}:/workspace/data"
fi

if [ -n "$DATA_CACHE_DIR" ]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DATA_CACHE_DIR}:${DATA_CACHE_DIR}"
fi

# =============================================================================
# Run Training via srun + Pyxis
# =============================================================================

echo "Starting Standard MoE training via Pyxis..."

srun --ntasks-per-node=1 \
    ${CONTAINER_FLAGS} \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir=/workspace/megatron \
    bash -c "
        export PYTHONPATH=/workspace/megatron:\${PYTHONPATH}
        export CUDA_DEVICE_MAX_CONNECTIONS=1

        torchrun \
            --nproc_per_node ${GPUS_PER_NODE} \
            --nnodes ${NUM_NODES} \
            --node_rank \${SLURM_NODEID:-0} \
            --master_addr ${MASTER_ADDR} \
            --master_port ${MASTER_PORT} \
            pretrain_gpt.py \
            ${ALL_ARGS}
    "

echo "=============================================="
echo "Training completed!"
echo "=============================================="
