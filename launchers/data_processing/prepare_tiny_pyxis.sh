#!/bin/bash
# =============================================================================
# Tiny RedPajama Dataset: Download + Preprocess (Pyxis/Enroot)
# =============================================================================
# Downloads 15k Wikipedia samples from RedPajama via HuggingFace, then
# preprocesses into Megatron binary format inside a Pyxis container.
#
# This creates enough data for the "tiny" training preset (10k samples).
#
# Prerequisites:
#   - Python with 'datasets' library on login/current node (for download)
#   - Active Slurm allocation with at least 1 GPU (for preprocessing)
#
# Usage:
#   # Option 1: Inside an salloc (recommended)
#   salloc --account=mrs_2 --qos=h200_mrs_2_high --time=1:00:00 \
#          --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
#          --cpus-per-task=10 --mem=0
#   bash launchers/data_processing/prepare_tiny_pyxis.sh
#
#   # Option 2: Just run from login node (srun will wait for allocation)
#   bash launchers/data_processing/prepare_tiny_pyxis.sh
#
# Output:
#   /fsx/karish/data/processed_data/redpajama_tiny/tiny_text_document.bin
#   /fsx/karish/data/processed_data/redpajama_tiny/tiny_text_document.idx
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

# Where to store data on the shared filesystem
DATA_ROOT="${DATA_ROOT:-/fsx/karish/data}"
RAW_DIR="${DATA_ROOT}/redpajama-1t/wikipedia"
PROCESSED_DIR="${DATA_ROOT}/processed_data/redpajama_tiny"

# Megatron source directory
MEGATRON_DIR="${MEGATRON_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Container settings
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.04-py3}"
# Use saved container if available (has transformers pre-installed)
# export CONTAINER_NAME="lorentz-moe"

# Download settings
export NUM_SAMPLES=${NUM_SAMPLES:-15000}  # Download a bit more than 10k for buffer

# Preprocessing settings
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-8B}"
WORKERS=${WORKERS:-8}

# Slurm settings for srun (used for preprocessing step)
SLURM_ACCOUNT="${SLURM_ACCOUNT:-mrs_2}"
SLURM_QOS="${SLURM_QOS:-h200_mrs_2_high}"

# Enroot workaround: /tmp is overlay (can't create device nodes for whiteouts)
# and Lustre is too slow for metadata-heavy extraction.
# Use /dev/shm (RAM-backed tmpfs) which is fast and supports all fs operations.
export TMPDIR="${TMPDIR:-/dev/shm}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# =============================================================================
# Step 1: Download Raw Data (runs on current node, no container needed)
# =============================================================================

log "=============================================="
log "Tiny RedPajama: Download + Preprocess"
log "=============================================="
log "Data root: ${DATA_ROOT}"
log "Megatron dir: ${MEGATRON_DIR}"
log "Samples to download: ${NUM_SAMPLES}"
log "Tokenizer: ${TOKENIZER_MODEL}"
log "=============================================="

mkdir -p "$RAW_DIR" "$PROCESSED_DIR"

export RAW_FILE="${RAW_DIR}/wiki.jsonl"

if [ -f "$RAW_FILE" ]; then
    EXISTING_LINES=$(wc -l < "$RAW_FILE")
    log "Raw data already exists: ${RAW_FILE} (${EXISTING_LINES} lines)"
    if [ "$EXISTING_LINES" -ge 10000 ]; then
        log "Sufficient samples, skipping download."
    else
        log "Not enough samples, re-downloading..."
        rm -f "$RAW_FILE"
    fi
fi

if [ ! -f "$RAW_FILE" ]; then
    log "[Step 1/3] Downloading ${NUM_SAMPLES} Wikipedia samples from RedPajama..."

    log "Downloading from https://data.together.xyz/redpajama-data-1T/v1.0.0/wikipedia/wiki.jsonl ..."
    wget -q --show-progress \
        'https://data.together.xyz/redpajama-data-1T/v1.0.0/wikipedia/wiki.jsonl' \
        -O "$RAW_FILE"

    if [ $? -ne 0 ]; then
        log "ERROR: Download failed."
        exit 1
    fi

    log "Download complete: $(wc -l < "$RAW_FILE") samples"
else
    log "[Step 1/3] Skipping download (data exists)."
fi

# =============================================================================
# Step 2: Preprocess inside Container (via srun + Pyxis)
# =============================================================================

OUTPUT_BIN="${PROCESSED_DIR}/tiny_text_document.bin"
OUTPUT_IDX="${PROCESSED_DIR}/tiny_text_document.idx"

if [ -f "$OUTPUT_BIN" ] && [ -f "$OUTPUT_IDX" ]; then
    log "[Step 2/3] Skipping preprocessing (output files exist)."
    log "  ${OUTPUT_BIN}"
    log "  ${OUTPUT_IDX}"
else
    log "[Step 2/3] Preprocessing inside container..."

    # Build container flags
    if [ -n "$CONTAINER_NAME" ]; then
        CONTAINER_FLAGS="--container-name=${CONTAINER_NAME}"
        PIP_CMD=""
    else
        CONTAINER_FLAGS="--container-image=${CONTAINER_IMAGE}"
        # Need to install transformers for the tokenizer if using base NGC image
        PIP_CMD="pip install --quiet transformers && "
    fi

    # Mount data and Megatron source into the container.
    # The noop_prolog overrides the cluster's task_prolog.sh which fails inside containers.
    NOOP_PROLOG="/fsx/karish/enroot/noop_prolog.sh"
    MOUNTS="${MEGATRON_DIR}:/workspace/megatron,${DATA_ROOT}:/workspace/data"
    MOUNTS="${MOUNTS},${NOOP_PROLOG}:/etc/slurm/other-scripts/task_prolog.sh:ro"

    srun --ntasks=1 \
        ${CONTAINER_FLAGS} \
        --container-mounts="${MOUNTS}" \
        --container-workdir=/workspace/megatron \
        bash -c "
            export PYTHONPATH=/workspace/megatron:\${PYTHONPATH}

            ${PIP_CMD}

            echo 'Extracting 10000 samples from wiki.jsonl...'
            head -n 10000 /workspace/data/redpajama-1t/wikipedia/wiki.jsonl \
                > /workspace/data/processed_data/redpajama_tiny/tiny_subset.jsonl

            echo 'Running Megatron preprocessing...'
            python tools/preprocess_data.py \
                --input /workspace/data/processed_data/redpajama_tiny/tiny_subset.jsonl \
                --output-prefix /workspace/data/processed_data/redpajama_tiny/tiny \
                --tokenizer-type HuggingFaceTokenizer \
                --tokenizer-model ${TOKENIZER_MODEL} \
                --workers ${WORKERS} \
                --partitions 1 \
                --append-eod \
                --json-keys text

            echo 'Preprocessing complete!'
        "

    if [ $? -ne 0 ]; then
        log "ERROR: Preprocessing failed."
        exit 1
    fi
fi

# =============================================================================
# Step 3: Validate and Create Blend File
# =============================================================================

log "[Step 3/3] Validating output..."

if [ -f "$OUTPUT_BIN" ] && [ -f "$OUTPUT_IDX" ]; then
    log "Output files:"
    ls -lh "${PROCESSED_DIR}"/tiny_text_document.*

    # Create data blend file
    BLEND_FILE="${PROCESSED_DIR}/data_blend.txt"
    echo "1.0 ${PROCESSED_DIR}/tiny_text_document" > "$BLEND_FILE"

    log ""
    log "=============================================="
    log "Tiny Dataset Ready!"
    log "=============================================="
    log ""
    log "Processed data:"
    log "  ${OUTPUT_BIN}"
    log "  ${OUTPUT_IDX}"
    log ""
    log "To train with this data, run:"
    log ""
    log "  # Single-node Lorentz MoE (within salloc):"
    log "  export CONTAINER_NAME=\"lorentz-moe\"  # if setup was run"
    log "  bash launchers/training/lorentz/single-node/train_moe_80m_redpajama-small_pyxis.sh"
    log ""
    log "  # Or submit multi-node:"
    log "  sbatch launchers/training/lorentz/multi-node/train_moe_80m_redpajama-small_2node_pyxis.sh"
    log ""
    log "=============================================="
else
    log "ERROR: Output files not found!"
    log "Expected:"
    log "  ${OUTPUT_BIN}"
    log "  ${OUTPUT_IDX}"
    exit 1
fi
