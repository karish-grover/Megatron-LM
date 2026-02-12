#!/bin/bash
# =============================================================================
# RedPajama Full Dataset Preprocessing (Pyxis/Enroot)
# =============================================================================
# Processes all 10 RedPajama-1T sources into Megatron binary format inside
# a Pyxis container. Uses LLaMA 3.1-8B tokenizer.
#
# Prerequisites:
#   - RedPajama data downloaded to /fsx/karish/data/redpajama-1t/
#     (run: bash launchers/data_processing/download_redpajama.sh)
#   - Active Slurm allocation with GPU and 64+ CPUs
#
# Usage:
#   salloc --account=mrs_2 --qos=h200_mrs_2_high --time=24:00:00 \
#          --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
#          --cpus-per-task=64 --mem=0
#   bash launchers/data_processing/prepare_redpajama_pyxis.sh
#
#   # Process only specific sources:
#   SOURCES="wikipedia stackexchange" bash launchers/data_processing/prepare_redpajama_pyxis.sh
#
#   # Dry run:
#   DRY_RUN=1 bash launchers/data_processing/prepare_redpajama_pyxis.sh
#
# Output:
#   /fsx/karish/data/processed_data/redpajama/*.bin, *.idx
#   /fsx/karish/data/processed_data/redpajama/data_blend.txt
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT="${DATA_ROOT:-/fsx/karish/data}"
RAW_DIR="${DATA_ROOT}/redpajama-1t"
PROCESSED_DIR="${DATA_ROOT}/processed_data/redpajama"

MEGATRON_DIR="${MEGATRON_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Container settings
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.04-py3}"

# Tokenizer (LLaMA 3.1-8B to match HELM paper)
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-3.1-8B}"

# Processing settings
WORKERS=${WORKERS:-64}
PARTITIONS=${PARTITIONS:-16}

# Optional: filter to specific sources (space-separated)
SOURCES="${SOURCES:-}"
DRY_RUN="${DRY_RUN:-0}"

# HuggingFace token (needed for gated LLaMA model)
# Set this or run: huggingface-cli login
HF_TOKEN="${HF_TOKEN:-}"

# Enroot workaround
export TMPDIR="${TMPDIR:-/dev/shm}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# =============================================================================
# Data Source Definitions (same as preprocess_redpajama.sh)
# =============================================================================

# Declare sources: NAME -> INPUT_PATTERN (relative to raw dir, container path)
declare -A SOURCE_PATTERNS=(
    ["arxiv"]="/workspace/data/redpajama-1t/arxiv/*.jsonl"
    ["c4"]="/workspace/data/redpajama-1t/c4/*.jsonl"
    ["github"]="/workspace/data/redpajama-1t/github/*.jsonl"
    ["stackexchange"]="/workspace/data/redpajama-1t/stackexchange/*.jsonl"
    ["wikipedia"]="/workspace/data/redpajama-1t/wikipedia/*.jsonl"
    ["common_crawl_2019-30"]="/workspace/data/redpajama-1t/common_crawl/2019-30/*.zst"
    ["common_crawl_2020-05"]="/workspace/data/redpajama-1t/common_crawl/2020-05/*.zst"
    ["common_crawl_2021-04"]="/workspace/data/redpajama-1t/common_crawl/2021-04/*.zst"
    ["common_crawl_2022-05"]="/workspace/data/redpajama-1t/common_crawl/2022-05/*.zst"
    ["common_crawl_2023-06"]="/workspace/data/redpajama-1t/common_crawl/2023-06/*.zst"
)

# Weights for data blend (sum to 1.0)
declare -A SOURCE_WEIGHTS=(
    ["arxiv"]="0.025"
    ["c4"]="0.15"
    ["github"]="0.045"
    ["stackexchange"]="0.02"
    ["wikipedia"]="0.02"
    ["common_crawl_2019-30"]="0.15"
    ["common_crawl_2020-05"]="0.15"
    ["common_crawl_2021-04"]="0.15"
    ["common_crawl_2022-05"]="0.15"
    ["common_crawl_2023-06"]="0.14"
)

# Host-side patterns for validation (relative to raw dir)
declare -A HOST_PATTERNS=(
    ["arxiv"]="${RAW_DIR}/arxiv/*.jsonl"
    ["c4"]="${RAW_DIR}/c4/*.jsonl"
    ["github"]="${RAW_DIR}/github/*.jsonl"
    ["stackexchange"]="${RAW_DIR}/stackexchange/*.jsonl"
    ["wikipedia"]="${RAW_DIR}/wikipedia/*.jsonl"
    ["common_crawl_2019-30"]="${RAW_DIR}/common_crawl/2019-30/*.zst"
    ["common_crawl_2020-05"]="${RAW_DIR}/common_crawl/2020-05/*.zst"
    ["common_crawl_2021-04"]="${RAW_DIR}/common_crawl/2021-04/*.zst"
    ["common_crawl_2022-05"]="${RAW_DIR}/common_crawl/2022-05/*.zst"
    ["common_crawl_2023-06"]="${RAW_DIR}/common_crawl/2023-06/*.zst"
)

# Determine which sources to process
if [ -n "$SOURCES" ]; then
    ACTIVE_SOURCES=($SOURCES)
else
    ACTIVE_SOURCES=(arxiv c4 github stackexchange wikipedia \
                    common_crawl_2019-30 common_crawl_2020-05 \
                    common_crawl_2021-04 common_crawl_2022-05 \
                    common_crawl_2023-06)
fi

# =============================================================================
# Print Configuration
# =============================================================================

log "=============================================="
log "RedPajama Full Dataset Preprocessing (Pyxis)"
log "=============================================="
log "Raw data: ${RAW_DIR}"
log "Output: ${PROCESSED_DIR}"
log "Megatron dir: ${MEGATRON_DIR}"
log "Tokenizer: ${TOKENIZER_MODEL}"
log "Workers: ${WORKERS}"
log "Partitions: ${PARTITIONS}"
log "Sources: ${ACTIVE_SOURCES[*]}"
log "Dry run: ${DRY_RUN}"
log "=============================================="

mkdir -p "$PROCESSED_DIR"

# =============================================================================
# Validate Raw Data Exists
# =============================================================================

log "Validating raw data..."
MISSING=()
for name in "${ACTIVE_SOURCES[@]}"; do
    pattern="${HOST_PATTERNS[$name]}"
    count=$(ls -1 $pattern 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        MISSING+=("$name")
        log "  WARNING: No files for ${name} (pattern: ${pattern})"
    else
        log "  OK: ${name} (${count} files)"
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    log ""
    log "WARNING: ${#MISSING[@]} sources have no raw data: ${MISSING[*]}"
    log "Run: bash launchers/data_processing/download_redpajama.sh"
    log "Continuing with available sources..."
    log ""
fi

# =============================================================================
# Build Processing Commands
# =============================================================================

# Build the processing script that runs inside the container
PROCESS_COMMANDS=""
PROCESSED_SOURCES=()

for name in "${ACTIVE_SOURCES[@]}"; do
    output_prefix="/workspace/data/processed_data/redpajama/${name}"
    output_bin="${PROCESSED_DIR}/${name}_text_document.bin"
    input_pattern="${SOURCE_PATTERNS[$name]}"

    # Skip if already processed
    if [ -f "$output_bin" ]; then
        log "Skipping ${name}: already processed"
        PROCESSED_SOURCES+=("$name")
        continue
    fi

    # Skip if no raw data
    host_pattern="${HOST_PATTERNS[$name]}"
    count=$(ls -1 $host_pattern 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        continue
    fi

    # Adjust partitions for small file counts
    use_partitions=$PARTITIONS
    if [ "$count" -lt "$use_partitions" ]; then
        use_partitions=$count
    fi
    # Ensure workers divisible by partitions
    use_workers=$WORKERS
    while [ $((use_workers % use_partitions)) -ne 0 ] && [ $use_partitions -gt 1 ]; do
        use_partitions=$((use_partitions - 1))
    done

    PROCESSED_SOURCES+=("$name")

    if [ "$DRY_RUN" = "1" ]; then
        log "[DRY-RUN] Would process: ${name} (${count} files, workers=${use_workers}, partitions=${use_partitions})"
        continue
    fi

    PROCESS_COMMANDS="${PROCESS_COMMANDS}
echo '========================================'
echo 'Processing: ${name} (${count} files)'
echo '========================================'
python tools/preprocess_data.py \\
    --input '${input_pattern}' \\
    --output-prefix '${output_prefix}' \\
    --tokenizer-type ${TOKENIZER_TYPE} \\
    --tokenizer-model ${TOKENIZER_MODEL} \\
    --workers ${use_workers} \\
    --partitions ${use_partitions} \\
    --append-eod \\
    --json-keys text \\
    2>&1 | tee /workspace/data/processed_data/redpajama/logs/${name}.log
echo '${name} processing complete!'
echo ''
"
done

if [ "$DRY_RUN" = "1" ]; then
    log "Dry run complete. No processing was done."
    exit 0
fi

# =============================================================================
# Run Processing Inside Container
# =============================================================================

if [ -z "$PROCESS_COMMANDS" ]; then
    log "All sources already processed. Skipping to blend file generation."
else
    log "Processing ${#PROCESSED_SOURCES[@]} sources inside container..."

    # Build container flags
    if [ -n "$CONTAINER_NAME" ]; then
        CONTAINER_FLAGS="--container-name=${CONTAINER_NAME}"
        PIP_CMD=""
    else
        CONTAINER_FLAGS="--container-image=${CONTAINER_IMAGE}"
        PIP_CMD="pip install --quiet transformers zstandard && "
    fi

    # Mounts
    NOOP_PROLOG="/fsx/karish/enroot/noop_prolog.sh"
    MOUNTS="${MEGATRON_DIR}:/workspace/megatron,${DATA_ROOT}:/workspace/data"
    MOUNTS="${MOUNTS},${NOOP_PROLOG}:/etc/slurm/other-scripts/task_prolog.sh:ro"

    # HF token for gated models
    HF_ENV=""
    if [ -n "$HF_TOKEN" ]; then
        HF_ENV="export HF_TOKEN=${HF_TOKEN} && "
    fi

    mkdir -p "${PROCESSED_DIR}/logs"

    srun --ntasks=1 \
        ${CONTAINER_FLAGS} \
        --container-mounts="${MOUNTS}" \
        --container-workdir=/workspace/megatron \
        bash -c "
            export PYTHONPATH=/workspace/megatron:\${PYTHONPATH}
            ${HF_ENV}
            ${PIP_CMD}

            # Ensure zstandard is available for .zst files
            python -c 'import zstandard' 2>/dev/null || pip install --quiet zstandard

            mkdir -p /workspace/data/processed_data/redpajama/logs

            ${PROCESS_COMMANDS}

            echo 'All sources processed!'
        "

    if [ $? -ne 0 ]; then
        log "ERROR: Processing failed."
        exit 1
    fi
fi

# =============================================================================
# Generate Data Blend File
# =============================================================================

BLEND_FILE="${PROCESSED_DIR}/data_blend.txt"
log "Generating data blend file: ${BLEND_FILE}"

> "$BLEND_FILE"
for name in "${ACTIVE_SOURCES[@]}"; do
    prefix="${PROCESSED_DIR}/${name}_text_document"
    weight="${SOURCE_WEIGHTS[$name]}"

    if [ -f "${prefix}.bin" ] && [ -f "${prefix}.idx" ]; then
        echo "${weight} ${prefix}" >> "$BLEND_FILE"
        log "  Added: ${weight} ${name}"
    else
        log "  Skipped: ${name} (files not found)"
    fi
done

# Also create a container-path version for training scripts
BLEND_FILE_CONTAINER="${PROCESSED_DIR}/data_blend_container.txt"
> "$BLEND_FILE_CONTAINER"
for name in "${ACTIVE_SOURCES[@]}"; do
    prefix="/workspace/data/processed_data/redpajama/${name}_text_document"
    weight="${SOURCE_WEIGHTS[$name]}"

    if [ -f "${PROCESSED_DIR}/${name}_text_document.bin" ]; then
        echo "${weight} ${prefix}" >> "$BLEND_FILE_CONTAINER"
    fi
done

# =============================================================================
# Summary
# =============================================================================

log ""
log "=============================================="
log "RedPajama Processing Complete!"
log "=============================================="
log ""
log "Output files:"
ls -lh ${PROCESSED_DIR}/*.bin 2>/dev/null | while read line; do log "  $line"; done
log ""
log "Blend file (host paths): ${BLEND_FILE}"
log "Blend file (container paths): ${BLEND_FILE_CONTAINER}"
log ""
cat "$BLEND_FILE"
log ""
log "To train with this data:"
log "  export DATA_BLEND_PATH=/workspace/data/processed_data/redpajama/data_blend_container.txt"
log "  bash launchers/training/lorentz/single-node/train_moe_1b_redpajama_pyxis.sh"
log "=============================================="
