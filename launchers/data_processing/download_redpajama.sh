#!/bin/bash
# =============================================================================
# RedPajama-1T Full Dataset Download Script (Parallel)
# =============================================================================
# Downloads the complete RedPajama-1T dataset (~4.4 TB) from together.xyz
# using parallel wget processes for much faster throughput.
#
# Usage:
#   # From login node (parallel downloads):
#   nohup bash launchers/data_processing/download_redpajama.sh > /fsx/karish/data/download.log 2>&1 &
#   tail -f /fsx/karish/data/download.log
#
#   # From compute node (faster network, recommended):
#   salloc --account=mrs_2 --qos=h200_mrs_2_high --time=24:00:00 \
#          --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=0
#   bash launchers/data_processing/download_redpajama.sh
#
#   # Download only specific sources:
#   SOURCES="wikipedia stackexchange c4" bash launchers/data_processing/download_redpajama.sh
#
#   # Adjust parallelism (default: 16 concurrent downloads):
#   PARALLEL=32 bash launchers/data_processing/download_redpajama.sh
#
# Output: /fsx/karish/data/redpajama-1t/ with the standard directory structure
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT="${DATA_ROOT:-/fsx/karish/data}"
DATASET_DIR="${DATA_ROOT}/redpajama-1t"
BASE_URL="https://data.together.xyz/redpajama-data-1T/v1.0.0"

# Number of parallel downloads (16 is a good balance)
PARALLEL="${PARALLEL:-16}"

# Optional: filter to specific sources (space-separated)
# Valid: arxiv c4 github stackexchange wikipedia common_crawl
SOURCES="${SOURCES:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

mkdir -p "$DATASET_DIR"

# =============================================================================
# Step 1: Download URL Manifest
# =============================================================================

URLS_FILE="${DATASET_DIR}/urls.txt"

if [ ! -f "$URLS_FILE" ]; then
    log "Downloading URL manifest..."
    wget -q "${BASE_URL}/urls.txt" -O "$URLS_FILE"
fi
log "URL manifest: $(wc -l < "$URLS_FILE") total URLs"

# =============================================================================
# Step 2: Filter URLs (if SOURCES specified)
# =============================================================================

if [ -n "$SOURCES" ]; then
    FILTERED_FILE="${DATASET_DIR}/urls_filtered.txt"
    > "$FILTERED_FILE"
    for source in $SOURCES; do
        grep -i "$source" "$URLS_FILE" >> "$FILTERED_FILE" || true
    done
    DOWNLOAD_FILE="$FILTERED_FILE"
    log "Filtered to $(wc -l < "$DOWNLOAD_FILE") URLs for sources: $SOURCES"
else
    DOWNLOAD_FILE="$URLS_FILE"
fi

# =============================================================================
# Step 3: Build list of URLs that still need downloading
# =============================================================================

PENDING_FILE="${DATASET_DIR}/.pending_urls.txt"
> "$PENDING_FILE"

TOTAL=$(wc -l < "$DOWNLOAD_FILE")
SKIPPED=0

while IFS= read -r url; do
    rel_path="${url#${BASE_URL}/}"
    local_path="${DATASET_DIR}/${rel_path}"

    # Skip if file already exists and is non-empty
    if [ -f "$local_path" ] && [ -s "$local_path" ]; then
        SKIPPED=$((SKIPPED + 1))
    else
        echo "$url" >> "$PENDING_FILE"
    fi
done < "$DOWNLOAD_FILE"

PENDING=$(wc -l < "$PENDING_FILE")

log "=============================================="
log "RedPajama-1T Parallel Download"
log "=============================================="
log "Target: ${DATASET_DIR}"
log "Total URLs: ${TOTAL}"
log "Already downloaded: ${SKIPPED}"
log "Remaining: ${PENDING}"
log "Parallel workers: ${PARALLEL}"
log "=============================================="

if [ "$PENDING" -eq 0 ]; then
    log "All files already downloaded!"
    rm -f "$PENDING_FILE"
    exit 0
fi

# =============================================================================
# Step 4: Download in parallel using xargs
# =============================================================================

# Function that downloads a single URL (called by xargs)
export DATASET_DIR BASE_URL
download_one() {
    local url="$1"
    local rel_path="${url#${BASE_URL}/}"
    local local_path="${DATASET_DIR}/${rel_path}"
    local local_dir="$(dirname "$local_path")"

    mkdir -p "$local_dir"

    if wget -q --tries=5 --timeout=120 --continue "$url" -O "$local_path" 2>/dev/null; then
        echo "[OK] $rel_path"
    else
        echo "[FAIL] $rel_path"
        rm -f "$local_path"  # Remove partial downloads
        return 1
    fi
}
export -f download_one

log "Starting ${PARALLEL} parallel downloads..."

# xargs runs PARALLEL concurrent download_one processes
# --halt soon,fail=20%: stop if 20%+ of downloads fail
cat "$PENDING_FILE" | xargs -P "$PARALLEL" -I {} bash -c 'download_one "$@"' _ {} 2>&1 | \
    while IFS= read -r line; do
        echo "[$(date '+%H:%M:%S')] $line"
    done

RESULT=${PIPESTATUS[0]}

# =============================================================================
# Step 5: Verify and Report
# =============================================================================

# Count what we have now
DOWNLOADED_COUNT=0
MISSING_COUNT=0
while IFS= read -r url; do
    rel_path="${url#${BASE_URL}/}"
    local_path="${DATASET_DIR}/${rel_path}"
    if [ -f "$local_path" ] && [ -s "$local_path" ]; then
        DOWNLOADED_COUNT=$((DOWNLOADED_COUNT + 1))
    else
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done < "$DOWNLOAD_FILE"

rm -f "$PENDING_FILE"

log ""
log "=============================================="
log "Download Summary"
log "=============================================="
log "Total files: ${TOTAL}"
log "Successfully downloaded: ${DOWNLOADED_COUNT}"
log "Missing/failed: ${MISSING_COUNT}"
log ""
log "Disk usage:"
du -sh "${DATASET_DIR}/" 2>/dev/null
du -sh "${DATASET_DIR}"/*/ 2>/dev/null
log ""

if [ "$MISSING_COUNT" -gt 0 ]; then
    log "WARNING: ${MISSING_COUNT} files missing. Re-run this script to retry."
    log "=============================================="
    exit 1
else
    log "All files downloaded successfully!"
    log ""
    log "Next step: process the data"
    log "  bash launchers/data_processing/prepare_redpajama_pyxis.sh"
    log "=============================================="
fi
