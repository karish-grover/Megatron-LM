#!/bin/bash
# =============================================================================
# Setup Pyxis/Enroot Container for Lorentz MoE Training
# =============================================================================
# This script pulls the NGC PyTorch base image and installs additional Python
# packages, saving the result as a named enroot container for fast reuse.
#
# This is the Pyxis/Enroot equivalent of build_lorentz_moe_image.sh (Docker).
# Run this ONCE before training. Subsequent training runs can reference the
# saved container by name, avoiding repeated image pulls and pip installs.
#
# Prerequisites:
#   - Active Slurm allocation with at least 1 GPU
#   - Pyxis and Enroot installed on the cluster
#
# Usage:
#   # Option 1: Within an existing salloc
#   salloc --account=mrs_2 --qos=h200_mrs_2_high --time=0:30:00 \
#          --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
#          --cpus-per-task=10 --mem=0
#   bash launchers/setup/setup_pyxis_container.sh
#
#   # Option 2: As a standalone sbatch job
#   sbatch launchers/setup/setup_pyxis_container.sh
#
# After setup, use the container in training scripts:
#   export CONTAINER_NAME="lorentz-moe"
#   bash launchers/training/lorentz/single-node/train_moe_80m_redpajama-small_pyxis.sh
# =============================================================================
#SBATCH --job-name=setup-lorentz-moe
#SBATCH --account=mrs_2
#SBATCH --qos=h200_mrs_2_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --output=logs/setup-lorentz-moe-%j.out
#SBATCH --error=logs/setup-lorentz-moe-%j.err

set -e

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"nvcr.io#nvidia/pytorch:25.04-py3"}
CONTAINER_NAME=${CONTAINER_NAME:-"lorentz-moe"}

# Enroot workaround: /tmp is overlay (can't create device nodes for whiteouts)
# and Lustre is too slow for extraction. Use /dev/shm (RAM-backed tmpfs).
export TMPDIR="${TMPDIR:-/dev/shm}"

echo "=============================================="
echo "Setting up Pyxis/Enroot Container"
echo "=============================================="
echo "Base image: ${CONTAINER_IMAGE}"
echo "Container name: ${CONTAINER_NAME}"
echo "=============================================="

mkdir -p logs

# Pull the image and install additional packages, saving as a named container.
# --container-name saves the container state after the command finishes,
# so pip-installed packages persist for future runs.
echo ""
echo "[Step 1/2] Pulling image and installing dependencies..."

srun --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts="/fsx/karish/enroot/noop_prolog.sh:/etc/slurm/other-scripts/task_prolog.sh:ro" \
    bash -c '
        echo "Installing Python dependencies..."
        pip install --no-cache-dir \
            transformers \
            wandb \
            tensorboard
        pip cache purge

        echo ""
        echo "Verifying installation..."
        python -c "import torch; print(f\"PyTorch: {torch.__version__}\")"
        python -c "import transformer_engine; print(f\"TransformerEngine: {transformer_engine.__version__}\")"
        python -c "import transformers; print(f\"Transformers: {transformers.__version__}\")"
        echo ""
        echo "All packages verified."
    '

echo ""
echo "[Step 2/2] Verifying saved container..."

srun --ntasks=1 \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts="/fsx/karish/enroot/noop_prolog.sh:/etc/slurm/other-scripts/task_prolog.sh:ro" \
    python -c "
import torch, transformer_engine, transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'TransformerEngine: {transformer_engine.__version__}')
print(f'Transformers: {transformers.__version__}')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo "Container name: ${CONTAINER_NAME}"
echo ""
echo "To use in training scripts, either:"
echo "  1. Set environment variable before running:"
echo "     export CONTAINER_NAME=${CONTAINER_NAME}"
echo "     bash launchers/training/lorentz/single-node/train_moe_80m_redpajama-small_pyxis.sh"
echo ""
echo "  2. Or uncomment the CONTAINER_NAME line in the training config script."
echo "=============================================="
