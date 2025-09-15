#!/bin/bash

#SBATCH --account=3dllms
#SBATCH --job-name=llava-3d_inf
#SBATCH --partition=mb-l40s
#SBATCH --nodes=1
#SBATCH --output=./slurm_logs/llava-3d_inf_%j.log
#SBATCH --error=./slurm_logs/llava-3d_inf_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00

# Fail fast and keep a clean environment
set -euo pipefail

# Ensure we run from the repo root, robustly resolving symlinks
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

echo "DEBUG: SCRIPT_PATH = $SCRIPT_PATH"
echo "DEBUG: SCRIPT_DIR = $SCRIPT_DIR"
echo "DEBUG: PWD = $(pwd)"

# Set a clean PYTHONPATH that only includes our repo directory
# This prevents /var/spool and other SLURM paths from polluting the path
export PYTHONPATH="$SCRIPT_DIR"
echo "Pythonpath after changes: $PYTHONPATH"

echo "[llava3d_inf] Working dir: $(pwd)" >&2
echo "[llava3d_inf] PYTHONPATH: $PYTHONPATH" >&2
ls -la "$SCRIPT_DIR/llava" >/dev/null 2>&1 || echo "[llava3d_inf][WARN] llava/ not found under $SCRIPT_DIR" >&2

#This ensures "conda activate <env>" works in non-interactive shells.
#(running "conda init" every time won't work.)
if [ -n "$CONDA_INSTALL_PATH" ]; then
    CONDA_SH=$CONDA_INSTALL_PATH/etc/profile.d/conda.sh
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
else
    CONDA_SH=/project/3dllms/melgin/conda/etc/profile.d/conda.sh
    echo "WARNING: CONDA_INSTALL_PATH is not set. Trying $CONDA_SH"
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
fi
# Now the activation should work
conda activate llava-3d

# Set PYTHONPATH after conda activation to ensure it doesn't get overridden
# Use explicit repo path to avoid any SLURM path issues
REPO_PATH="/project/3dllms/melgin/LLaVA-3D_for_UPD-3D"
export PYTHONPATH="$REPO_PATH"
echo "PYTHONPATH after conda activation: $PYTHONPATH"
echo "DEBUG: Using hardcoded repo path: $REPO_PATH"

# Move to repo root directory
cd "$REPO_PATH"
echo "DEBUG: Changed to directory: $(pwd)"

# Run updated inference script for LLaVA-3D
python llava-3d_inf.py "$@"