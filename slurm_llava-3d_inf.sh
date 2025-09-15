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

# Ensure Python can import the top-level package regardless of cwd quirks
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

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

# Move to repo root directory
cd "$SCRIPT_DIR"

# Run updated inference script for LLaVA-3D
python llava-3d_inf.py "$@"