#!/bin/bash

#SBATCH --account=3dllms
#SBATCH --partition=mb-l40s
#SBATCH --job-name=llava-3d_test_finetune
#SBATCH --output=./slurm_logs/llava-3d_test_finetune%j.out
#SBATCH --error=./slurm_logs/llava-3d_test_finetune%j.out
#SBATCH --gres=gpu:5
#SBATCH --mem=144G
#SBATCH --time=7-00:00:00

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

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21
bash scripts/train/finetune.sh