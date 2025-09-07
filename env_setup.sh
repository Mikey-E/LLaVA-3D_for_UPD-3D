#!/usr/bin/env bash

#https://github.com/ZCMax/LLaVA-3D #commit 11 (3c56223)

#Prerequisites:
	#Download and place the camera parameters file as described in the instructions.
	#Download this flash attention whl and put it in the top level directory:
		#https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.0.4
		#flash_attn-2.7.3+cu118torch2.1-cp310-cp310-linux_x86_64.whl
    #salloc an l40s gpu (others might also work)
        #salloc -A 3dllms --nodes=1 -G 1 -t 8:00:00 --mem=48G --partition=mb-l40s

# Lightweight colored step logger
STEP_COLOR="\033[1;36m"  # bold cyan
NC="\033[0m"
step() { printf "%b==> %s%b\n" "$STEP_COLOR" "$1" "$NC"; }

# Start dataset clean on every run
step "Removing demo dir if there"
rm -rf demo
step "Making demo dir"
mkdir -p demo
step "Entering demo dir"
cd demo
step "Loading gcc"
ml gcc/14.2.0
step "Loading git-lfs"
ml git-lfs
step "Cloning LLaVA-3D-Demo-Data dataset"
git clone https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Demo-Data .
step "Pulling LFS data"
git lfs pull
step "Purging modules"
module purge
step "Leaving demo dir"
cd ..

# If the current conda environment is already 'llava-3d', deactivate first
step "Deactivating current env if it's 'llava-3d'"
if [ "${CONDA_DEFAULT_ENV:-}" = "llava-3d" ]; then
    conda deactivate
fi

step "Removing conda env 'llava-3d'"
conda env remove -n llava-3d -y

step "Creating conda env 'llava-3d' with CUDA 11.8 and Python 3.10"
conda create -n llava-3d -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8.0 python=3.10 -y

step "Activating env 'llava-3d'"
conda activate llava-3d

step "Configuring CUDA env vars in activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/lmod.sh" <<'EOSH'
module purge
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
EOSH

step "Re-loading env to apply CUDA vars"
conda deactivate
conda activate llava-3d

#pyproject.toml gets modified
step "Writing pyproject.toml"
cat > ./pyproject.toml <<'PYPROJECT'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "llava-3d"
version = "1.0.0"
description = "A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness."
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
]
dependencies = [
    # Pins to keep Torch stack at 2.1.0 with CUDA 11.8 (cu118)
    # Note: These wheels are hosted on the PyTorch index. Ensure installs use:
    #   --index-url https://download.pytorch.org/whl/cu118
    "torch==2.1.0+cu118", "torchvision==0.16.0+cu118", "torchaudio==2.1.0+cu118",
    #Rest of your deps (unchanged)
    "transformers==4.37.2", "tokenizers==0.15.1", "sentencepiece==0.1.99", "shortuuid",
    "accelerate==0.21.0", "peft==0.8.2", "bitsandbytes",
    "pydantic", "markdown2[all]", "numpy", "scikit-learn==1.2.2",
    "gradio==4.29.0", "gradio_client==0.16.1",
    "requests", "httpx==0.27.0", "uvicorn", "fastapi",
    "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13", "opencv-python", "protobuf"
]

[project.optional-dependencies]
train = ["deepspeed==0.12.6", "ninja", "wandb"]
build = ["build", "twine"]
pyg = []

[tool.setuptools.packages.find]
exclude = ["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]

[tool.wheel]
exclude = ["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]
PYPROJECT

step "Installing project (editable) via PyTorch cu118 index"
pip install -e . --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple

step "Installing torch-scatter (torch 2.1.0 + cu118)"
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

step "Installing training extras"
pip install -e ".[train]"

step "Installing FlashAttention wheel (cu118, torch 2.1)"
pip install ./flash_attn-2.7.3+cu118torch2.1-cp310-cp310-linux_x86_64.whl