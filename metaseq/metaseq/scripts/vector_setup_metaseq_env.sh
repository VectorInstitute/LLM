#!/bin/bash

# Description:
#	Script to automatically build python environment for use with metaseq.
# Usage:
#	1. git clone metaseq repo
#	2. Create python environment
#		2.1. Create folder to hold git repo dependencies (if needed)
#	3. Launch script with the path to the activate file, the folder to
#	   place the git repo dependencies, and the metaseq git repo.
# Note: Required CUDA version is 11.1 for installing both pytorch and apex.

ENV_PATH=$1
EXTERNAL_REPO_PATH=$2
METASEQ_REPO_PATH=$3

if [ ! -d "$ENV_PATH" ]; then
	echo ""$ENV_PATH" not found: Please specify a valid env."
	exit 1
fi

if [ ! -d "$METASEQ_REPO_PATH" ]; then
	echo ""$METASEQ_REPO_PATH" not found. Please specify a valid metaseq repo."
	exit 1
fi

if [ ! -d "$EXTERNAL_REPO_PATH" ]; then
	echo ""$EXTERNAL_REPO_PATH" not found, creating it..."
	mkdir "$EXTERNAL_REPO_PATH"
fi


#ENV_PATH="/h/mchoi/envs/opt_env/bin/activate"
#EXTERNAL_REPO_PATH="/h/mchoi/metaseq_dependencies"
#METASEQ_REPO_PATH="/h/mchoi/metaseq_vector"

echo "Installing dependencies for "$ENV_PATH""

# Inject cluster-specific CUDA binary paths
# Note: You may need to add the following line to the deactivate() method:
#	export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
# TODO: Untested
read -r -d '' CUDA_PATH_SPEC << EOM
# Cluster CUDA binary paths
_OLD_VIRTUAL_PATH="$PATH"
PATH="/pkgs/cuda-11.1/bin:$VIRTUAL_ENV/bin:$PATH"
export PATH

_OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/pkgs/cudnn-11.1-v8.2.4.15/lib64:/pkgs/cuda-11.1/lib64"
export NCCL_IB_DISABLE=1
EOM

echo "$CUDA_PATH_SPEC"

echo "$CUDA_PATH_SPEC" >> "$ENV_PATH"

source activate "$ENV_PATH"

export PATH=/pkgs/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH="/pkgs/cudnn-11.1-v8.2.4.15/lib64:/pkgs/cuda-11.1/lib64"
export NCCL_IB_DISABLE=1


# Install torch
pip install --upgrade setuptools
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
python -c 'import torch; print("Torch version:", torch.__version__)'

# Install common dependencies
#pip install setuptools==59.5.0
#pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda111 -U
#python -c 'import torch; print("Torch version:", torch.__version__)'
#python -m torch.utils.collect_env

# Install fairscale
if ! python -c 'import fairscale'; then
	cd "$EXTERNAL_REPO_PATH"
	git clone https://github.com/facebookresearch/fairscale.git
	cd fairscale
	git checkout 1bc96fa8c69def6d990e42bfbd75f86146ce29bd
	pip install .
	cd ~/
fi

# Install fused ops
if ! python -c 'import apex'; then
	cd "$EXTERNAL_REPO_PATH"
	git clone https://github.com/NVIDIA/apex
	cd apex
	git checkout e2083df5eb96643c61613b9df48dd4eea6b07690
	pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
	cd ~/
fi
if ! python -c 'import megatron_lm'; then
	cd "$EXTERNAL_REPO_PATH"
	git clone --depth=1 --branch fairseq_v2 https://github.com/ngoyal2707/Megatron-LM.git
	cd Megatron-LM
	pip install -r requirements.txt
	pip install -e .
	cd ~/
fi

# Install metaseq repo
cd "$METASEQ_REPO_PATH"
pip install -e .
python setup.py build_ext --inplace