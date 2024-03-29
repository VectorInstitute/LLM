FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /build
WORKDIR /build

RUN apt-key del 7fa2af80 && \
    apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends curl && \
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
    git \
    python3-pip python3-dev

# Install Pytorch
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install APEX
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /build/apex
RUN git checkout e2083df5eb96643c61613b9df48dd4eea6b07690
RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./

ARG CUDA_VISIBLE_DEVICES=none
ARG TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Install Megatron-LM branch
WORKDIR /build

RUN git clone --branch fairseq_v2 https://github.com/ngoyal2707/Megatron-LM.git
WORKDIR /build/Megatron-LM
RUN pip3 install six==1.16.0 regex==2023.5.5
RUN pip3 install -e .

# Install Fairscale
WORKDIR /build
RUN git clone --branch prefetch_fsdp_params_simple https://github.com/facebookresearch/fairscale.git
WORKDIR /build/fairscale
RUN git checkout 1bc96fa8c69def6d990e42bfbd75f86146ce29bd
RUN pip3 install -e .

# Install metaseq
WORKDIR /build
ADD . .
WORKDIR /build/metaseq
RUN pip3 install -e .
RUN python3 setup.py install

# Install transformers
RUN pip install transformers==4.29.2

# Install PyTriton
RUN pip install --upgrade pip
RUN pip install -U https://github.com/triton-inference-server/pytriton/releases/download/v0.1.4/pytriton-0.1.4-py3-none-manylinux_2_31_x86_64.whl
RUN python3 -m pip install datasets==2.10.1 pandas==1.5.3 requests==2.28.2
