# Web Server Setup Guide

This document will guide you through the steps to run the default metaseq web-server on the Vector Vaughn cluster. Note - this guide will only cover launching the server for OPT-125M on a single node.

### Setup 

1. Launch an interactive job on Vaughan cluster.

```bash
    $ sh /path/to/LLM/metaseq/metaseq/scripts/vaughan_interactive_srun.sh
```

This script will launch an interactive job with 2 t4v2 GPUs.

2. Verify the config parameters in `/path/to/LLM/metaseq/metaseq/service/constants.py`.

    * Set `MODEL_PARALLEL = 2`
    * Set `TOTAL_WORLD_SIZE = 2`
    * Set `CHECKPOINT_FOLDER = "/checkpoint/opt_test/original/OPT-125M"`
    * Set `CHECKPOINT_LOCAL = "/checkpoint/opt_test/original/OPT-125M"`

3. Activate metaseq conda environment build in metaseq setup guide.

```bash 
    $ source activate ~/.conda/envs/metaseq
```

Note: that this will create the environment in the default `~/.conda` location

4. Set env variables.

    * export PATH=/pkgs/cuda-11.1/bin:$PATH
    * export LD_LIBRARY_PATH="/pkgs/cudnn-11.1-v8.2.4.15/lib64:/pkgs/cuda-11.1/lib64"

5. Run the server.

```bash 
    $ python /path/to/LLM/metaseq/metaseq_cli/interactive_hosted.py
```