# Vector MetaSeq Setup Guide

Welcome to the Vector MetaSeq setup guide. This document will guide you through the steps to get the metaseq environment setup on the Vector Vaughan Cluster.


### Setup 

1. Clone the LLM github repo which contains the Vector fork of MetaSeq

```bash
    $ git clone git@github.com:VectorInstitute/LLM.git
```

2. Create a conda environment 

```bash
    $ conda create --name metaseq python=3.9
```

Note: that this will create the environment in the default `~/.conda` location

3. Run the environment setup script

```bash 
    $ bash /path/to/LLM/metaseq/metaseq/scripts/vector_setup_metaseq_env.sh /
        /path/to/your/conda/environment /
        /path/to/metaseq/build/directory /
        /path/to/LLM/metaseq
```

The setup script takes 3 arguments, the first is the path to the conda environment created in step 2. The second is a path to a directory that will be used during metaseq for megatron and apex dependencies (this director will be created at the path provided it does not already exist). The third parameter is the path to your instance of metaseq.

4. Take a break! Installation may take a while!


