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


