#!/usr/bin/env python
import logging
import os
import sys

import fire

from metaseq.distributed.megatron_resharding import reshard_model_parallel


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


if __name__ == "__main__":
    fire.Fire(reshard_model_parallel)
