#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m metaseq_cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""

import os
import queue
import pkg_resources
from collections import defaultdict
import random
import threading
import traceback
import pickle
import codecs
import functools

import torch

import megatron.mpu.initialize as mpu_init

from metaseq import options, tasks
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.queue import PriorityQueueRingShard
from metaseq.service.constants import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    MAX_BEAM,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    LAUNCH_ARGS,
    UNBATCHED_ARG_DICT,
)
from metaseq.service.utils import get_my_ip, encode_fn, build_logger
from metaseq.service.responses import OAIResponse

from metaseq_cli.hook_utils import get_activation_capture_hook_dict, apply_forward_hook


# global state (mutable!)
cfg = None
port = DEFAULT_PORT
BATCH_QUEUE = PriorityQueueRingShard()

logger = build_logger()


def _get_total_param_buffer_size(model):
    mem_params = sum([
        param.nelement() * param.element_size()
        for param in model.parameters()
    ])
    mem_bufs = sum([
        buf.nelement() * buf.element_size()
        for buf in model.buffers()
    ])
    mem = mem_params + mem_bufs # in bytes
    return mem


def worker_main(cfg1: MetaseqConfig, namespace_args=None):
    # disable multithreading in tokenizers and torch, as different Flask threads
    # may then fight for resources.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    global generator
    global MODE

    # make sure generations are stochastic since we have many workers
    torch.manual_seed(6 + torch.distributed.get_rank())
    torch.cuda.manual_seed(6 + torch.distributed.get_rank())
    MODE = "worker"
    cfg = cfg1

    logger.info(f"Rank{torch.distributed.get_rank()}: TP world size = {mpu_init.get_tensor_model_parallel_world_size()}")
    logger.info(f"Rank{torch.distributed.get_rank()}: PP world size = {mpu_init.get_pipeline_model_parallel_world_size()}")
    logger.info(f"Rank{torch.distributed.get_rank()}: TP rank = {mpu_init.get_tensor_model_parallel_rank()}")
    logger.info(f"Rank{torch.distributed.get_rank()}: PP rank = {mpu_init.get_pipeline_model_parallel_rank()}")

    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    model.make_generation_fast_()

    breakpoint()
    quit()

    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841


def cli_main():
    """
    Hosted version of the web UI for generation.
    """

    global port, MODE, cfg
    parser = options.get_generation_parser()

    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    for s in LAUNCH_ARGS:
        flat_launch_args += s.split()
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    port = DEFAULT_PORT
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE

    distributed_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    cli_main()
