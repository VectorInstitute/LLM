import logging
import json
import os

from omegaconf import OmegaConf
import torch

from megatron.mpu.initialize import (
    get_tensor_model_parallel_rank, get_data_parallel_rank)


logger = logging.getLogger(__name__)


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = "release"
    else:
        d = "{:d}".format(iteration)
    if zero:
        dp_rank = get_data_parallel_rank()
        d += "_zero_dp_rank_{}".format(dp_rank)
    return os.path.join(
        checkpoints_path,
        d,
        "glmreshard-model_part-{}.pt".format(get_tensor_model_parallel_rank()),
    )   # NOTE: All GLM ckpts should be renamed similar to OPT ckpts


def get_checkpoint_tracker_filename(checkpoints_path, old_checkpoint=False):
    return os.path.join(checkpoints_path, "latest")


def get_checkpoint_iteration(load_path):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        raise ValueError(
            "could not find the metadata file {}, please check --load".format(
                tracker_filename
            )
        )
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, "r") as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == "release"
            if not release:
                exit()
    assert iteration > 0 or release, "error parsing metadata file {}".format(
        tracker_filename
    )

    return iteration, release, True


def get_model(cfg, model_cls):
    """Build the model."""

    # print_rank_0(f'building {model_cls.__name__} model ...')
    model = model_cls(cfg)

    logger.info(
        "Number of parameters on model parallel rank {}: {}".format(
            cfg.distributed_training.distributed_rank,
            sum([p.nelement() for p in model.parameters()]),
        )
    )

    if cfg.common.fp16:
        model.half()
    elif cfg.common.bf16:
        model.bfloat16()
    model.cuda(torch.cuda.current_device())

    return model


def load_checkpoint(model, cfg, load_path=None, prefix=""):
    """Load a model checkpoint."""
    if load_path is None:
        load_path = cfg.load

    iteration, release, success = get_checkpoint_iteration(load_path)
    if not success:
        return 0

    checkpoint_name = get_checkpoint_name(load_path, iteration, release)

    if get_data_parallel_rank() == 0:
        logger.info(
            f"Global rank {torch.distributed.get_rank()} is "
            f"loading checkpoint {checkpoint_name}"
        )

    try:
        sd = torch.load(checkpoint_name, map_location="cpu")
    except FileNotFoundError:
        print(f"Model checkpoint under {checkpoint_name} does not exist")

    if hasattr(model, "module"):
        module = model.module
    else:  # inference without deepspeed
        module = model

    missing_keys, unexpected_keys = module.load_state_dict(sd["model"], strict=False)

    if len(unexpected_keys) > 0:
        raise Exception(f"Found unexpected keys: {unexpected_keys}")

    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys for inference: {missing_keys}.")

    module.eval()

    if get_data_parallel_rank() == 0:
        print("  successfully loaded {}".format(checkpoint_name))
    del sd
    return iteration


def update_args_with_file(cfg, path):
    """
    Open the metadata file included with pretrained SwissArmyTransformer, and
    replace main cfg default args with those found in the metadata file
    """
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    """
    # expand relative path
    folder = os.path.dirname(path)
    for k in config:
        # all the relative paths in config are based on the folder
        if k.endswith('_path'):
            config[k] = os.path.join(folder, config[k])
            if cfg.distributed_training.distributed_rank == 0:
                print(f'> parsing relative path {k} in model_config as {config[k]}.')
    """
    # TODO (mchoi): This can all be done in one loop
    # Add new keys present in loaded model config
    OmegaConf.set_struct(
        cfg, False
    )  # This should only be used when loading configs from ckpts
    for k in config.keys():
        if k not in cfg.model:
            cfg.model.k = config[k]

    # Overwrite duplicate keys in cfg.model with loaded model config
    for k in cfg.model.keys():
        if k in config and cfg.model[k] != config[k]:
            if cfg.distributed_training.distributed_rank == 0:
                logger.info(f"Replacing {k}:{cfg.model[k]} with {config[k]}")
            cfg.model[k] = config[k]

    OmegaConf.set_struct(cfg, True)
    return cfg
