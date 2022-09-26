import argparse
import glob
import logging
import os
import sys

import torch

from metaseq import options, tasks, checkpoint_utils, utils
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.distributed.stitch_fsdp_ckpt import reshard_megatron_parts
#from metaseq.models.glm import GLMModel


logger = logging.getLogger(__name__)


def create_config_with_defaults(
    model_path,
    arch,
    prefix="reshard",
):
    """Given the path to a model and its tokenizer files, create a config."""
    files = glob.glob(f"{model_path}/{prefix}*.pt")

    MP = len(files)
    BPE_MERGES = model_path + "/gpt2-merges.txt"
    BPE_VOCAB = model_path + "/gpt2-vocab.json"

    # Skeleton out all the annoying command line args we can infer
    ARGS = [
        "--arch",
        arch,
        "--model-parallel-size",
        str(MP),
        "--distributed-world-size",
        str(MP),
        "--task",
        "language_modeling",
        "--bpe-merges",
        BPE_MERGES,
        "--merges-filename",
        BPE_MERGES,
        "--bpe-vocab",
        BPE_VOCAB,
        "--vocab-filename",
        BPE_VOCAB,
        "--bpe",
        "hf_byte_bpe",
        "--path",
        model_path + "/reshard.pt",
        "--checkpoint-shard-count",
        "1",
        "--use-sharded-state",
        model_path,
    ]

    # build up the config file
    parser = options.get_generation_parser()
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    args = options.parse_args_and_arch(parser, input_args=ARGS)
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = MP

    return cfg


def convert_shards_to_singleton_worker_main(cfg: MetaseqConfig):
    """
    Load up the model on all workers for Model Parallelism, then
    unflatten, move to cpu, and save to "restored.pt".
    """
    task = tasks.setup_task(cfg.task)

    if torch.distributed.is_initialized():
        logger.info(f"Rank {torch.distributed.get_rank()}")

    def _build_model(cfg, task):
        # Fix so we don't have to manually comment out dtype arg in megatron-lm
        # source code
        cfg.model.tensor_parallel_init_model_on_gpu = True
        model = task.build_model(cfg.model).cuda()
        return fsdp_wrap(model)

    if cfg.model._name == "glm_large":
        # TODO (mchoi): Test this loading on MP=2 GLM model
        model, cfg = GLMModel.from_pretrained(cfg, 'glm-large-en-blank')
        breakpoint()

    # TODO (mchoi): Nothing past this is tested for GLM
        
    else:
        # TODO (mchoi): Add GLM under this checkpoint loading util
        models, _model_args, _task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=None,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=True,
            num_shards=cfg.checkpoint.checkpoint_shard_count,
            build_model_hook=_build_model,
        )
        model = models[0]
    # consolidate everything on rank0
    mp_size = distributed_utils.get_model_parallel_world_size()
    model_parts = [{} for _ in range(mp_size)]

    with model.summon_full_params():
        for name, p in model.named_parameters():
            gathered = [torch.zeros_like(p) for _ in range(mp_size)]
            torch.distributed.all_gather(
                gathered, p, group=distributed_utils.get_global_group()
            )
            for r, t in enumerate(gathered):
                model_parts[r][name] = t.cpu()

    if distributed_utils.get_global_rank() == 0:
        print("############ loaded up a model ###############")
        shard_metadata = model.local_metadata_dict()
        import pickle
        # TODO: Make this non-fixed
        with open('/checkpoint/opt_test/original/OPT-125M/test_shard_metadata.pkl', 'wb') as f:
              pickle.dump(shard_metadata, f)
        print("#################################")
        #exit(1)

    glued = reshard_megatron_parts(model_parts, new_model_part_count=1)[0]
    # glued['decoder.output_projection.weight'] = glued['decoder.embed_tokens.weight']

    glued["decoder.version"] = model.state_dict()["decoder.version"].cpu()

    if "decoder.output_projection.weight" in glued:
        del glued["decoder.output_projection.weight"]

    output_sd = checkpoint_utils.load_checkpoint_to_cpu(
        cfg.common_eval.path.replace("reshard.pt", "reshard-model_part-0.pt")
    )
    output_sd["model"] = utils.move_to_cpu(glued)
    output_sd["cfg"]["model"].arch = "transformer_lm"
    output_sd["cfg"]["model"]._name = "transformer_lm"

    if distributed_utils.get_global_rank() == 0:
        with open(cfg.task.data + "/restored.pt", "wb") as f:
            torch.save(output_sd, f)


def convert_shards_to_singleton(model_path, arch):
    cfg = create_config_with_defaults(
        model_path=model_path,
        arch=arch,
        prefix="mp_rank_"   # Case with default GLM checkpoints
    )
    distributed_utils.call_main(cfg, convert_shards_to_singleton_worker_main)


def upgrade_glm_sd(sd, prefix):
    new_sd = {'model':{}}   # Replace module with model for consistency with OPT
    for k in sd:
        if k != 'module':
            new_sd[k] = sd[k]

    for k in sd['module']:
        if k.startswith(prefix):
            new_sd['model'][k[len(prefix):]] = sd['module'][k]

    return new_sd


def reshard_model_parallel(
    model_singleton_path,
    part=0,
    target_mp_size=512,
    no_pad=False,
    drop_optimizer_state=False,
):
    try:
        state = torch.load(model_singleton_path)
    except FileNotFoundError:
        logger.info(f"Model checkpoint {model_singleton_path} does not exist")

    state = upgrade_glm_sd(state, prefix="")

    breakpoint()
