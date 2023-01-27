#!/usr/bin/env python

"""
Script for converting the same sharding into megatron

Note: This may take a while for larger models!

Usage:
    $ ls 125m
    dict.txt
    gpt2-merges.txt
    gpt2-vocab.json
    reshard-model_part-0.pt
    reshard-model_part-1.pt

    $ python -m metaseq.scripts.convert_to_megatron 125m

    $ ls 125m
    dict.txt
    gpt2-merges.txt
    gpt2-vocab.json
    reshard-model_part-0.pt
    reshard-model_part-1.pt
    restored.pt
"""

import argparse
import glob
import logging
import os
import psutil
import sys
import tempfile

import torch

from metaseq import options, tasks, checkpoint_utils, utils
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.distributed.stitch_fsdp_ckpt import reshard_megatron_parts

# Globals
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)

# Max cpu ram in units number of fp16 parameters (150G // 2 params)
#MAX_CPU_RAM = 150000000000 // 2
# TODO: Testing for 6.7B
MAX_CPU_RAM = 4_000_000_000 // 2
UNITS = 1_000_000_000


def probe_gpu_and_cpu_mem(location=None):
    """Log the current used GPU and CPU memory."""
    rank = torch.distributed.get_rank()

    gpu_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / UNITS
    gpu_reserved = torch.cuda.memory_reserved() / UNITS
    gpu_allocated = torch.cuda.memory_allocated() / UNITS
    gpu_free = gpu_reserved - gpu_allocated

    mem_snapshot = psutil.virtual_memory()
    cpu_available = mem_snapshot.available / UNITS
    cpu_used = mem_snapshot.used / UNITS

    #logger.info(f"Rank {rank} GPU mem: total-{gpu_total} | reserved-{gpu_reserved} | "
    #            f"allocated-{gpu_allocated} | free-{gpu_free}")

    logger.info(f"Rank {rank} CPU mem: {cpu_used} / "
                f"{cpu_available} | threshold(user-defined) - "
                f"{MAX_CPU_RAM / UNITS} | location {location}")


def make_temp_workspace(cfg):
    """Create the temp workspace directory for temporary checkpoints."""
    try:
        tmpdir = tempfile.TemporaryDirectory(dir=cfg.task.data)

    except:
        print(f"Couldn't make the temp workspace at: {cfg.task.data}")
        raise Exception

    return tmpdir


def make_directory(cfg, dirname):
    path = os.path.join(cfg.task.data, dirname)

    if not os.path.isdir(path):
        os.mkdir(path)

    return path


def create_generation_config_with_defaults(model_path):
    files = glob.glob(f"{model_path}/reshard*.pt")

    MP = len(files)
    BPE_MERGES = model_path + "/gpt2-merges.txt"
    BPE_VOCAB = model_path + "/gpt2-vocab.json"

    # Skeleton out all the annoying command line args we can infer
    ARGS = [
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
    print(ARGS)

    # build up the config file
    parser = options.get_generation_parser()
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    args = options.parse_args_and_arch(parser, input_args=ARGS)
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = MP

    return cfg


def cleanup_model_state_dict(mp_state_dict):
    """
    Given a (MP sharded) shard of a model state dict, do some cleaning of its
    items.
    """
    # glued['decoder.output_projection.weight'] = glued['decoder.embed_tokens.weight']

    # NOTE: Decoder version assignment happens at the end of pipeline now

    if "decoder.output_projection.weight" in mp_state_dict:
        del mp_state_dict["decoder.output_projection.weight"]

    return mp_state_dict


def save_checkpoint_to_disk(path, output_sd):
    """
    Given a state dict and a name (typically with a specific MP rank attached),
    save it to dist and return the save path.
    """
    if distributed_utils.get_global_rank() == 0:
        with open(
            path, "wb"
        ) as f:
            torch.save(output_sd, f)


def clean_and_dump_sd(tmpdir_path, partition_id, partial_megatron_glued_parts):
    """
    Given a layer batch of megatron glued parts and its metadata, clean the
    state dict and dump everything to disk.
    """
    if torch.distributed.get_rank() != 0:
        raise Exception("Non-rank0 trying to dump state dict!")

    probe_gpu_and_cpu_mem("clean_and_dump_sd")

    # Post-process and dump layer batch
    for mp_rank, mp_state_dict in enumerate(partial_megatron_glued_parts):
        # Clean the state dict of certain keys
        mp_state_dict = cleanup_model_state_dict(mp_state_dict)

    assert len(partial_megatron_glued_parts) == torch.distributed.get_world_size()

    # Dump
    save_path = os.path.join(
        tmpdir_path,
        #f"temp_batch-p{partition_id}_megatronreshard-model_part-{mp_rank}.pt",
        f"temp_megatron_reshard-{partition_id}.pt",
    )
    save_checkpoint_to_disk(save_path, partial_megatron_glued_parts)
    return save_path


def undo_fsdp_and_save_pipeline_chunks(cfg, model, mp_size, tmpdir_path):
    """
    Given an FSDP-wrapped model, run a convert-and-dump pipeline given by the
    procedure:
        1. Summon unflattened params from FSDP
        2. For each parameter, starting at layer 0 -> layer N
        3. Gather parameter across MP dim
        4. Reshard it megatron-style, unity MP-degree (may be idempotent!
           double check this)
        5. Dump to cpu
            5.a If no room left on cpu, dump parameters to temp disk file
            5.b This temp disk file will contain parameters comprising of full
                MP dimension, but only a subset of the total layers
        6. If anything is left over on cpu, finish by dumping to disk like the
           rest
    """
    layer_batch_save_paths = []
    param_checksum = 0

    dumped_params = set()

    # Summon all params using fsdp
    with model.summon_full_params():
        # Need to keep track of CPU RAM mem usage (2 bytes per fp16 param)
        params_used = 0
        partition_id = 0    # Layer batch number

        model_parts = [{} for _ in range(mp_size)]

        # Begin pipeline dim defined along model parameters
        for name, param in model.named_parameters():
            # Gather full params to only rank0
            gathered = [torch.zeros_like(param) for _ in range(mp_size)]

            # NCCL version doesnt support gather
            torch.distributed.all_gather(
                gathered,
                param,
                group=distributed_utils.get_global_group(),
            )

            # Only rank0 needs to do the book-keeping
            # Rest of ranks should wait() at the next all-gather
            if torch.distributed.get_rank() == 0:
                num_params = sum(p.numel() for p in gathered)

                # Dump current partial state dict to disk if RAM OOM incoming
                if num_params + params_used >= MAX_CPU_RAM:
                    logger.info(f"Dumping partition: {partition_id} to disk. ")
                    # Stitch the MP-sharded params
                    model_parts = reshard_megatron_parts(
                        model_parts,
                        new_model_part_count=len(model_parts),
                    )

                    # Clean state dict and dump to disk
                    save_path = clean_and_dump_sd(
                        tmpdir_path,
                        partition_id,
                        model_parts,
                    )
                    layer_batch_save_paths.append(save_path)

                    # INC layer batch id
                    partition_id += 1

                    # Delete dumped model parts
                    model_parts = [{} for _ in range(mp_size)]

                    # Reset param counter
                    params_used = 0

                # Put on RAM
                for mp_rank, tensor in enumerate(gathered):
                    model_parts[mp_rank][name] = tensor.cpu()
                    params_used += tensor.numel()
                    param_checksum += tensor.numel()

                # Update checksum
                dumped_params.update([name])

        if len(model_parts[0]) > 0:
            # Again, only dump if rank0
            if torch.distributed.get_rank() == 0:
                save_path = clean_and_dump_sd(
                    tmpdir_path,
                    partition_id,
                    model_parts,
                )
                layer_batch_save_paths.append(save_path)

    return layer_batch_save_paths, param_checksum, dumped_params


def reshard_partial_checkpoints(save_paths, target_mp, tmpdir_path):
    new_save_paths = []

    # Pipeline through saved state dict partitions
    for partition_id, path in enumerate(save_paths):
        # Load and reshard to new model parallel degree
        state_dict = torch.load(path)

        probe_gpu_and_cpu_mem("reshard_partial_checkpoints pre megatron")

        state_dict = reshard_megatron_parts(
            state_dict,
            new_model_part_count=target_mp,
        )

        probe_gpu_and_cpu_mem("reshard_partial_checkpoints post megatron")

        # Each MP rank shard should be saved, resuling in coordinates (PP, MP)
        for mp_rank, mp_state_dict in enumerate(state_dict):
            save_path = os.path.join(
                tmpdir_path,
                f"temp_megatron_reshard-{partition_id}-{mp_rank}.pt",
            )
            new_save_paths.append(save_path)
            save_checkpoint_to_disk(
                save_path,
                mp_state_dict,
            )
            # Aggressively free mem
            state_dict[mp_rank] = None

        # Delete the old file
        os.remove(path)

        # Force deletion of state dict
        del state_dict

    return new_save_paths


def worker_main(cfg: MetaseqConfig):
    """
    Load up the model on all workers for Model Parallelism, then
    unflatten, move to cpu, and save to "restored.pt".
    """
    if torch.distributed.get_rank() == 0:
        probe_gpu_and_cpu_mem("inital")

    task = tasks.setup_task(cfg.task)

    # Build the full FSDP model and load checkpoint to GPU
    def _build_model(cfg, task):
        cfg.model.tensor_parallel_init_model_on_gpu = True
        model = task.build_model(cfg.model).cuda()
        return fsdp_wrap(model)

    with fsdp_enable_wrap(
        cfg.distributed_training,
        use_sharded_state=cfg.distributed_training.use_sharded_state,
    ):
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

    # Init output state dict for consolidation to rank0
    mp_size = distributed_utils.get_model_parallel_world_size()

    # NOTE: We need to pipeline the conversion since we don't have enough CPU
    #       RAM per node. Therefore we define a pipeline which is just a for
    #       loop over all the layers. Once we reach the max memory that we can
    #       dump to RAM, we package that batch of layers and dump to disk. Once
    #       everything is converted to megatron and dumped to disk, we can then
    #       do the MP-degree resharding offline, per batch of layers. Finally,
    #       we can then consolidate along the batch of layers to give us the
    #       new degree of model parallelism.

    # Fails inside context manager, cache for later
    decoder_version = model.state_dict()["decoder.version"].item()

    # All ranks make temp dir, but only rank0 should use it
    tmpdir = make_temp_workspace(cfg)

    # All pipeline logic here, output should be new resharded checkpoints in a
    # permanent location
    with tmpdir as tmpdir_path:
        save_paths, param_checksum, dumped_params = undo_fsdp_and_save_pipeline_chunks(
            cfg,
            model,
            mp_size,
            tmpdir_path,
        )

        target_mp = 2

        # Rest can be done by only rank0
        if torch.distributed.get_rank() == 0:
            logger.info(f"Save paths: {save_paths}")
            logger.info(f"Number of saved params: {param_checksum / UNITS}B")

            mp_pipeline_save_paths = reshard_partial_checkpoints(
                save_paths,
                target_mp=target_mp,
                tmpdir_path=tmpdir_path,
            )
            logger.info(f"Save paths: {mp_pipeline_save_paths}")

            finaldir = make_directory(cfg, "resharded_megatron")
            final_save_paths = gather_layers_per_mp_rank(
                cfg,
                mp_pipeline_save_paths,
                finaldir,
                target_mp,
                decoder_version,
                dumped_params,
            )

        else:
            return

    logger.info("Completed conversion to megatron")
    logger.info(f"New conversion paths are: {final_save_paths}")


def format_sd_with_template(model_sd, template_sd, decoder_version):
    """
    Given a (MP sharded) shard of a model state dict, load its original
    FSDP model checkpoint to CPU, and replace the FSDP model state dict with
    the new one.
    """
    model_sd["decoder.version"] = torch.tensor(
        [decoder_version],
        dtype=torch.float32,
    )

    output_sd = dict(model=model_sd)
    output_sd.update(template_sd)

    #output_sd["model"] = utils.move_to_cpu(output_sd)
    output_sd["cfg"]["model"].arch = "transformer_lm_megatron"
    output_sd["cfg"]["model"]._name = "transformer_lm_megatron"

    return output_sd


def gather_layers_per_mp_rank(cfg, save_paths, output_dir, target_mp, decoder_version, all_keys):
    """
    Given a list of partial model checkpoint paths, run a pipeline which takes
    one MP rank at a time, and combines its pipeline sharded state dicts.
    """
    def _get_mp_rank(path):
        path = path.split("-")[-1]  # Get "mp_rank.pt"
        rank = path.split(".")[0]   # Remove ".pt"
        return int(rank)

    def _get_partition_rank(path):
        rank = path.split("-")[-2]
        return int(rank)

    # Group by model parallel rank
    mp_grouped_paths = [[] for _ in range(target_mp)]
    for mp_pipeline_path in save_paths:
        # Put save path in corresponding bucket according to MP rank
        mp_rank = _get_mp_rank(mp_pipeline_path)
        mp_grouped_paths[mp_rank].append(mp_pipeline_path)

    # Load dummy checkpoint, prune loaded model checkpoint
    dummy_sd = checkpoint_utils.load_checkpoint_to_cpu(
        cfg.common_eval.path.replace(
            "reshard.pt",
            "reshard-model_part-0.pt",
        )
    )
    template_sd = dict(
        cfg=dummy_sd["cfg"],
        extra_state=dummy_sd["extra_state"],
        optimizer_history=dummy_sd["optimizer_history"],
        args=dummy_sd.get("args"),
    )

    # Delete everything but the metadata we need
    del dummy_sd

    # Load all partitions and combine into single dict for each rank
    final_save_paths = []
    for mp_group in mp_grouped_paths:
        model_state_dict = {}

        probe_gpu_and_cpu_mem("gather_layers_per_mp_rank")

        mp_rank = _get_mp_rank(mp_group[0])

        # Can be list since partitions contain mutex parameter groups
        already_merged_partition_ids = []
        for path in mp_group:
            # Guard against merging across MP ranks
            partition_id = _get_partition_rank(path)
            assert partition_id not in already_merged_partition_ids

            # Load partition shard, and empty it into corresponding MP rank
            model_state_dict.update(torch.load(path))

            # Delete loaded file
            os.remove(path)

            already_merged_partition_ids.append(partition_id)

        model_state_dict = format_sd_with_template(
            model_state_dict,
            template_sd,
            decoder_version,
        )

        # Make sure we haven't lost any keys
        key_diff = all_keys.symmetric_difference(model_state_dict["model"].keys())
        assert key_diff == set(["decoder.version"]), f"Key difference: {key_diff}"

        output_path = os.path.join(
            output_dir,
            f"megatronreshard-model_part-{mp_rank}.pt",
        )
        save_checkpoint_to_disk(
            output_path,
            model_state_dict,
        )
        final_save_paths.append(output_path)
        logger.info(f"Saved rank{mp_rank} shard to {output_path}")

    return final_save_paths


def main():
    # parser to be used like docstring shows
    real_parser = argparse.ArgumentParser()
    real_parser.add_argument("location")
    args = real_parser.parse_args()

    cfg = create_generation_config_with_defaults(args.location)
    distributed_utils.call_main(cfg, worker_main)


if __name__ == "__main__":
    main()
