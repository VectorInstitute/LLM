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
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch import Tensor

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

_MetaseqConfig = Any
_FSDPModel = Any

MAX_CPU_RAM = None
TARGET_MP_SIZE = None
UNITS = 1_000_000_000


def create_generation_config_with_defaults(
    model_path: str,
    target_mp_size: int,
    max_cpu_ram: int,
) -> _MetaseqConfig:
    files = glob.glob(f"{model_path}/reshard*.pt")

    MP = len(files)
    BPE_MERGES = model_path + "/gpt2-merges.txt"
    BPE_VOCAB = model_path + "/gpt2-vocab.json"

    global MAX_CPU_RAM
    MAX_CPU_RAM = max_cpu_ram

    global TARGET_MP_SIZE
    TARGET_MP_SIZE = target_mp_size

    logger.info(
        "Running megatron resharding with args: ",
        f"location: {model_path} | ",
        f"target_mp_size: {target_mp_size} | ",
        f"max_cpu_ram: {max_cpu_ram} | ",
    )

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


def probe_gpu_mem(location: Optional[str] = None) -> None:
    """
    Log the current used GPU memory. Optionally print some string with a
    location description.
    """
    rank = torch.distributed.get_rank()

    gpu_total = torch.cuda.get_device_properties(
        torch.cuda.current_device()).total_memory / UNITS
    gpu_reserved = torch.cuda.memory_reserved() / UNITS
    gpu_allocated = torch.cuda.memory_allocated() / UNITS
    gpu_free = gpu_reserved - gpu_allocated

    logger.info(
        f"Rank {rank} GPU mem: total-{gpu_total} | reserved-{gpu_reserved} | "
        f"allocated-{gpu_allocated} | free-{gpu_free}",
    )


def probe_cpu_mem(location: Optional[str] = None) -> None:
    """
    Log the current used GPU and CPU memory. Optionally print some string with
    a location description.
    """
    rank = torch.distributed.get_rank()
    
    mem_snapshot = psutil.virtual_memory()
    cpu_available = mem_snapshot.available / UNITS
    cpu_used = mem_snapshot.used / UNITS

    logger.info(f"Rank {rank} CPU mem: {cpu_used} / "
                f"{cpu_available} | threshold(user-defined) - "
                f"{MAX_CPU_RAM / UNITS} | location {location}")


def make_temp_workspace(cfg: _MetaseqConfig) -> tempfile.TemporaryDirectory:
    """Create the temp workspace directory for temporary checkpoints."""
    try:
        tmpdir = tempfile.TemporaryDirectory(dir=cfg.task.data)

    except:
        print(f"Couldn't make the temp workspace at: {cfg.task.data}")
        raise Exception

    return tmpdir


def make_directory(cfg: _MetaseqConfig, dirname: str) -> str:
    """Make a directory. If it already exists, simply return the path."""
    path = os.path.join(cfg.task.data, dirname)

    if not os.path.isdir(path):
        os.mkdir(path)

    return path


def save_checkpoint_to_disk(path: str, output_sd: Dict[str, Tensor]) -> None:
    """
    Given a state dict and a name (typically with a specific MP rank attached),
    save it to dist and return the save path.
    """
    if distributed_utils.get_global_rank() == 0:
        with open(
            path, "wb"
        ) as f:
            torch.save(output_sd, f)


def cleanup_model_state_dict(
    mp_state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """
    Given a (MP sharded) shard of a model state dict, do some cleaning of its
    items.
    """
    # glued['decoder.output_projection.weight'] = glued[
    #                                           'decoder.embed_tokens.weight']

    # NOTE: Decoder version assignment happens at the end of pipeline now

    if "decoder.output_projection.weight" in mp_state_dict:
        del mp_state_dict["decoder.output_projection.weight"]

    return mp_state_dict


def clean_and_dump_sd(
    tmpdir_path: str,
    partition_id: int,
    partial_megatron_glued_parts: List[Dict[str, Tensor]],
) -> str:
    """
    Given a pipeline chunk of megatron glued parts and its metadata, clean the
    state dict and dump everything to disk.
    """
    if torch.distributed.get_rank() != 0:
        raise Exception("Non-rank0 trying to dump state dict!")

    probe_cpu_mem("clean_and_dump_sd")

    # Post-process pipeline chunk
    for mp_state_dict in partial_megatron_glued_parts:
        mp_state_dict = cleanup_model_state_dict(mp_state_dict)

    assert (
        len(partial_megatron_glued_parts)
        == torch.distributed.get_world_size()
    )

    # Dump to disk
    save_path = os.path.join(
        tmpdir_path,
        f"temp_megatron_reshard-{partition_id}.pt",
    )
    save_checkpoint_to_disk(save_path, partial_megatron_glued_parts)

    return save_path


def make_pipeline_chunks(
    model: _FSDPModel,
    mp_size: int,
    tmpdir_path: str,
) -> Tuple[List[str], int, List[str]]:
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
    def gather_params(param: Tensor) -> List[Tensor]:
        """
        Given a parameter sharded across MP workers, all-gather. The
        resulting list of tensors has MP-degree elements.
        """
        # Gather full params to only rank0
        gathered = [torch.zeros_like(param) for _ in range(mp_size)]

        # NCCL version doesnt support gather
        torch.distributed.all_gather(
            gathered,
            param,
            group=distributed_utils.get_global_group(),
        )
        return gathered

    def dump_model_parts(
        model_parts: List[Dict[str, Tensor]],
        partition_id: int,
    ) -> str:
        """Given an MP-sharded model state dict, dump it to disk."""
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

        return save_path

    # For downstream correctness-checks
    param_checksum = 0
    dumped_params = set()

    # Pipeline state
    params_used = 0
    partition_id = 0

    model_parts = [{} for _ in range(mp_size)]
    pipeline_chunk_save_paths = []

    # Begin pipeline
    with model.summon_full_params():
        for name, param in model.named_parameters():
            gathered = gather_params(param)

            # Only rank0 needs to do the book-keeping
            # Rest of ranks should wait() at the next all-gather
            if torch.distributed.get_rank() == 0:
                num_params = sum(p.numel() for p in gathered)

                # Dump current partial state dict to disk if RAM OOM incoming
                if num_params + params_used >= (MAX_CPU_RAM // 2):  # fp16
                    save_path = dump_model_parts(
                        model_parts,
                        partition_id,
                    )
                    pipeline_chunk_save_paths.append(save_path)

                    # Update pipeline state
                    partition_id += 1
                    params_used = 0
                    model_parts = [{} for _ in range(mp_size)]

                # Put parameter on RAM
                for mp_rank, tensor in enumerate(gathered):
                    model_parts[mp_rank][name] = tensor.cpu()
                    params_used += tensor.numel()
                    param_checksum += tensor.numel()
                dumped_params.update([name])

        # Flush pipeline leftovers to disk when finished
        if len(model_parts[0]) > 0:
            if torch.distributed.get_rank() == 0:
                save_path = clean_and_dump_sd(
                    tmpdir_path,
                    partition_id,
                    model_parts,
                )
                pipeline_chunk_save_paths.append(save_path)

    return pipeline_chunk_save_paths, param_checksum, dumped_params


def reshard_pipeline_chunks(
    save_paths: List[str],
    tmpdir_path: str,
) -> List[List[str]]:
    """
    Given the paths to the pipeline chunks on disk, load them one-by-one and
    change their MP sharding degree.
    """
    new_save_paths = [[] for _ in range(TARGET_MP_SIZE)]

    # Pipeline through saved state dict partitions
    for partition_id, path in enumerate(save_paths):
        state_dict = torch.load(path)

        probe_cpu_mem("reshard_pipeline_chunks pre megatron")

        state_dict = reshard_megatron_parts(
            state_dict,
            new_model_part_count=TARGET_MP_SIZE,
        )

        probe_cpu_mem("reshard_pipeline_chunks post megatron")

        # NOTE: Can't gather across pipeline chunks yet, since we would have to
        # materialize entire model onto RAM! Hence we dump to disk again
        for mp_rank, mp_state_dict in enumerate(state_dict):
            save_path = os.path.join(
                tmpdir_path,
                f"temp_megatron_reshard-{partition_id}-{mp_rank}.pt",
            )
            new_save_paths[mp_rank].append(save_path)
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


def format_sd_with_template(
    model_sd: Dict[str, Tensor],
    template_sd: Dict[str, Any],
    decoder_version: float,
) -> Dict[str, Any]:
    """
    Given a MP shard of a model state dict, load its original
    FSDP model checkpoint to CPU, and replace the FSDP model state dict with
    the new one.
    """
    model_sd["decoder.version"] = torch.tensor(
        [decoder_version],
        dtype=torch.float32,
    )

    output_sd = dict(model=model_sd)
    output_sd.update(template_sd)

    # output_sd["model"] = utils.move_to_cpu(output_sd)
    output_sd["cfg"]["model"].arch = "transformer_lm_megatron"
    output_sd["cfg"]["model"]._name = "transformer_lm_megatron"

    return output_sd


def gather_pipeline_chunks(
    cfg: _MetaseqConfig,
    save_paths: List[List[str]],
    output_dir: str,
    decoder_version: float,
    all_keys: List[str],
) -> List[str]:
    """
    Given a list of partial model checkpoint paths, run a pipeline which takes
    one MP rank at a time, and combines its pipeline sharded state dicts.
    """
    def _get_mp_rank(path: str) -> int:
        """Grep and return MP rank from path."""
        path = path.split("-")[-1]  # Get "mp_rank.pt"
        rank = path.split(".")[0]   # Remove ".pt"
        return int(rank)

    def _get_partition_rank(path: str) -> int:
        """Grep and return pipeline chunk id rank from path."""
        rank = path.split("-")[-2]
        return int(rank)

    def merge_pipeline_chunks(mp_group: List[str]) -> List[str]:
        """
        Given a group save paths corresponding to one MP worker, load the
        corresponding pipeline chunk files from disk and merge them into one
        state dict for the MP worker. Finish by dumping the final state dict
        to disk, and return the dump path.
        """
        # Guard against merging across MP ranks
        already_merged_partition_ids = []

        # Populate mp-worker state dict
        mp_state_dict = {}
        for path in mp_group:
            partition_id = _get_partition_rank(path)
            assert partition_id not in already_merged_partition_ids

            # Load partition shard, and empty it into corresponding MP rank
            mp_state_dict.update(torch.load(path))

            # Delete loaded file
            os.remove(path)

            already_merged_partition_ids.append(partition_id)

        # Add metadata to state dict
        mp_state_dict = format_sd_with_template(
            mp_state_dict,
            template_sd,
            decoder_version,
        )

        # Make sure we haven't lost any keys
        key_diff = all_keys.symmetric_difference(mp_state_dict["model"].keys())
        assert key_diff == set(
            ["decoder.version"]
        ), f"Key difference: {key_diff}"

        output_path = os.path.join(
            output_dir,
            f"megatronreshard-model_part-{mp_rank}.pt",
        )
        save_checkpoint_to_disk(
            output_path,
            mp_state_dict,
        )
        return output_path

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
    del dummy_sd

    # Begin pipeline
    final_save_paths = []
    for mp_rank, mp_group in enumerate(save_paths):
        probe_cpu_mem("gather_pipeline_chunks pre merge")

        mp_output_path = merge_pipeline_chunks(mp_group)

        probe_cpu_mem("gather_pipeline_chunks post merge")

        # Make sure list isn't out of order for some reason
        assert mp_rank == _get_mp_rank(mp_group[0])

        final_save_paths.append(mp_output_path)

        logger.info(f"Saved rank{mp_rank} shard to {mp_output_path}")

    return final_save_paths


def worker_main(cfg: MetaseqConfig) -> None:
    """
    We need to pipeline the conversion since we don't have enough CPU
    RAM per node. Therefore we define a pipeline which is just a for
    loop over all the layers. Once we reach the max memory that we can
    dump to RAM, we package that batch of layers and dump to disk. Once
    everything is converted to megatron and dumped to disk, we can then
    do the MP-degree resharding offline, per batch of layers. Finally,
    we can then consolidate along the batch of layers to give us the
    new degree of model parallelism.

    """
    if torch.distributed.get_rank() == 0:
        probe_cpu_mem("inital")

    task = tasks.setup_task(cfg.task)

    # Build the full FSDP model and load checkpoint to GPU
    def _build_model(cfg: _MetaseqConfig, task: Any):
        """Build model hook."""
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
        model = models[0]   # Not MoE

    # Init output state dict for consolidation to rank0
    mp_size = distributed_utils.get_model_parallel_world_size()

    # Fails inside context manager, cache for later
    decoder_version = model.state_dict()["decoder.version"].item()

    # All ranks make temp dir, but only rank0 should use it
    tmpdir = make_temp_workspace(cfg)

    with tmpdir as tmpdir_path:
        save_paths, param_checksum, dumped_params = make_pipeline_chunks(
            model,
            mp_size,
            tmpdir_path,
        )

        # Rest can be done by only rank0
        if torch.distributed.get_rank() == 0:
            logger.info(f"Save paths: {save_paths}")
            logger.info(f"Number of saved params: {param_checksum / UNITS}B")

            mp_pipeline_save_paths = reshard_pipeline_chunks(
                save_paths,
                tmpdir_path=tmpdir_path,
            )
            logger.info(f"Save paths: {mp_pipeline_save_paths}")

            finaldir = make_directory(cfg, "resharded_megatron")
            final_save_paths = gather_pipeline_chunks(
                cfg,
                mp_pipeline_save_paths,
                finaldir,
                decoder_version,
                dumped_params,
            )
        else:
            return

    logger.info("Completed conversion to megatron")
    logger.info(f"New conversion paths are: {final_save_paths}")


def main():
    # parser to be used like docstring shows
    real_parser = argparse.ArgumentParser()
    real_parser.add_argument("location")
    real_parser.add_argument("target_mp_size")
    real_parser.add_argument("max_cpu_ram")
    args = real_parser.parse_args()

    cfg = create_generation_config_with_defaults(args.location)
    distributed_utils.call_main(cfg, worker_main)


if __name__ == "__main__":
    main()
