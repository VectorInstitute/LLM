import argparse
from collections import defaultdict
from glob import glob
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm

import torch

from metaseq.distributed.stitch_fsdp_ckpt import (
    reshard_megatron_parts,
    find_num_parts,
)
from metaseq.file_io import load_and_pop_last_optimizer_state


logger = logging.getLogger(__name__)


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path_prefix",
        type=str,
        default="/checkpoint/opt_test/original/OPT-125M/megatronreshard",
    )
    parser.add_argument(
        "--tgt_path_prefix",
        type=str,
        default="/checkpoint/opt_test/original/TEST_glm_merge/OPT-125M/megatronreshard",
    )
    parser.add_argument("--target_mp_size", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="opt")
    return parser.parse_args()


def _get_model_part_num(filename):
    return int(os.path.basename(filename).replace(".pt", "").split("-")[-1])


def reshard_model_parallel(
    src_path_prefix,
    tgt_path_prefix,
    target_mp_size,
):
    all_ckpt_files = sorted(
        list(glob(f"{src_path_prefix}*.pt")), key=_get_model_part_num
    )

    if len(all_ckpt_files) > 1:
        for ckpt_file in all_ckpt_files:
            assert "model_part" in os.path.basename(ckpt_file)

    print(f"Sharding {all_ckpt_files} into MP={target_mp_size}")

    # Load the checkpoints and add weights to a dict
    weights = []
    names = []
    start_time = time.time()
    for p in tqdm(all_ckpt_files):
        names.append(Path(p).name)
        ckpt = load_and_pop_last_optimizer_state(p)
        weights.append(ckpt["model"])

    num_parts = find_num_parts(names) if len(all_ckpt_files) > 1 else 1

    model_parts = defaultdict()

    assert len(weights) == num_parts
    for p in range(num_parts):
        model_parts[p] = weights[p]

    # Reshard the weights in the dict
    resharded_models = reshard_megatron_parts(model_parts, target_mp_size)

    # Save the resharded checkpoint
    def save_checkpoint(weights_to_save, prefix):
        ckpt_resharded = dict(
            model=weights_to_save,
            cfg=ckpt["cfg"],
            extra_state=ckpt["extra_state"],
            optimizer_history=ckpt["optimizer_history"],
            args=ckpt.get("args"),
        )
        save_path = f"{prefix}.pt"
        logger.info(f"Saving to {save_path} ...")
        torch.save(ckpt_resharded, save_path)
        logger.info(f"Done after {time.time() - start_time} minutes")
        return save_path

    saved_paths = []
    for part_id, resharded_weights in enumerate(resharded_models):
        saved_paths.append(
            save_checkpoint(
                resharded_weights, f"{tgt_path_prefix}-model_part-{part_id}"
            )
        )
    return saved_paths


def main(args):
    # NOTE: This is temporarly running as a script
    reshard_model_parallel(
        args.src_path_prefix,
        args.tgt_path_prefix,
        args.target_mp_size,
        args.model_type,
    )


if __name__ == "__main__":
    args = prepare_args()
    main(args)
