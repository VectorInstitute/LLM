import argparse
from collections import defaultdict, OrderedDict
from functools import partial
from glob import glob
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm

import torch

from metaseq.distributed.stitch_fsdp_ckpt import reshard_megatron_parts, find_num_parts
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


def upgrade_glm_sd(sd, prefix):
    """
    Pre-processing function for GLM/SwissArmyTransformer checkpoints (decoder-
    only currently). First changes top-level key to model, similar to metaseq.
    """
    new_sd = {"model": {}}  # Replace module with model for consistency with OPT
    for k in sd:
        if k != "module":
            new_sd[k] = sd[k]

    for k in sd["module"]:
        if k.startswith(prefix):
            new_sd["model"][k[len(prefix) :]] = sd["module"][k]

    return new_sd


def _get_model_part_num(filename):
    return int(os.path.basename(filename).replace(".pt", "").split("-")[-1])


def reshard_glm_parts(model_parts, new_model_part_count=1):
    """
    Reshard to a different number of model parts for GLM SwissArmyTransformer-
    based model.
    """
    new_model_parts = [OrderedDict() for _ in range(new_model_part_count)]

    # TODO (mchoi): Add this aliasing to convert from glm to metaseq-like
    #               submodule/layer names. This should be done after the
    #               glm model merge has completed, else we can't load up the
    #               re-aliased resharded glm checkpoints.
    """
    glm_metaseq_submodule_aliases = {
        "transformer": "decoder",
    }
    """

    def assert_all_close(key):
        for part_id in range(len(model_parts)):
            if not torch.allclose(model_parts[part_id][key], model_parts[0][key]):
                err = (
                    (model_parts[part_id][key] - model_parts[0][key])
                    .float()
                    .abs()
                    .max()
                    .item()
                )
                print(f"max discrepancy {key}: {err}")

    def consolidate_and_reshard(key, dim):
        consolidated_tensor = torch.cat(
            [model_parts[part_id][key] for part_id in range(len(model_parts))],
            dim=dim,
        )
        assert consolidated_tensor.size(dim) % new_model_part_count == 0

        newly_resharded_tensors = torch.chunk(
            consolidated_tensor,
            new_model_part_count,
            dim=dim,
        )
        for i in range(new_model_part_count):
            new_model_parts[i][key] = newly_resharded_tensors[i].clone()

    def copy_key_to_all_parts(key):
        assert_all_close(key)

        for new_model_part in new_model_parts:
            new_model_part[key] = model_parts[0][key].clone()

    def handle_qkv_proj_glm(key):
        parts = [model_parts[part_id][key] for part_id in range(len(model_parts))]

        # Scatter each MP part along qkv
        ks, vs, qs = [], [], []
        for p in parts:
            k, v, q = torch.split(p, p.shape[0] // 3)
            ks.append(k)
            vs.append(v)
            qs.append(q)

        # Gather old MP, then scatter into new MP
        resharded_ks = torch.chunk(torch.cat(ks, dim=0), new_model_part_count)
        resharded_vs = torch.chunk(torch.cat(vs, dim=0), new_model_part_count)
        resharded_qs = torch.chunk(torch.cat(qs, dim=0), new_model_part_count)

        # Gather each MP part along qkv
        for i in range(new_model_part_count):
            new_model_parts[i][key] = torch.cat(
                (resharded_ks[i], resharded_vs[i], resharded_qs[i]), dim=0
            )

    def handle_layer_norm_and_position_embeddings_glm(key):
        # Make sure that duplicated layernorms have duplicate weights
        # NOTE: May not be necessary in GLM, but necessary in OPT
        copy_key_to_all_parts(key)

    def handle_dense_glm(key):
        # Pytorch does left multiplication, so everything is transposed
        if "dense_h_to_4h" in key:
            # ColumnParallel weight and bias split along outer
            if key.endswith("weight") or key.endswith("bias"):
                consolidate_and_reshard(key, dim=0)
            else:
                raise KeyError(f"Key: {key} doesn't exist")

        elif "dense_4h_to_h" in key or "dense" in key:
            # RowParallel weight split along inner, bias replicated
            if key.endswith("weight"):
                consolidate_and_reshard(key, dim=1)

            elif key.endswith("bias"):
                copy_key_to_all_parts(key)

            else:
                raise KeyError(f"Key: {key} doesn't exist")

    def handle_word_embeddings_glm(key):
        consolidate_and_reshard(key, dim=0)

    for key in model_parts[0]:
        if "attention" in key and "layernorm" not in key:
            if "query_key_value" in key:
                handle_qkv_proj_glm(key)
            elif "dense" in key:
                handle_dense_glm(key)

        elif "mlp" in key:
            handle_dense_glm(key)

        elif (
            "input_layernorm" in key
            or "post_attention_layernorm" in key
            or "final_layernorm" in key
            or "position_embeddings" in key
        ):
            handle_layer_norm_and_position_embeddings_glm(key)

        elif "word_embeddings" in key:
            handle_word_embeddings_glm(key)

        else:
            raise KeyError(f"Key: {key} doesn't exist")

    assert model_parts[0].keys() == new_model_parts[0].keys(), "Keys don't match!"

    return new_model_parts


def reshard_model_parallel(
    src_path_prefix,
    tgt_path_prefix,
    target_mp_size,
    model_type="opt",
):
    """
    Reshard model parallel checkpoint to a different MP size. Model type can be
    opt or glm. Note that opt and glm have different ways to case out the
    resharding of model parallel layers.
    Usage:
        GLM:
            src_path_prefix = "/checkpoint/opt_test/original/TEST_glm_merge/glm-large-en-blank/250000/glmreshard"
            tgt_path_prefix = "/checkpoint/opt_test/original/TEST_glm_merge/glm-large-en-blank/250000/glmreshard"
            target_mp_size = 2
            model_type="glm"
        OPT:
            src_path_prefix = "/checkpoint/opt_test/original/TEST_glm_merge/OPT-125M/megatronreshard"
            tgt_path_prefix = "/checkpoint/opt_test/original/TEST_glm_merge/OPT-125M/megatronreshard"
            target_mp_size = 4
            model_type="opt"
    """
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
    t0 = time.time()

    load_fn = load_and_pop_last_optimizer_state if model_type == "opt" else torch.load
    assert model_type in ["opt", "glm"], f"Invalid model type selected: {model_type}"

    format_fn = (
        partial(upgrade_glm_sd, prefix="") if model_type == "glm" else lambda x: x
    )

    for p in tqdm(all_ckpt_files):
        names.append(Path(p).name)
        ckpt = load_fn(p)
        ckpt = format_fn(ckpt)
        weights.append(ckpt["model"])

    num_parts = find_num_parts(names) if len(all_ckpt_files) > 1 else 1

    model_parts = defaultdict()

    assert len(weights) == num_parts
    for p in range(num_parts):
        model_parts[p] = weights[p]

    # Reshard the weights in the dict
    reshard_fn = reshard_megatron_parts if model_type == "opt" else reshard_glm_parts

    resharded_models = reshard_fn(model_parts, target_mp_size)

    # Save the resharded checkpoint
    def save_checkpoint(weights_to_save, prefix):
        if model_type == "opt":
            ckpt_resharded = dict(
                model=weights_to_save,
                cfg=ckpt["cfg"],
                extra_state=ckpt["extra_state"],
                optimizer_history=ckpt["optimizer_history"],
                args=ckpt.get("args"),
            )
        # TODO (mchoi): Make GLM save with similar cfg when configs are merged
        elif model_type == "glm":
            ckpt_resharded = dict(
                model=weights_to_save,
                cfg=None,
                extra_state=None,
                optimizer_history=None,
                args=None,
            )
        save_path = f"{prefix}.pt"
        logger.info(f"Saving to {save_path} ...")
        torch.save(ckpt_resharded, save_path)
        logger.info(f"Done after {time.time()-t0//60} minutes")
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
        args.src_path_prefix, args.tgt_path_prefix, args.target_mp_size, args.model_type
    )


if __name__ == "__main__":
    args = prepare_args()
    main(args)
