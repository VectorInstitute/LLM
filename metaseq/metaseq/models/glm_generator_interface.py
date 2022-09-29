from argparse import Namespace
import re
from typing import Optional, Dict, Any
import logging

import numpy as np

from metaseq import utils
from metaseq import tasks
from metaseq.file_io import PathManager, torch_load_cpu
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.checkpoint_utils import load_checkpoint_to_cpu
from metaseq.models.glm import GLMModel
from metaseq.models.glm_checkpoint_utils import update_args_with_file


logger = logging.getLogger(__name__)


def load_swiss_model_ensemble_and_task(
    cfg,        # NOTE: Need cfg since GLM doesn't save with one we can use
    filenames,  #       as a replacement
    arg_overrides: Optional[Dict[str, Any]] = None,
    task=None,
    strict=True,
    suffix="",
    num_shards=1,   # NOTE: Shards are not MP shards
    state=None,
    build_model_hook=None,
):
    ensemble = []
    # TODO (mchoi): Tasks in this case are not implemented yet
    for filename in filenames:
        orig_filename = filename
        assert num_shards > 0
        for shard_idx in range(num_shards):
            if num_shards == 1:
                filename = filename.replace(".pt", suffix + ".pt")
            else:
                filename = orig_filename[:-3] + f"_part{shard_idx}.pt"

            if PathManager.exists(filename):
                logger.info(f"found {filename}")
            else:
                logger.info(f"Could not find {filename}, changing the suffix")
                found_filename = ''
                suffix_candidates = [suffix+'-shard0', '-model_part-'+re.findall(r'\d+', suffix)[0]]
                for s in suffix_candidates:
                    filename = orig_filename
                    filename = filename.replace(".pt", s + ".pt")
                    if PathManager.exists(filename):
                        logger.info(f"found {filename}")
                        found_filename = filename
                filename = found_filename
                assert filename != '', "could not find reshard.pt to load"

            logger.info(f"Loading {filename} to cpu")
            if state is None:
                state = torch_load_cpu(filename)

            logger.info("build model from checkpoint cfg")
            model = build_model_hook(cfg, task)

            logger.info("load the checkpoint state dict")

            model.load_state_dict(state["model"], strict=strict)
            logger.info("Done loading state dict")
            # reset state so it gets loaded for the next model in ensemble
            state = None

        ensemble.append(model)

    breakpoint()
    if task is None:
        tasks.setup_task(cfg.task)

    return ensemble, task


class SwissArmyTransformerGeneratorInterface:
    """
    PyTorch hub interface for generating sequence from a pre-trained traslation
    or language model.
    """

    def __init__(self, cfg: MetaseqConfig):
        # Load the config from model.json file
        self.cfg = update_args_with_file(cfg)
        if isinstance(self.cfg, Namespace):
            self.cfg = convert_namespace_to_omegaconf(self.cfg)

    def load_model(self):
        utils.import_user_module(self.cfg.common)

        # Fix seed for stochastic decoding
        if (
            self.cfg.common.seed is not None
            and not self.cfg.generation.no_seed_provided
        ):
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        def _build_model(cfg, task):
            assert task is None
            model = GLMModel(cfg).cuda()
            return model

        logger.info("loading model(s) from {}".format(self.cfg.common_eval.path))
        model, task = load_swiss_model_ensemble_and_task(
            self.cfg,
            utils.split_paths(self.cfg.common_eval.path),
            arg_overrides=None,
            task=None,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
            build_model_hook=_build_model,
        )

        breakpoint()

        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # TODO (mchoi): Implement task for swissarmytransformer
        # Handle tokenization and BPE
        bpe = task.build_tokenizer(self.cfg.tokenizer)

        # Set state
        self.bpe = bpe
        self.task = task
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        # store special token indices for
        self._pad_token_ind = self.tgt_dict.pad_index
        self._special_token_inds = {
            self.tgt_dict.eos_index,
            self.tgt_dict.pad_index,
            self.tgt_dict.bos_index,
            self.tgt_dict.unk_index,
        }

        return models

    def generate():
        raise NotImplementedError
