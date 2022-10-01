import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from omegaconf import II

from metaseq import utils
from metaseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from metaseq.data.indexed_dataset import get_available_dataset_impl
from metaseq.data.shorten_dataset import maybe_shorten_dataset
from metaseq.dataclass import ChoiceEnum, MetaseqDataclass
from metaseq.tasks import LegacyTask, register_task


try:
    from transformers import T5TokenizerFast

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig(MetaseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    # Begin args from StreamingLanguageModelingConfig
    vocab_filename: Optional[str] = field(
        default="", metadata={"help": "path to tokenizer.json"}
    )
    end_of_document_symbol: Optional[str] = field(
        default="</s>", metadata={"help": "symbol indicating an end-of-document"}
    )
    final_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "force vocab size to this"}
    )  # End of args from StreamingLanguageModelingConfig
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False,
        metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False,
        metadata={"help": "boolean to pad to fixed batch size"},
    )
    shuffle_docs: bool = field(
        default=False,
        metadata={
            "help": "Only for sample break mode EOS, first shuffle docs and then create blocks"
        },
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task("translation", dataclass=TranslationConfig)
class TranslationTask(LegacyTask):
    """
    Train a translation model.
    """
    def __init__(self, args):
        super().__init__(args)

        if not has_hf_tokenizers:
            raise ImportError("Please install tokenizers with: pip install tokenizers")

        self.tokenizer = T5TokenizerFast.from_pretrained(args.vocab_filename)

        self.eod = self.tokenizer.convert_tokens_to_ids(args.end_of_document_symbol)
        if self.eod is None:
            # This will be executed for old models that do not have the args.end_of_document_symbol explicitly set
            # and do not use <s/> (the default) but <EOS>
            self.eod = self.tokenizer.convert_tokens_to_ids("<EOS>")

        assert (
            self.eod is not None
        ), "Cannot find end-of-document symbol ({}) in tokenizer".format(
            args.end_of_document_symbol
        )

        # construct a dummy metaseq Dictionary corresponding to the given tokenizer
        self.dictionary = Dictionary(bos=None)
        tok_vocab_size = self.tokenizer.vocab_size

        for id in range(self.dictionary.nspecial, tok_vocab_size):
            self.dictionary.add_symbol(self.tokenizer.convert_ids_to_tokens(id))
        final_vocab_size = args.final_vocab_size
        # final_vocab_size = 51200 for roberta dictionary
        if final_vocab_size is not None:
            if final_vocab_size < tok_vocab_size:
                raise ValueError(
                    f"incompatible: {final_vocab_size}, tok_vocab_size: {tok_vocab_size}"
                )
            self.dictionary.pad_to_multiple_(final_vocab_size)
        else:
            self.dictionary.pad_to_multiple_(8)

        # confirm that metaseq dictionary and BPE have matching special symbols
        assert self.dictionary.pad_index == 0
        assert self.tokenizer.convert_ids_to_tokens(0) in {"<PAD>", "<pad>"}
        assert self.dictionary.eos_index == 1
        assert self.tokenizer.convert_ids_to_tokens(1) in {"<EOS>", "</s>"}
        assert self.dictionary.unk_index == 2
        assert self.tokenizer.convert_ids_to_tokens(2) in {"<UNK>", "<unk>"}

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None, **kwargs):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                bos_token=bos_token,
                **kwargs,
            )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
