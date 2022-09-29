import logging

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

from metaseq.dataclass.constants import UNSPECIFIED_DOC_SEP

from metaseq import utils
from metaseq.dataclass import ChoiceEnum, MetaseqDataclass
from metaseq.models import (
    LanguageModel,
    register_model,
    register_model_architecture,
)
from metaseq.models.swiss_army_transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    Embedding,
    TransformerDecoder,
)


DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class SwissTransformerLanguageModelConfig(MetaseqDataclass):
    num_layers: int = field(default=24, metadata={"help": "Number of decoder layers"})
    hidden_size: int = field(
        default=1024, metadata={"help": "Transformer hidden dim size"}
    )
    num_attention_heads: int = field(
        default=16, metadata={"help": "Number of transformer attention heads"}
    )
    vocab_size: int = field(default=0, metadata={"help": "Vocab size for tokenization"})
    max_sequence_length: int = field(
        default=512, metadata={"help": "Max number of position embeddings to use"}
    )
    layernorm_order: str = field(
        default="pre", metadata={"help": "Order of layernorm (post, pre, sandwich)"}
    )
    inner_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Inner hidden size in MLP, None meaning 4 * hidden size"},
    )
    hidden_size_per_attention_head: Optional[int] = field(
        default=None,
        metadata={
            "help": "Hidden size per attention head in self and cross attention. None means hidden_sized / num_attention_heads"
        },
    )
    skip_init: bool = field(
        default=False, metadata={"help": "Skip model initialization"}
    )
    use_gpu_initialization: bool = field(
        default=False, metadata={"help": "Initialize model on GPU"}
    )
    layernorm_epsilon: float = field(
        default=1e-5, metadata={"help": "Layer norm epsilon"}
    )
    hidden_dropout: float = field(
        default=0.1, metadata={"help": "Dropout prob for hidden state"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "Dropout prob for attention weights"}
    )
    make_vocab_size_divisible_by: int = field(
        default=128,
        metadata={"help": "Pad the vocab size to be divisible by this value"},
    )
    sandwich_ln: bool = field(
        default=False, metadata={"help": "Add sandwich ln in cogview"}
    )
    block_lm: bool = field(
        default=False, metadata={"help": "Whether to use the BlockLM pre-training"}
    )
    masked_lm: bool = field(
        default=False, metadata={"help": "Whether to use the MLM objective"}
    )
    bert_prob: float = field(default=0.5, metadata={"help": ""})
    gpt_infill_prob: float = field(default=0.5, metadata={"help": ""})
    gpt_min_ratio: float = field(default=0.5, metadata={"help": ""})
    gap_sentence_prob: float = field(default=0.0, metadata={"help": ""})
    gap_sentence_ratio: float = field(default=0.15, metadata={"help": ""})
    avg_block_length: int = field(default=3, metadata={"help": ""})
    short_seq_prob: float = field(default=0.0, metadata={"help": ""})
    single_span_prob: float = field(default=0.0, metadata={"help": ""})
    task_mask: bool = field(
        default=False,
        metadata={"help": "Use different mask for generation and blank infilling"},
    )
    no_shuffle_block: bool = field(
        default=False,
        metadata={"help": "Not shuffle the blocks when filling the blank"},
    )
    no_block_position: bool = field(
        default=False,
        metadata={"help": "Use (rought) absolute positions instead of block positions"},
    )
    sentinel_token: bool = field(
        default=False,
        metadata={"help": "Use sentinel (mask) tokens to replace 2d position encoding"},
    )
    block_mask_prob: float = field(default=0.0, metadata={"help": ""})
    context_mask_ratio: float = field(default=0.0, metadata={"help": ""})
    random_position: bool = field(
        default=False,
        metadata={
            "help": "Use random start position to cover all the position embeddings"
        },
    )
    cloze_eval: bool = field(
        default=False, metadata={"help": "Evaluation dataset with cloze task"}
    )
    old_checkpoint: bool = field(
        default=False, metadata={"help": "Loading the checkpoint from old library"}
    )


class SwissArmyTransformerLanguageModel(LanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_input_dim
        )
        decoder = TransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return Embedding(
            len(dictionary),
            embed_dim,
            dictionary.pad(),
            initialize_params_on_gpu=getattr(
                args, "tensor_parallel_init_model_on_gpu", False
            ),
        )


