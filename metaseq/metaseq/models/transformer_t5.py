import logging

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

from metaseq.dataclass.constants import UNSPECIFIED_DOC_SEP

from metaseq import utils
from metaseq.dataclass import ChoiceEnum, MetaseqDataclass
from metaseq.models.base_model import TranslationModel
from metaseq.models import (
    register_model,
    register_model_architecture,
)
from metaseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    Embedding,
    TransformerEncoder,
    TransformerDecoder,
)

DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)


@dataclass
class TransformerT5ModelConfig(MetaseqDataclass):
    # NOTE: Defaults are T5 large
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"})

    no_final_layer_norm: bool = field(
        default=False, metadata={"help": "final layer norm at end of encoder"})
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension"})
    encoder_ffn_embed_dim: int = field(
        default=4096, metadata={"help": "encoder embedding dimension for FFN"})
    encoder_layers: int = field(
        default=24, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(
        default=16, metadata={"help": "num encoder attention heads"})
    encoder_normalize_before: bool = field(
        default=True,
        metadata={"help": "apply layernorm before each encoder block"})
    encoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the encoder"})

    decoder_embed_dim: int = field(
        default=1024, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=1024, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=1024, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=4096, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=24, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=16, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_learned_sinusoidal: bool = field(
        default=False,
        metadata={
            "help": "use learned positional embeddings init with sinusoidal in the decoder"
        },
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        },
    )
    # ALiBi
    alibi: bool = field(
        default=False,
        metadata={
            "help": "use the ALiBi position method instead of regular position embeddings"
        },
    )
    # Dynamic Attention
    self_attn_doc_sep: int = field(
        default=UNSPECIFIED_DOC_SEP,
        metadata={
            "help": "use dynamic self attention masking when document separator ID is specified"
        },
    )
    fsdp_checkpoint_wrap_layer_frequency: int = field(
        default=1,
        metadata={
            "help": "group transformer blocks and wrap the group in checkpoint and FSDP wrapper together"
        },
    )
    distribute_checkpointed_activations: bool = field(
        default=False,
        metadata={
            "help": "distribute offloaded checkpoints to tensor parallel gpus. "
            "It adds extra within node all_reduce but reduces checkpointed activations significantly,"
            "so a good way to trade speed for gpu memory."
        },
    )
    tensor_parallel_init_model_on_gpu: bool = field(
        default=False,
        metadata={
            "help": "initialize model directly on gpu and possibly fp16 for tensor parallel, shoudl be faster to init model."
        },
    )
    full_megatron_init: bool = field(
        default=False,
        metadata={"help": "Exact same init as Megatron"},
    )
    megatron_init_sigma: float = field(
        default=0.006,
        metadata={"help": "Sigma for megatron initialization"},
    )
    no_emb_dropout: Optional[bool] = field(
        default=False, metadata={"help": "Avoid emb dropout for decoder"}
    )

    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    fp16: bool = II("common.fp16")
    fp16_no_flatten_grads: bool = II("common.fp16_no_flatten_grads")
    ddp_backend: str = II("distributed_training.ddp_backend")
    world_size: int = II("distributed_training.distributed_world_size")
    distributed_rank: int = II("distributed_training.distributed_rank")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    model_parallel_size: int = II("common.model_parallel_size")


@register_model("transformer_t5", dataclass=TransformerT5ModelConfig)
class TransformerT5(TranslationModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

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

        encoder = TransformerEncoder(
            args,
            task.target_dictionary,
            embed_tokens,
        )

        decoder = TransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=False,
        )
        return cls(encoder, decoder)
