import torch
import torch.nn as nn
from metaseq.model_parallel.models.transformer import (
    ModelParallelTransformerEncoder,
    ModelParallelTransformerDecoder,
)
from metaseq.models import register_model, register_model_architecture
from metaseq.models.transformer_t5 import TransformerT5


try:
    from megatron.mpu import VocabParallelEmbedding

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


DEFAULT_MAX_TARGET_POSITIONS = 1024


# Main T5 model
@register_model("model_parallel_transformer_t5")
class ModelParallelTransformerT5(TransformerT5):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )

        base_t5_architecture(args)

        task.source_dictionary.pad_to_multiple_(8)
        task.target_dictionary.pad_to_multiple_(8)


        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_input_dim
        )
        assert getattr(
            args, "use_sharded_state", False
        ), "Use sharded state must be True for tensor parallel, otherwise model saving and loaded might be broken"

        encoder = ModelParallelTransformerEncoder(
            args,
            task.target_dictionary,
            embed_tokens,
        )

        decoder = ModelParallelTransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=False,
        )
        return cls(encoder, decoder)

    @staticmethod
    def add_args(parser):
        TransformerT5.add_args(parser)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        def _vocab_init(tensor, **kwargs):
            nn.init.normal_(tensor, mean=0, std=embed_dim**-0.5)
            nn.init.constant_(tensor[1], 0)

        def _vocab_init_megatron(tensor, **kwargs):
            nn.init.normal_(
                tensor, mean=0, std=getattr(args, "megatron_init_sigma", 0.006)
            )
            nn.init.constant_(tensor[1], 0)

        if getattr(args, "memory_efficient_fp16", False):
            dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.half
        else:
            dtype = torch.float32

        embed_tokens = VocabParallelEmbedding(
            len(dictionary),
            embed_dim,
            dictionary.pad(),
            init_method=_vocab_init_megatron
            if getattr(args, "full_megatron_init", False)
            else _vocab_init,
            use_cpu_initialization=not getattr(
                args, "tensor_parallel_init_model_on_gpu", False
            ),
            dtype=dtype,
        )
        return embed_tokens


def base_t5_architecture(args):
    # TODO (mchoi): Change the deffaults to T5
    # NOTE: Defaults are for T5 large
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.no_final_layer_norm = getattr(args, "no_final_layer_norm", False)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_learned_sinusoidal = getattr(args, "decoder_learned_sinusoidal", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.add_bos_token = getattr(args, "add_bos_token", False)


@register_model_architecture("model_parallel_transformer_t5", "transformer_t5_large")
def transformer_t5_megatron(args):
    # TODO (mchoi): Change the defaults to T5
    # NOTE: Defaults are for T5 large
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.no_final_layer_norm = getattr(args, "no_final_layer_norm", False)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    
    base_t5_architecture(args)
