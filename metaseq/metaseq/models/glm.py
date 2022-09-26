# This file is an all-encompassing module which contains all logic for
# constructing and loading the GLM model. Logic for inference can be found
# in the inference_glm.py file in the examples folder (in the SwissArmyTransformer
# repository).
#
# This version of the file is based off of the version within SwissArmyTransformer.
#

from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import os
import copy
import logging
import requests

import torch

from metaseq.dataclass import MetaseqDataclass
from metaseq.dataclass.utils import gen_parser_from_dataclass
from metaseq.models import register_model

from megatron.mpu.mappings import (
    gather_from_tensor_model_parallel_region,
    copy_to_tensor_model_parallel_region,
)
from megatron.mpu.layers import VocabParallelEmbedding
from metaseq.models.glm_checkpoint_utils import (
    update_args_with_file,
    get_model,
    load_checkpoint,
)
from metaseq.models.glm_download_utils import auto_create
from metaseq.models.glm_mixins import BaseMixin, non_conflict
from metaseq.models.glm_modules import (
    gelu,
    scaled_init_method,
    unscaled_init_method,
    BaseTransformerLayer,
)
from metaseq.models.glm_forward_defaults import (
    standard_attention,
    attention_forward_default,
    cross_attention_forward_default,
    mlp_forward_default,
    word_embedding_forward_default,
    position_embedding_forward_default,
    final_forward_default,
    layer_forward_default,
)

# Import fused layer norm with fallback to regular layer norm
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm

    class LayerNorm(FusedLayerNorm):
        def __init__(self, *args, pb_relax=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.pb_relax = pb_relax

        def forward(self, x):
            if not self.pb_relax:
                return super().forward(x)
            return super().forward(x / (x.abs().max().detach() / 8))

except ModuleNotFoundError:
    print("Install apex to use fused_layer_norm. Fallback: torch.nn.LayerNorm")
    from torch.nn import LayerNorm


logger = logging.getLogger(__name__)


HOOKS_DEFAULT = {
    "attention_fn": standard_attention,
    "attention_forward": attention_forward_default,
    "cross_attention_forward": cross_attention_forward_default,
    "mlp_forward": mlp_forward_default,
    "word_embedding_forward": word_embedding_forward_default,
    "position_embedding_forward": position_embedding_forward_default,
    "final_forward": final_forward_default,
    "layer_forward": layer_forward_default,
}


class BaseTransformer(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size,
        num_attention_heads,
        max_sequence_length,
        embedding_dropout_prob,
        attention_dropout_prob,
        output_dropout_prob,
        checkpoint_activations,
        checkpoint_num_layers=1,
        layernorm_epsilon=1.0e-5,
        init_method_std=0.02,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        layernorm_order="pre",
        parallel_output=True,
        is_decoder=False,
        use_bias=True,
        activation_func=gelu,
        layernorm=LayerNorm,
        init_method=None,
        use_final_layernorm=True,
        hooks={},
        params_dtype=torch.float,
        skip_init=False,
        device=torch.device("cpu"),
    ):
        super(BaseTransformer, self).__init__()

        # recording parameters
        self.is_decoder = is_decoder
        self.parallel_output = parallel_output
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_sequence_length = max_sequence_length
        self.layernorm_order = layernorm_order
        self.hooks = copy.copy(hooks)  # hooks will be updated each forward
        object.__setattr__(
            self, "transformer", self
        )  # to give the default hooks the same api as outer hooks

        # create embedding parameters
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # self.word_embeddings = VocabParallelEmbedding(
        #    num_embeddings=vocab_size, embedding_dim=hidden_size,
        #    params_dtype=params_dtype, skip_init=skip_init, device=device)
        # TODO: skip_init and device shouldn't be needed
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=None,  # Not used in GLM
            dtype=params_dtype,
        )

        self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(
            self.position_embeddings.weight, mean=0.0, std=init_method_std
        )

        # create all layers
        if init_method is None:
            self.output_layer_init_method = scaled_init_method(
                init_method_std, num_layers
            )
            self.init_method = unscaled_init_method(init_method_std)
        else:
            self.output_layer_init_method = init_method
            self.init_method = init_method

        def get_layer(layer_id):
            return BaseTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                self.init_method,
                layer_id,
                inner_hidden_size=inner_hidden_size,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=self.output_layer_init_method,
                is_decoder=self.is_decoder,
                layernorm_order=layernorm_order,
                layernorm=layernorm,
                use_bias=use_bias,
                activation_func=activation_func,
                hooks=self.hooks,
                transformer_pointer=self,
                params_dtype=params_dtype,
                skip_init=skip_init,
                device=device,
            )

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)]
        )

        # Final layer norm before output.
        self.use_final_layernorm = use_final_layernorm
        if use_final_layernorm:
            self.final_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        *,
        output_hidden_states=False,
        **kw_args,
    ):
        # sanity check
        assert len(input_ids.shape) >= 2
        batch_size, query_length = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones(1, 1, device=input_ids.device).type_as(
                next(self.parameters())
            )  # None means full attention
        assert (
            len(attention_mask.shape) == 2
            or len(attention_mask.shape) == 4
            and attention_mask.shape[1] == 1
        )

        # initial output_cross_layer might be generated by word/position_embedding_forward
        output_cross_layer = {}

        # embedding part
        if "word_embedding_forward" in self.hooks:
            hidden_states = self.hooks["word_embedding_forward"](
                input_ids, output_cross_layer=output_cross_layer, **kw_args
            )
        else:  # default
            hidden_states = HOOKS_DEFAULT["word_embedding_forward"](
                self, input_ids, output_cross_layer=output_cross_layer, **kw_args
            )

        if "position_embedding_forward" in self.hooks:
            position_embeddings = self.hooks["position_embedding_forward"](
                position_ids, output_cross_layer=output_cross_layer, **kw_args
            )
        else:
            assert len(position_ids.shape) <= 2
            assert position_ids.shape[-1] == hidden_states.shape[1], (
                position_ids.shape,
                hidden_states.shape,
            )
            position_embeddings = HOOKS_DEFAULT["position_embedding_forward"](
                self, position_ids, output_cross_layer=output_cross_layer, **kw_args
            )
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        output_per_layers = []
        if self.checkpoint_activations:
            # define custom_forward for checkpointing
            def custom(start, end, kw_args_index, cross_layer_index):
                def custom_forward(*inputs):
                    layers_ = self.layers[start:end]
                    x_, mask = inputs[0], inputs[1]

                    # recover kw_args and output_cross_layer
                    flat_inputs = inputs[2:]
                    kw_args, output_cross_layer = {}, {}
                    for k, idx in kw_args_index.items():
                        kw_args[k] = flat_inputs[idx]
                    for k, idx in cross_layer_index.items():
                        output_cross_layer[k] = flat_inputs[idx]
                    # -----------------

                    output_per_layers_part = []
                    for i, layer in enumerate(layers_):
                        output_this_layer_obj, output_cross_layer_obj = {}, {}
                        if "layer_forward" in self.hooks:
                            layer_ret = self.hooks["layer_forward"](
                                x_,
                                mask,
                                layer_id=layer.layer_id,
                                **kw_args,
                                **output_cross_layer,
                                output_this_layer=output_this_layer_obj,
                                output_cross_layer=output_cross_layer_obj,
                            )
                        else:
                            layer_ret = layer(
                                x_,
                                mask,
                                layer_id=layer.layer_id,
                                **kw_args,
                                **output_cross_layer,
                                output_this_layer=output_this_layer_obj,
                                output_cross_layer=output_cross_layer_obj,
                            )
                        if isinstance(layer_ret, tuple):
                            layer_ret = layer_ret[0]  # for legacy API
                        x_, output_this_layer, output_cross_layer = (
                            layer_ret,
                            output_this_layer_obj,
                            output_cross_layer_obj,
                        )
                        if output_hidden_states:
                            output_this_layer["hidden_states"] = x_
                        output_per_layers_part.append(output_this_layer)

                    # flatten for re-aggregate keywords outputs
                    flat_outputs = []
                    for output_this_layer in output_per_layers_part:
                        for k in output_this_layer:
                            # TODO add warning for depth>=2 grad tensors
                            flat_outputs.append(output_this_layer[k])
                            output_this_layer[k] = len(flat_outputs) - 1
                    for k in output_cross_layer:
                        flat_outputs.append(output_cross_layer[k])
                        output_cross_layer[k] = len(flat_outputs) - 1
                    # --------------------

                    return (
                        x_,
                        output_per_layers_part,
                        output_cross_layer,
                        *flat_outputs,
                    )

                return custom_forward

            # prevent to lose requires_grad in checkpointing.
            # To save memory when only finetuning the final layers, don't use checkpointing.
            if self.training:
                hidden_states.requires_grad_(True)

            l, num_layers = 0, len(self.layers)
            chunk_length = self.checkpoint_num_layers
            output_this_layer = []
            while l < num_layers:
                args = [hidden_states, attention_mask]
                # flatten kw_args and output_cross_layer
                flat_inputs, kw_args_index, cross_layer_index = [], {}, {}
                for k, v in kw_args.items():
                    flat_inputs.append(v)
                    kw_args_index[k] = len(flat_inputs) - 1
                for k, v in output_cross_layer.items():
                    flat_inputs.append(v)
                    cross_layer_index[k] = len(flat_inputs) - 1
                # --------------------
                # TODO: Get rid of all deepspeed calls
                # hidden_states, output_per_layers_part, output_cross_layer, *flat_outputs = \
                #    checkpoint(custom(l, l + chunk_length, kw_args_index, cross_layer_index), *args, *flat_inputs)
                (
                    hidden_states,
                    output_per_layers_part,
                    output_cross_layer,
                    *flat_outputs,
                ) = custom(l, l + chunk_length, kw_args_index, cross_layer_index)

                # recover output_per_layers_part, output_cross_layer
                for output_this_layer in output_per_layers_part:
                    for k in output_this_layer:
                        output_this_layer[k] = flat_outputs[output_this_layer[k]]
                for k in output_cross_layer:
                    output_cross_layer[k] = flat_outputs[output_cross_layer[k]]
                # --------------------

                output_per_layers.extend(output_per_layers_part)
                l += chunk_length
        else:
            output_this_layer = []
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask]

                output_this_layer_obj, output_cross_layer_obj = {}, {}

                if "layer_forward" in self.hooks:  # customized layer_forward
                    layer_ret = self.hooks["layer_forward"](
                        *args,
                        layer_id=torch.tensor(i),
                        **kw_args,
                        position_ids=position_ids,
                        **output_cross_layer,
                        output_this_layer=output_this_layer_obj,
                        output_cross_layer=output_cross_layer_obj,
                    )
                else:
                    layer_ret = layer(
                        *args,
                        layer_id=torch.tensor(i),
                        **kw_args,
                        **output_cross_layer,
                        output_this_layer=output_this_layer_obj,
                        output_cross_layer=output_cross_layer_obj,
                    )
                if isinstance(layer_ret, tuple):
                    layer_ret = layer_ret[0]  # for legacy API
                hidden_states, output_this_layer, output_cross_layer = (
                    layer_ret,
                    output_this_layer_obj,
                    output_cross_layer_obj,
                )

                if output_hidden_states:
                    output_this_layer["hidden_states"] = hidden_states
                output_per_layers.append(output_this_layer)

        # Final layer norm.
        if self.use_final_layernorm:
            logits = self.final_layernorm(hidden_states)
        else:
            logits = hidden_states

        logits = copy_to_tensor_model_parallel_region(logits)
        if "final_forward" in self.hooks:
            logits_parallel = self.hooks["final_forward"](logits, **kw_args)
        else:
            logits_parallel = HOOKS_DEFAULT["final_forward"](self, logits, **kw_args)

        if not self.parallel_output:
            logits_parallel = gather_from_tensor_model_parallel_region(logits_parallel)

        outputs = [logits_parallel]
        outputs.extend(output_per_layers)

        return outputs


class BaseModel(torch.nn.Module):
    def __init__(self, cfg, transformer=None, params_dtype=torch.float, **kwargs):
        super(BaseModel, self).__init__()
        self.mixins = torch.nn.ModuleDict()
        self.collect_hooks_()
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = BaseTransformer(
                num_layers=cfg.model.num_layers,
                vocab_size=cfg.model.vocab_size,
                hidden_size=cfg.model.hidden_size,
                num_attention_heads=cfg.model.num_attention_heads,
                max_sequence_length=cfg.model.max_sequence_length,
                embedding_dropout_prob=cfg.model.hidden_dropout,
                attention_dropout_prob=cfg.model.attention_dropout,
                output_dropout_prob=cfg.model.hidden_dropout,
                inner_hidden_size=cfg.model.inner_hidden_size,
                hidden_size_per_attention_head=cfg.model.hidden_size_per_attention_head,
                # checkpoint_activations=cfg.model.checkpoint_activations, # TODO: Deepspeed turned off
                checkpoint_activations=False,
                # checkpoint_num_layers=cfg.model.checkpoint_num_layers, #TODO: Deepspeed turned off
                checkpoint_num_layers=1,
                layernorm_order=cfg.model.layernorm_order,
                hooks=self.hooks,
                params_dtype=params_dtype,
                skip_init=cfg.model.skip_init,
                device=torch.cuda.current_device()
                if cfg.model.use_gpu_initialization
                else torch.device("cpu"),
                **kwargs,
            )

    def reinit(
        self, mixin_names=None
    ):  # will be called when loading model, None means all
        # if some mixins are loaded, overrides this function
        for k, m in self.mixins.items():
            if k in mixin_names or mixin_names is None:
                m.reinit(self)

    def add_mixin(self, name, new_mixin, reinit=False):
        assert name not in self.mixins
        assert isinstance(new_mixin, BaseMixin)

        self.mixins[name] = new_mixin  # will auto-register parameters
        object.__setattr__(
            new_mixin, "transformer", self.transformer
        )  # cannot use pytorch set_attr

        self.collect_hooks_()
        if reinit:
            new_mixin.reinit(self)  # also pass current mixins

    def del_mixin(self, name):
        assert name in self.mixins
        del self.mixins[name]
        self.collect_hooks_()

    def get_mixin(self, name):
        return self.mixins[name]

    def forward(self, *args, **kwargs):
        # update hooks as the current model (overrided forwards)
        # Attention! the transformer might be shared by multiple models
        self.transformer.hooks.clear()
        self.transformer.hooks.update(self.hooks)
        return self.transformer(*args, **kwargs)

    def collect_hooks_(self):
        names = list(HOOKS_DEFAULT.keys())
        hooks = {}
        hook_origins = {}
        for name in names:
            if hasattr(self, name):
                hooks[name] = getattr(self, name)
                hook_origins[name] = "model"

            for mixin_name, m in self.mixins.items():
                if hasattr(m, name):
                    if hasattr(getattr(m, name), "non_conflict"):
                        if name in hooks:
                            old_impl = hooks[name]
                        elif name == "attention_fn":  # the only hook without self
                            old_impl = HOOKS_DEFAULT[name]
                        else:
                            old_impl = partial(HOOKS_DEFAULT[name], self)
                        old_origin = hook_origins.get(name, "default")
                        hooks[name] = partial(getattr(m, name), old_impl=old_impl)
                        hook_origins[name] = mixin_name + " -> " + old_origin
                    elif name in hooks:  # if this hook name is already registered
                        raise ValueError(
                            f"Hook {name} conflicts at {mixin_name} and {hook_origins[name]}."
                        )
                    else:  # new hook
                        hooks[name] = getattr(m, name)
                        hook_origins[name] = mixin_name

        self.hooks = hooks
        self.hook_origins = hook_origins
        return hooks

    def disable_untrainable_params(self):
        pass

    @classmethod
    def from_pretrained(cls, cfg, name, *, home_path=None, url=None, prefix=""):
        model_path = auto_create(name, path=home_path, url=url)
        cfg = update_args_with_file(
            cfg, path=os.path.join(model_path, "model_config.json")
        )
        model = get_model(cfg, cls)
        load_checkpoint(model, cfg, load_path=model_path, prefix=prefix)
        return model, cfg


class CachedAutoregressiveMixin(BaseMixin):
    def __init__(self):
        super().__init__()

    @non_conflict
    def attention_fn(
        self,
        q,
        k,
        v,
        mask,
        dropout_fn,
        mems=None,
        cross_attention=False,
        old_impl=standard_attention,
        **kw_args,
    ):
        if not cross_attention:
            mem = (
                mems[kw_args["layer_id"]] if mems is not None else None
            )  # 2, batch, head, seqlen, hidden_size
            b, nh, seq_len, hidden_size = k.shape

            cache_kv = (
                torch.stack((k, v))
                .permute(1, 3, 0, 2, 4)
                .detach()
                .contiguous()
                .view(b, seq_len, nh * hidden_size * 2)
            )
            kw_args["output_this_layer"]["mem_kv"] = cache_kv

            if mem is not None:  # the first time, mem is None
                # might change batch_size
                mem = (
                    mem.expand(b, -1, -1)
                    .reshape(b, mem.shape[1], 2, nh, hidden_size)
                    .permute(2, 0, 3, 1, 4)
                )
                memk, memv = mem[0], mem[1]
                k = torch.cat((memk, k), dim=2)
                v = torch.cat((memv, v), dim=2)
        return old_impl(
            q,
            k,
            v,
            mask,
            dropout_fn,
            cross_attention=cross_attention,
            mems=mems,
            **kw_args,
        )


class CachedAutoregressiveModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.add_mixin("auto-regressive", CachedAutoregressiveMixin())


class BlockPositionEmbeddingMixin(BaseMixin):
    def __init__(self, max_sequence_length, hidden_size, init_method_std=0.02):
        super(BlockPositionEmbeddingMixin, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.block_position_embeddings = torch.nn.Embedding(
            max_sequence_length, hidden_size
        )
        torch.nn.init.normal_(
            self.block_position_embeddings.weight, mean=0.0, std=init_method_std
        )

    def position_embedding_forward(self, position_ids, **kwargs):
        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.transformer.position_embeddings(position_ids)
        block_position_embeddings = self.block_position_embeddings(block_position_ids)
        return position_embeddings + block_position_embeddings


@dataclass
class GLMModelConfig(MetaseqDataclass):
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
    tokenizer_model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model type to use for sentencepiece tokenization [bpe, char, unigram, word]"
        },
    )


@register_model("glm_large", dataclass=GLMModelConfig)
class GLMModel(BaseModel):
    def __init__(self, cfg, transformer=None, parallel_output=True):
        super().__init__(cfg, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin(
            "block_position_embedding",
            BlockPositionEmbeddingMixin(
                cfg.model.max_sequence_length, cfg.model.hidden_size
            ),
        )

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            # do not set defaults so that settings defaults from various architectures still works
            gen_parser_from_dataclass(parser, dc(), delete_default=True)
