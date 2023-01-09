from functools import partial
import logging
from typing import Tuple, List, Union

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor

from megatron.mpu.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from megatron.mpu.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from metaseq.modules.layer_norm import FusedLayerNorm
from metaseq.modules.dropout import Dropout
from metaseq.model_parallel.modules.transformer_layer import (
    ModelParallelTransformerDecoderLayer,
)
from metaseq.model_parallel.modules.multihead_attention import (
    ModelParallelMultiheadAttention,
)
from metaseq.model_parallel.models.transformer import (
    ModelParallelTransformerDecoder,
)
from transformers.models.opt.modeling_opt import (
    OPTDecoderLayer,
    OPTAttention,
    OPTLearnedPositionalEmbedding,
)
from metaseq.quantization import (
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear,
)


class ActivationManipulationFunctions:
    # Gather functions
    def column_parallel_linear_gather(registered_name, module, x, aux):
        if not module.gather_output:
            output = (gather_from_tensor_model_parallel_region(x[0]), x[1:])
        return output

    def self_attn_dropout_gather(registered_name, module, x, aux):
        if "self_attn" in registered_name:
            # NOTE: Newest megatron supports both first and last dim
            # megatron only gathers the last dim, but we want to
            # gather on the first dim so we permute it to the end
            # and then permute it back
            assert aux is not None, (f"Rank {torch.distributed.get_rank()}: "
                                     f"Required auxillary data for "
                                     f"self-attention maps is not present")

            output = gather_from_tensor_model_parallel_region(
                rearrange(x, "(b k) s1 s2 -> s1 s2 b k", b=aux[0]))
        else:
            output = x

        return output

    def output_projection_gather(registered_name, module, x, aux):
        output = gather_from_tensor_model_parallel_region(x)
        return output

    # Shard functions
    def column_parallel_linear_scatter(registered_name, module, x, aux):
        if not module.gather_output:
            output = (scatter_to_tensor_model_parallel_region(x[0]), x[1:])
        return output

    def self_attn_dropout_scatter(registered_name, module, x, aux):
        if "self_attn" in registered_name:
            output = scatter_to_tensor_model_parallel_region(x)
            output = rearrange(output, "s1 s2 b k -> (b k) s1 s2")
        else:
            output = x

        return output

    def output_projection_scatter(registered_name, module, x, aux):
        output = scatter_to_tensor_model_parallel_region(x)
        return output

    # Rearrange functions
    def column_parallel_linear_rearrange(registered_name, module, x, aux):
        """
        Edits can be done either before bias addition or after bias addition.
        In the latter case, be careful since we will naively subtract out the
        bias again when the hook returns the edited activation, since the next
        layer in the model will add it back again.

        The forward (fwd) function returns the rearranged activation, which is
        the first return value. The second return value is the backward (bwd)
        function, which only needs to be run on the edited activation
        downstream to prepare for the hook return.
        """
        layer_type = registered_name.split(".")[-1]
        activation = x[0]
        bias = x[1]

        def fwd_fn(fwd_activation):
            # Manually add bias if needed
            if module.skip_bias_add:
                fwd_output = fwd_activation + bias

            # not always S x B x D, it's only when it's used in qkv_proj
            # NOTE: OPT's qkv_proj combined projection is hard to compare with HF
            if layer_type in ["q_proj", "k_proj", "v_proj", "qkv_proj"]:
                fwd_output = rearrange(fwd_output, "s b d -> b s d")

            elif "fc" in layer_type:
                fwd_output = rearrange(fwd_output, "(s b) d -> b s d", b=aux[0])

            # Need to return output and bias for this to be reverisble
            return fwd_output

        def bwd_fn(bwd_activation):
            if layer_type in ["q_proj", "k_proj", "v_proj", "qkv_proj"]:
                bwd_output = rearrange(bwd_activation, "b s d -> s b d")
            elif "fc" in layer_type:
                bwd_output = rearrange(bwd_activation, "b s d -> (s b) d")

            if module.skip_bias_add:
                bwd_output = bwd_output - bias

            return bwd_output

        return fwd_fn(activation), bwd_fn

    def row_parallel_linear_rearrange(registered_name, module, x, aux):
        layer_type = registered_name.split(".")[-1]
        activation = x[0]
        bias = x[1]

        def fwd_fn(fwd_activation):
            # Manually add bias if needed
            if module.skip_bias_add:
                fwd_output = fwd_activation + bias

            # not always S x B x D, it's only when it's used in out_proj
            if layer_type == "out_proj":
                fwd_output = rearrange(fwd_output, "s b d -> b s d")

            elif "fc" in layer_type:
                fwd_output = rearrange(fwd_output, "(s b) d -> b s d", b=aux[0])

            return fwd_output

        def bwd_fn(bwd_activation):
            if layer_type == "out_proj":
                bwd_output = rearrange(bwd_activation, "b s d -> s b d")

            elif "fc" in layer_type:
                bwd_output = rearrange(bwd_activation, "b s d -> (s b) d")

            if module.skip_bias_add:
                bwd_output = bwd_output - bias

            return bwd_output

        return fwd_fn(activation), bwd_fn

    def model_parallel_decoder_rearrange(registered_name, module, x, aux):
        activation = x[0]
        auxillary_output = x[1:]

        def fwd_fn(fwd_activation):
            return fwd_activation

        def bwd_fn(bwd_activation):
            # Preserve auxillary data for downstream use
            return tuple(bwd_activation, *auxillary_output)

        return fwd_fn(activation), bwd_fn

    def model_parallel_decoder_layer_rearrange(registered_name, module, x, aux):
        activation = x[0]
        auxillary_output = x[1:]

        def fwd_fn(fwd_activation):
            fwd_output = rearrange(fwd_activation, "s b d -> b s d")
            return fwd_output

        def bwd_fn(bwd_activation):
            bwd_output = rearrange(bwd_activation, "b s d -> s b d")
            return tuple(bwd_output, *auxillary_output)

        return fwd_fn(activation), bwd_fn

    def fused_layer_norm_rearrange(registered_name, module, x, aux):
        activation = x

        def fwd_fn(fwd_activation):
            fwd_output = rearrange(fwd_activation, "s b d -> b s d")
            return fwd_output

        def bwd_fn(bwd_activation):
            bwd_output = rearrange(bwd_activation, "b s d -> s b d")
            return bwd_output

        return fwd_fn(activation), bwd_fn

    def model_parallel_multihead_attn_rearrange(registered_name, module, x, aux):
        activation, bias = x[0]

        attn_weights = x[1]

        def fwd_fn(fwd_activation):
            if bias is not None:
                fwd_output = fwd_activation + bias

            fwd_output = rearrange(fwd_output, "s b d -> b s d")

            return fwd_output

        def bwd_fn(bwd_activation):
            bwd_output = rearrange(bwd_activation, "b s d -> s b d")

            if bias is not None:
                bwd_output = bwd_activation - bias

            return tuple((bwd_output, bias), attn_weights)

        return fwd_fn(activation), bwd_fn

    def self_attn_dropout_rearrange(registered_name, module, x, aux):
        activation = x

        def fwd_fn(fwd_activation):
            if "self_attn" in registered_name:
                fwd_output = rearrange(fwd_activation, "s1 s2 b k -> b k s1 s2")

            else:
                fwd_output = rearrange(fwd_activation, "s b d -> b s d")

            return fwd_output

        def bwd_fn(bwd_activation):
            if "self_attn" in registered_name:
                bwd_output = rearrange(bwd_activation, "b k s1 s2 -> s1 s2 b k")

            else:
                bwd_output = rearrange(bwd_activation, "b s d -> s b d")

            return bwd_output

        return fwd_fn(activation), bwd_fn

    def vocab_parallel_and_out_proj_rearrange(registered_name, module, x, aux):
        activation = x

        def fwd_fn(fwd_activation):
            return fwd_activation

        def bwd_fn(bwd_activation):
            return bwd_activation

        return fwd_fn(activation), bwd_fn


mp_hook_fns = ActivationManipulationFunctions()

GATHER_RULES = {
    ColumnParallelLinear: mp_hook_fns.column_parallel_linear_gather,
    QuantizedColumnParallelLinear: mp_hook_fns.column_parallel_linear_gather,
    Dropout: mp_hook_fns.self_attn_dropout_gather,
    torch.nn.Linear: mp_hook_fns.output_projection_gather,
}
SHARD_RULES = {
    ColumnParallelLinear: mp_hook_fns.column_parallel_linear_scatter,
    QuantizedColumnParallelLinear: mp_hook_fns.column_parallel_linear_scatter,
    Dropout: mp_hook_fns.self_attn_dropout_scatter,
    torch.nn.Linear: mp_hook_fns.output_projection_scatter,
}
REARRANGE_RULES = {
    ColumnParallelLinear: mp_hook_fns.column_parallel_linear_rearrange,
    QuantizedColumnParallelLinear: mp_hook_fns.column_parallel_linear_rearrange,
    RowParallelLinear: mp_hook_fns.row_parallel_linear_rearrange,
    QuantizedRowParallelLinear: mp_hook_fns.row_parallel_linear_rearrange,
    ModelParallelTransformerDecoder: mp_hook_fns.model_parallel_decoder_rearrange,
    ModelParallelTransformerDecoderLayer: mp_hook_fns.model_parallel_decoder_layer_rearrange,
    FusedLayerNorm: mp_hook_fns.fused_layer_norm_rearrange,
    ModelParallelMultiheadAttention: mp_hook_fns.model_parallel_multihead_attn_rearrange,
    Dropout: mp_hook_fns.self_attn_dropout_rearrange,
    VocabParallelEmbedding: mp_hook_fns.vocab_parallel_and_out_proj_rearrange,
    torch.nn.Linear: mp_hook_fns.vocab_parallel_and_out_proj_rearrange,
}


class ShardedActivation:
    def __init__(
        self,
        registered_name: str,
        module: nn.Module,
        aux: Tuple,
        sharded_activations: Union[List[Tensor], Tuple[Tensor]],
    ) -> None:
        self.registered_name = registered_name
        self.module = module
        self.module_type = type(module)
        self.aux = aux
        self.activations = sharded_activations

        self.is_gathered = False
        self.is_rearranged = False

        if (
            self.module_type in GATHER_RULES
            and self.module_type in SHARD_RULES
            and self.module_type in REARRANGE_RULES
        ):
            self.gather_fn = GATHER_RULES[self.module]
            self.scatter_fn = SHARD_RULES[self.module]
            self.rearrange_fn = REARRANGE_RULES[self.module]
            self.undo_rearrange_fn = None
        elif (
            self.module_type in GATHER_RULES
            or self.module_type in SHARD_RULES
            or self.module_type in REARRANGE_RULES
        ):
            raise Exception(f"Module: {self.registered_name} is missing "
                            f"either shard or gather rules for hooks")
        else:
            raise Exception(f"Module: {registered_name} has no shard or "
                            f"gather rules for hooks")

    def gather(self):
        assert not self.is_gathered, (f"Module: {self.registered_name} "
                                      f"activation is already gathered!")

        self.activations = self.gather_fn(self.activations)
        self.is_gathered = True

    def rearrange(self):
        assert self.is_gathered
        assert not self.is_rearranged
        self.activations, self.undo_rearrange_fn = self.rearrange_fn(
            self.activations)
        self.is_rearranged = True

    def _verify_editing_fn(self, editing_fn):
        # TODO: Verify that editing function is valid
        #   - Time for breakpoints and runtime
        #   - Input shape == output shape
        return True

    def edit_activation(self, editing_fn):
        assert self._verify_editing_fn(editing_fn), ("Editing function did "
                                                     "not pass verification.")
        self.activations = editing_fn(self.activations)

    def undo_rearrange(self):
        assert self.is_gathered
        assert self.is_rearranged
        self.activations = self.undo_rearrange_fn(self.activations)
        self.is_rearranged = False

    def scatter(self):
        assert self.is_gathered, (f"Module: {self.registered_name} activation "
                                  f"is already sharded!")
        assert not self.is_rearranged
        self.activations = self.scatter_fn(self.activations)
        self.is_gathered = False
