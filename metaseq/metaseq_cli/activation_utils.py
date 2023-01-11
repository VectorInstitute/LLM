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


# TODO: Remove before PR
def breakpoint_rank0():
    if torch.distributed.get_rank() == 0:
        breakpoint()


class ActivationManipulationFunctions:
    # Gather functions
    @staticmethod
    def column_parallel_linear_gather(registered_name, module, x, aux):
        if not module.gather_output:
            output = (gather_from_tensor_model_parallel_region(x[0]), x[1:])
        return output

    @staticmethod
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

    @staticmethod
    def output_projection_gather(registered_name, module, x, aux):
        output = gather_from_tensor_model_parallel_region(x)
        return output

    # Shard functions
    @staticmethod
    def column_parallel_linear_scatter(registered_name, module, x, aux):
        if not module.gather_output:
            output = (scatter_to_tensor_model_parallel_region(x[0]), x[1:])
        return output

    @staticmethod
    def self_attn_dropout_scatter(registered_name, module, x, aux):
        if "self_attn" in registered_name:
            output = scatter_to_tensor_model_parallel_region(x)
            output = rearrange(output, "s1 s2 b k -> (b k) s1 s2")
        else:
            output = x

        return output

    @staticmethod
    def output_projection_scatter(registered_name, module, x, aux):
        output = scatter_to_tensor_model_parallel_region(x)
        return output

    # Rearrange functions
    @staticmethod
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
            else:
                fwd_output = fwd_activation

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
            else:
                bwd_output = bwd_activation

            if module.skip_bias_add:
                bwd_output = bwd_output - bias

            return bwd_output

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def row_parallel_linear_rearrange(registered_name, module, x, aux):
        layer_type = registered_name.split(".")[-1]
        activation = x[0]
        bias = x[1]

        def fwd_fn(fwd_activation):
            # Manually add bias if needed
            if module.skip_bias_add:
                fwd_output = fwd_activation + bias
            else:
                fwd_output = fwd_activation

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
            else:
                bwd_output = bwd_activation

            if module.skip_bias_add:
                bwd_output = bwd_output - bias

            return bwd_output

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def model_parallel_decoder_rearrange(registered_name, module, x, aux):
        activation = x[0]
        auxillary_output = x[1:]

        def fwd_fn(fwd_activation):
            return fwd_activation

        def bwd_fn(bwd_activation):
            # Preserve auxillary data for downstream use
            return (bwd_activation, *auxillary_output)

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def model_parallel_decoder_layer_rearrange(registered_name, module, x, aux):
        activation = x[0]
        auxillary_output = x[1:]

        def fwd_fn(fwd_activation):
            fwd_output = rearrange(fwd_activation, "s b d -> b s d")
            return fwd_output

        def bwd_fn(bwd_activation):
            bwd_output = rearrange(bwd_activation, "b s d -> s b d")
            return (bwd_output, *auxillary_output)

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def fused_layer_norm_rearrange(registered_name, module, x, aux):
        activation = x

        def fwd_fn(fwd_activation):
            fwd_output = rearrange(fwd_activation, "s b d -> b s d")
            return fwd_output

        def bwd_fn(bwd_activation):
            bwd_output = rearrange(bwd_activation, "b s d -> s b d")
            return bwd_output

        return fwd_fn(activation), bwd_fn

    @staticmethod
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

            return ((bwd_output, bias), attn_weights)

        return fwd_fn(activation), bwd_fn

    @staticmethod
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

    @staticmethod
    def vocab_parallel_and_out_proj_rearrange(registered_name, module, x, aux):
        # This function is also just unity
        activation = x

        def fwd_fn(fwd_activation):
            return fwd_activation

        def bwd_fn(bwd_activation):
            return bwd_activation

        return fwd_fn(activation), bwd_fn


class LayerRules:
    def __init__(self):
        self.mp_hook_fns = ActivationManipulationFunctions()

        # Imported + uninitialized layers and type()-checked instantiated
        # layers hash the same
        self.gather_rules = {
            ColumnParallelLinear: self.mp_hook_fns.column_parallel_linear_gather,
            QuantizedColumnParallelLinear: self.mp_hook_fns.column_parallel_linear_gather,
            Dropout: self.mp_hook_fns.self_attn_dropout_gather,
            torch.nn.Linear: self.mp_hook_fns.output_projection_gather,
        }
        self.scatter_rules = {
            ColumnParallelLinear: self.mp_hook_fns.column_parallel_linear_scatter,
            QuantizedColumnParallelLinear: self.mp_hook_fns.column_parallel_linear_scatter,
            Dropout: self.mp_hook_fns.self_attn_dropout_scatter,
            torch.nn.Linear: self.mp_hook_fns.output_projection_scatter,
        }
        self.rearrange_rules = {
            ColumnParallelLinear: self.mp_hook_fns.column_parallel_linear_rearrange,
            QuantizedColumnParallelLinear: self.mp_hook_fns.column_parallel_linear_rearrange,
            RowParallelLinear: self.mp_hook_fns.row_parallel_linear_rearrange,
            QuantizedRowParallelLinear: self.mp_hook_fns.row_parallel_linear_rearrange,
            ModelParallelTransformerDecoder: self.mp_hook_fns.model_parallel_decoder_rearrange,
            ModelParallelTransformerDecoderLayer: self.mp_hook_fns.model_parallel_decoder_layer_rearrange,
            FusedLayerNorm: self.mp_hook_fns.fused_layer_norm_rearrange,
            ModelParallelMultiheadAttention: self.mp_hook_fns.model_parallel_multihead_attn_rearrange,
            Dropout: self.mp_hook_fns.self_attn_dropout_rearrange,
            VocabParallelEmbedding: self.mp_hook_fns.vocab_parallel_and_out_proj_rearrange,
            torch.nn.Linear: self.mp_hook_fns.vocab_parallel_and_out_proj_rearrange,
        }

    def get_gather_rule(self, module_type):
        if module_type in self.gather_rules:
            return self.gather_rules[module_type]
        else:
            return None

    def get_scatter_rule(self, module_type):
        if module_type in self.scatter_rules:
            return self.scatter_rules[module_type]
        else:
            return None

    def get_rearrange_rule(self, module_type):
        if module_type in self.rearrange_rules:
            return self.rearrange_rules[module_type]
        else:
            return None


LAYER_RULES = LayerRules()


def unity_gather(registered_name, module, x, aux):
    """
    Layers with no defined gather function use this. Simply returns the
    activation.
    """
    return x


def unity_scatter(registered_name, module, x, aux):
    """
    Layers with no defined scatter function use this. Simply returns the
    activation.
    """
    return x


def unity_rearrange(registered_name, module, x, aux):
    """
    Layers with no defined rearrange function use this. If the output is
    a tuple with more than one element, then we assume the first element is
    always the activation, and the rest is auxillary data.
    If the output is just a single element, then we don't prune any auxillary
    data.
    """
    if isinstance(x, tuple) and len(x) > 1:
        activation = x[0]
        auxillary_output = x[1:]
    else:
        activation = x
        auxillary_output = None

    def fwd_fn(fwd_activation):
        return fwd_activation

    def bwd_fn(bwd_activation):
        # Preserve auxillary data for downstream use
        if auxillary_output is None:
            return bwd_activation
        else:
            return (bwd_activation, *auxillary_output)

    return fwd_fn(activation), bwd_fn


class ShardedActivation:
    def __init__(
        self,
        registered_name: str,
        module: nn.Module,
        aux: Tuple,
        layer_outputs: Union[List[Tensor], Tuple[Tensor]],
    ) -> None:
        # TODO: Support self.activations = Tuple[activations]
        """
        layer_outputs is some tuple of arbitrary length carrying anything. This
        gets converted via some rearrange function into one or more activations
        which are saved under self.activations
        
        layer_outputs -> gather_fn -> layer_outputs -> rearrange_fn -> activations
        -> edit_fn -> activations -> undo_rearrange_fn -> layer_outputs -> scatter_fn
        -> layer_outputs
        """
        self.registered_name = registered_name
        self.module = module
        self.module_type = type(module)
        self.aux = aux
        self.layer_outputs = layer_outputs
        self.activations = None

        self.is_gathered = False
        self.is_rearranged = False

        self.gather_fn = LAYER_RULES.get_gather_rule(self.module_type)
        self.scatter_fn = LAYER_RULES.get_scatter_rule(self.module_type)
        self.rearrange_fn = LAYER_RULES.get_rearrange_rule(self.module_type)
        self.undo_rearrange_fn = None

    # Gather and rearrange always happen in hook, so give them unity fallbacks
    def gather(self):
        """
        Gather functions take a tuple of (registered_name, module,
        layer_outputs, aux), and call gather on some subset of the
        layer_outputs. They may also do some preliminary rearranging if
        necessary. The gather functions return the gathered layer_outputs.
        """
        assert not self.is_gathered, (f"Module: {self.registered_name} "
                                      f"activation is already gathered!")

        gather_fn = unity_gather if self.gather_fn is None else self.gather_fn

        self.layer_outputs = gather_fn(
            self.registered_name,
            self.module,
            self.layer_outputs,
            self.aux,
        )
        self.is_gathered = True

    def rearrange(self):
        """
        Rearrange functions take a tuple of (registered_name, module,
        layer_outputs, aux), prune out non-activation layer_outputs, rearrange
        the extracted activation, and return the activation. Also returns the
        inverse rearrange function to restore the original shape of the
        activation.

        Rearrange functions also must handle auxillary data in
        self.activations, ie. the layer output (tuple potentially). The
        auxillary data is different from layer to layer, so we
        must special case them along with their respective rearrange logic.
        """
        assert self.is_gathered
        assert not self.is_rearranged

        rearrange_fn = unity_rearrange if self.rearrange_fn is None else self.rearrange_fn

        self.activations, self.undo_rearrange_fn = rearrange_fn(
            self.registered_name,
            self.module,
            self.layer_outputs,
            self.aux,
        )
        self.is_rearranged = True

    def _verify_editing_fn(self, editing_fn):
        # TODO: Verify that editing function is valid
        #   - Time for breakpoints and runtime
        #   - Input shape == output shape
        assert callable(editing_fn), f"Editing fn: {editing_fn} is not callable"
        return True

    # Editing, undo rearrange, and scatter only happen if there is a valid
    # editing_fn provided. Give them unity fallbacks only if gather/rearrange
    # are also None, else raise exception
    def edit_activation(self, editing_fn):
        assert self._verify_editing_fn(editing_fn), ("Editing function did "
                                                     "not pass verification.")
        self.activations = editing_fn(self.activations)

    def undo_rearrange(self):
        """
        Undo rearrange functions are defined within the rearrange functions,
        and are one of their return values. They are simply the inverse of the
        respective rearrange function. The undo rearrange functions take an
        activation as input, undo the rearrange, and return the restored
        activation.
        """
        assert self.is_gathered
        assert self.is_rearranged

        if self.undo_rearrange_fn is None and self.rearrange_fn is None:
            self.undo_rearrange_fn = unity_undo_rearrange
        elif self.undo_rearrange_fn is None and self.rearrange_fn is not None:
            raise Exception(f"Module: {self.registered_name} has unpaired "
                            f"rearrange fn")
        else:
            undo_rearrange_fn = self.undo_rearrange_fn

        self.layer_outputs = undo_rearrange_fn(self.activations)
        self.is_rearranged = False

    def scatter(self):
        """
        Scatter functions are just the inverse of the defined gather functions.
        They take some layer outputs and scatter them, returning the scattered
        layer outputs.
        """
        assert self.is_gathered, (f"Module: {self.registered_name} activation "
                                  f"is already sharded!")
        assert not self.is_rearranged

        if self.scatter_fn is None and self.gather_fn is None:
            scatter_fn = unity_scatter
        elif self.scatter_fn is None and self.gather_fn is not None:
            raise Exception(f"Module: {self.registered_name} has unpaired "
                            f"gather fn")
        else:
            scatter_fn = self.scatter_fn

        self.layer_outputs = scatter_fn(
            self.registered_name,
            self.module,
            self.layer_outputs,
            self.aux,
        )
        self.is_gathered = False
