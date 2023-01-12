from typing import Tuple, List, Union, Any

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
from metaseq.modules.learned_positional_embedding import (
    LearnedPositionalEmbedding,
)
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


_LayerOutput = tuple[Any]
_Activation = Union[Tensor, tuple[Tensor]]


# TODO: Remove before PR
def breakpoint_rank0():
    if torch.distributed.get_rank() == 0:
        breakpoint()


class GatherFunctions:
    @staticmethod
    def unity_gather(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _LayerOutput:
        """
        Layers with no defined gather function use this. Simply returns the
        activation.
        """
        return layer_output

    @staticmethod
    def column_parallel_linear_gather(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _LayerOutput:
        if not module.gather_output:
            output = (
                gather_from_tensor_model_parallel_region(layer_output[0]),
                layer_output[1:]
            )
        return output

    @staticmethod
    def self_attn_dropout_gather(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _LayerOutput:
        if "self_attn" in registered_name:
            # NOTE: Newest megatron supports both first and last dim
            # megatron only gathers the last dim, but we want to
            # gather on the first dim so we permute it to the end
            # and then permute it back
            assert aux is not None, (f"Rank {torch.distributed.get_rank()}: "
                                     f"Required auxillary data for "
                                     f"self-attention maps is not present")

            output = gather_from_tensor_model_parallel_region(
                rearrange(layer_output, "(b k) s1 s2 -> s1 s2 b k", b=aux[0]))
        else:
            output = layer_output

        return output

    @staticmethod
    def output_projection_gather(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _LayerOutput:
        output = gather_from_tensor_model_parallel_region(layer_output)
        return output


class ScatterFunctions:
    @staticmethod
    def unity_scatter(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _LayerOutput:
        """
        Layers with no defined scatter function use this. Simply returns the
        activation.
        """
        return layer_output

    @staticmethod
    def column_parallel_linear_scatter(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _LayerOutput:
        if not module.gather_output:
            output = (
                scatter_to_tensor_model_parallel_region(layer_output[0]),
                layer_output[1:]
            )
        return output

    @staticmethod
    def self_attn_dropout_scatter(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _LayerOutput:
        if "self_attn" in registered_name:
            output = scatter_to_tensor_model_parallel_region(layer_output)
            output = rearrange(output, "s1 s2 b k -> (b k) s1 s2")
        else:
            output = layer_output

        return output

    @staticmethod
    def output_projection_scatter(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _LayerOutput:
        output = scatter_to_tensor_model_parallel_region(layer_output)
        return output


class RearrangeFunctions:
    # TODO: In the future, it could be helpful to define an additional class
    #       called ExtractActivationFunctions. This would pull out the logic
    #       in the current rearrange functions which separates the
    #       activation(s) from any auxillary data that needs to be
    #       re-scattered.
    @staticmethod
    def unity_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
        """
        Layers with no defined rearrange function use this. If the output is
        a tuple with more than one element, then we assume the first element is
        always the activation, and the rest is auxillary data.
        If the output is just a single element, then we don't prune any
        auxillary data.
        """
        if isinstance(layer_output, tuple) and len(layer_output) > 1:
            activation = layer_output[0]
            auxillary_output = layer_output[1:]
        else:
            activation = layer_output
            auxillary_output = None

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            return fwd_activation

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
            # Preserve auxillary data for downstream use
            if auxillary_output is None:
                return bwd_activation
            else:
                return (bwd_activation, *auxillary_output)

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def column_parallel_linear_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
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
        activation = layer_output[0]
        bias = layer_output[1]

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            # Manually add bias if needed
            if module.skip_bias_add:
                fwd_output = fwd_activation + bias
            else:
                fwd_output = fwd_activation

            # not always S x B x D, it's only when it's used in qkv_proj
            # NOTE: OPT's qkv_proj combined projection is hard to compare with
            #       HF
            if layer_type in ["q_proj", "k_proj", "v_proj", "qkv_proj"]:
                fwd_output = rearrange(fwd_output, "s b d -> b s d")

            elif "fc" in layer_type:
                fwd_output = rearrange(
                    fwd_output,
                    "(s b) d -> b s d",
                    b=aux[0]
                )

            # Need to return output and bias for this to be reverisble
            return fwd_output

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
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
    def row_parallel_linear_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
        layer_type = registered_name.split(".")[-1]
        activation = layer_output[0]
        bias = layer_output[1]

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            # Manually add bias if needed
            if module.skip_bias_add:
                fwd_output = fwd_activation + bias
            else:
                fwd_output = fwd_activation

            # not always S x B x D, it's only when it's used in out_proj
            if layer_type == "out_proj":
                fwd_output = rearrange(fwd_output, "s b d -> b s d")

            elif "fc" in layer_type:
                fwd_output = rearrange(
                    fwd_output, "(s b) d -> b s d", b=aux[0])

            return fwd_output

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
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
    def model_parallel_decoder_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
        activation = layer_output[0]
        auxillary_output = layer_output[1:]

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            return fwd_activation

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
            # Preserve auxillary data for downstream use
            return (bwd_activation, *auxillary_output)

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def model_parallel_decoder_layer_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
        activation = layer_output[0]
        auxillary_output = layer_output[1:]

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            fwd_output = rearrange(fwd_activation, "s b d -> b s d")
            return fwd_output

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
            bwd_output = rearrange(bwd_activation, "b s d -> s b d")
            return (bwd_output, *auxillary_output)

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def fused_layer_norm_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
        activation = layer_output

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            fwd_output = rearrange(fwd_activation, "s b d -> b s d")
            return fwd_output

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
            bwd_output = rearrange(bwd_activation, "b s d -> s b d")
            return bwd_output

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def model_parallel_multihead_attn_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
        activation, bias = layer_output[0]

        attn_weights = layer_output[1]

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            if bias is not None:
                fwd_output = fwd_activation + bias

            fwd_output = rearrange(fwd_output, "s b d -> b s d")

            return fwd_output

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
            bwd_output = rearrange(bwd_activation, "b s d -> s b d")

            if bias is not None:
                bwd_output = bwd_activation - bias

            return ((bwd_output, bias), attn_weights)

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def self_attn_dropout_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
        activation = layer_output

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            if "self_attn" in registered_name:
                fwd_output = rearrange(
                    fwd_activation, "s1 s2 b k -> b k s1 s2")

            else:
                fwd_output = rearrange(fwd_activation, "s b d -> b s d")

            return fwd_output

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
            if "self_attn" in registered_name:
                bwd_output = rearrange(
                    bwd_activation, "b k s1 s2 -> s1 s2 b k")

            else:
                bwd_output = rearrange(bwd_activation, "b s d -> s b d")

            return bwd_output

        return fwd_fn(activation), bwd_fn

    @staticmethod
    def vocab_parallel_and_out_proj_rearrange(
        registered_name: str,
        module: nn.Module,
        layer_output: _LayerOutput,
        aux: tuple[Any],
    ) -> _Activation:
        # This function is also just unity
        activation = layer_output

        def fwd_fn(fwd_activation: _Activation) -> _Activation:
            return fwd_activation

        def bwd_fn(bwd_activation: _Activation) -> _LayerOutput:
            return bwd_activation

        return fwd_fn(activation), bwd_fn


class LayerRules:
    def __init__(self):
        """
        A rule consists of a mapping (Layer, Gather/Scatter/Rearrange). This
        class sets rules for nn.Module layers statically. It also provides
        functions for retrieving these rules given a Layer, as well as validity
        checks for rules.
        """
        # Collection of all the layer we wish to define forward hook functions
        # for
        self.defined_layers = (
            ColumnParallelLinear,
            QuantizedColumnParallelLinear,
            Dropout,
            torch.nn.Linear,
            RowParallelLinear,
            QuantizedRowParallelLinear,
            ModelParallelTransformerDecoder,
            ModelParallelTransformerDecoderLayer,
            FusedLayerNorm,
            ModelParallelMultiheadAttention,
            VocabParallelEmbedding,
            LearnedPositionalEmbedding,
        )

        # These contain all the defined functions for each hook operation
        self.gather_fns = GatherFunctions()
        self.scatter_fns = ScatterFunctions()
        self.rearrange_fns = RearrangeFunctions()

        # Each function can be used by any number of layers
        self.gather_rules = {
            self.gather_fns.column_parallel_linear_gather: (
                ColumnParallelLinear,
                QuantizedColumnParallelLinear,
            ),
            self.gather_fns.self_attn_dropout_gather: (
                Dropout,
            ),
            self.gather_fns.output_projection_gather: (
                torch.nn.Linear,
            ),
            self.gather_fns.unity_gather: (
                RowParallelLinear,
                QuantizedRowParallelLinear,
                ModelParallelTransformerDecoder,
                ModelParallelTransformerDecoderLayer,
                FusedLayerNorm,
                ModelParallelMultiheadAttention,
                VocabParallelEmbedding,
                LearnedPositionalEmbedding,
            ),
        }
        self.scatter_rules = {
            self.scatter_fns.column_parallel_linear_scatter: (
                ColumnParallelLinear,
                QuantizedColumnParallelLinear,
            ),
            self.scatter_fns.self_attn_dropout_scatter: (
                Dropout,
            ),
            self.scatter_fns.output_projection_scatter: (
                torch.nn.Linear,
            ),
            self.scatter_fns.unity_scatter: (
                RowParallelLinear,
                QuantizedRowParallelLinear,
                ModelParallelTransformerDecoder,
                ModelParallelTransformerDecoderLayer,
                FusedLayerNorm,
                ModelParallelMultiheadAttention,
                VocabParallelEmbedding,
                LearnedPositionalEmbedding,
            )
        }
        self.rearrange_rules = {
            self.rearrange_fns.column_parallel_linear_rearrange: (
                ColumnParallelLinear,
                QuantizedColumnParallelLinear,
            ),
            self.rearrange_fns.row_parallel_linear_rearrange: (
                RowParallelLinear,
                QuantizedRowParallelLinear,
            ),
            self.rearrange_fns.model_parallel_decoder_rearrange: (
                ModelParallelTransformerDecoder,
            ),
            self.rearrange_fns.model_parallel_decoder_layer_rearrange: (
                ModelParallelTransformerDecoderLayer,
            ),
            self.rearrange_fns.fused_layer_norm_rearrange: (
                FusedLayerNorm,
            ),
            self.rearrange_fns.model_parallel_multihead_attn_rearrange: (
                ModelParallelMultiheadAttention,
            ),
            self.rearrange_fns.self_attn_dropout_rearrange: (
                Dropout,
            ),
            self.rearrange_fns.vocab_parallel_and_out_proj_rearrange: (
                VocabParallelEmbedding,
                torch.nn.Linear,
            ),
            self.rearrange_fns.unity_rearrange: (
                LearnedPositionalEmbedding,
            ),
        }

        # The previous rules are in a readable format, but for runtime the
        # inverted dictionary is much better to use
        def invert_dict(x):
            """
            The layer rule mappings will grow as we add more models, so we keep
            the mapping tree inverted for readability. This function produces
            the inverted mapping tree in format:
                Dict[LayerType: Function]
            Rather than the format:
                Dict[Function: Tuple[LayerType]]
            """
            intermediate = {v: k for k, v in x.items()}
            return {
                layer_type: fn
                for layer_tuple, fn in intermediate.items()
                for layer_type in layer_tuple
            }
        self.gather_rules = invert_dict(self.gather_rules)
        self.scatter_rules = invert_dict(self.scatter_rules)
        self.rearrange_rules = invert_dict(self.rearrange_rules)

        # Validate that every layer type we support is fully defined, ie. has
        # gather, scatter, and rearrange functions defined.
        for layer_type in self.defined_layers:
            assert (layer_type in self.gather_rules
                    and layer_type in self.scatter_rules
                    and layer_type in self.rearrange_rules)

    def get_gather_rule(self, module_type):
        if module_type in self.gather_rules:
            return self.gather_rules[module_type]
        else:
            raise Exception(f"Module: {module_type} missing gather rule")

    def get_scatter_rule(self, module_type):
        if module_type in self.scatter_rules:
            return self.scatter_rules[module_type]
        else:
            raise Exception(f"Module: {module_type} missing scatter rule")

    def get_rearrange_rule(self, module_type):
        if module_type in self.rearrange_rules:
            return self.rearrange_rules[module_type]
        else:
            raise Exception(f"Module: {module_type} missing rearrange rule")


LAYER_RULES = LayerRules()


class ShardedActivation:
    def __init__(
        self,
        registered_name: str,
        module: nn.Module,
        aux: tuple[Any],
        layer_outputs: _LayerOutput,
    ) -> None:
        # TODO: Support self.activations = Tuple[activations]
        #       - Hence the edit_fn should become edit_fns = Tuple[fns] of the
        #         same len as self.activations
        """
        Takes some layer outputs, and defines operations on them or the
        activations contained within.

        Logical order of function use:
            1. gather_fn(layer_outputs) -> layer_outputs
            2. rearrange_fn(layer_outputs) -> activations
            3. edit_fn(activations) -> activations
            4. undo_rearrange_fn(activations) -> layer_outputs
            5. scatter_fn(layer_outputs) -> layer_outputs
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

    def gather(self):
        """
        Gather functions take a tuple of (registered_name, module,
        layer_outputs, aux), and call gather on some subset of the
        layer_outputs. They may also do some preliminary rearranging if
        necessary. The gather functions return the gathered layer_outputs.
        """
        assert not self.is_gathered, (f"Module: {self.registered_name} "
                                      f"activation is already gathered!")

        self.layer_outputs = self.gather_fn(
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

        self.activations, self.undo_rearrange_fn = self.rearrange_fn(
            self.registered_name,
            self.module,
            self.layer_outputs,
            self.aux,
        )
        self.is_rearranged = True

    def _verify_editing_fn(self, editing_fn: callable):
        # TODO: Verify that editing function is valid
        #   - Time for breakpoints and runtime
        #   - Input shape == output shape
        assert callable(editing_fn), (f"Editing fn: {editing_fn} is not "
                                      f"callable")
        return True

    def edit_activation(self, editing_fn: callable):
        """
        Runs the activation editing function on the activation.
        """
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

        if self.undo_rearrange_fn is not None:
            undo_rearrange_fn = self.undo_rearrange_fn
        else:
            raise Exception(f"Module: {self.registered_name} should never "
                            f"have None rearrange fn")

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

        self.layer_outputs = self.scatter_fn(
            self.registered_name,
            self.module,
            self.layer_outputs,
            self.aux,
        )
        self.is_gathered = False
