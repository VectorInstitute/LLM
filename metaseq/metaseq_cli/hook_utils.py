import codecs
from contextlib import contextmanager
from functools import partial
import logging
from typing import Optional

import cloudpickle
from einops import rearrange
import torch
from torch import Tensor

from metaseq_cli import activation_utils
from megatron.mpu.mappings import gather_from_tensor_model_parallel_region
from megatron.mpu.utils import split_tensor_along_last_dim
from megatron.mpu.layers import ColumnParallelLinear, RowParallelLinear
from megatron.mpu.initialize import get_tensor_model_parallel_group
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
    OPTDecoder,
    OPTDecoderLayer,
    OPTAttention,
    OPTLearnedPositionalEmbedding,
)
from metaseq.quantization import (
    QuantizedColumnParallelLinear,
    QuantizedRowParallelLinear,
)


logger = logging.getLogger(__name__)


def decode_str(obj_in_str):
    return cloudpickle.loads(
        codecs.decode(obj_in_str.encode("utf-8"), "base64")
    )


@contextmanager
def apply_forward_hook(model, hook_dict):
    """
    Hook dict should be names of modules keyed by functions all hooks must
    have the actual signature as the register forward hook in pytorch
    """
    all_hooks = []

    for n, m in model.named_modules():
        if n in hook_dict:
            all_hooks.append(m.register_forward_hook(hook_dict[n]))

    try:
        yield

    finally:
        for h in all_hooks:
            h.remove()

        all_hooks.clear()


def get_activation_capture_hook_dict(
    model,
    encoded_activation_payload: activation_utils.ActivationPayload,
    aux=None,
    model_type="opt"
):
    """
    Attach the specified hook forward-pass hook functions onto the given
    model. The model types are one of [opt, hf]
    """
    activation_dict, hook_dict = {}, {}

    # Skip decoding if payload is fed directly (case of local HF models)
    if isinstance(
        encoded_activation_payload,
        activation_utils.ActivationPayload,
    ):
        activation_payload = encoded_activation_payload
    else:
        activation_payload = decode_str(encoded_activation_payload)

    module_names_activation_retrieval = set(
        activation_payload.module_names_activation_retrieval
    )
    module_editing_fn_pairs = activation_payload.module_editing_fn_pairs

    for n, m in model.named_modules():
        editing_fn = module_editing_fn_pairs.get(n, None)

        if n in module_names_activation_retrieval:
            if model_type == "opt":
                hook_dict[n] = partial(
                    generic_forward_hook_fn,
                    n,
                    activation_dict,
                    aux,
                    editing_fn,
                )

            elif model_type == "hf":
                hook_dict[n] = partial(
                    hf_forward_hook_fn,
                    n,
                    activation_dict,
                    aux,
                    editing_fn,
                )

    return hook_dict, activation_dict


def generic_forward_hook_fn(
    registered_name,
    save_dict,
    aux,
    editing_fn,
    self,
    _inputs,
    outputs,
) -> Optional[Tensor]:
    """
    Generic forward hook function that can be used for activation retrieval,
    editing, etc.
    """
    activation = activation_utils.ShardedActivation(
        registered_name=registered_name,
        module=self,
        aux=aux,
        layer_outputs=outputs,
    )

    # Gather the full activation using all ranks
    activation.gather()

    # Only rank0 handles the full activation
    if torch.distributed.get_rank() == 0:
        # Rearrange to correct shape
        activation.rearrange()

        # Run the editing function
        if editing_fn is not None:
            activation.edit_activation(editing_fn)

        # Bind a copy of output to save dict
        save_dict[registered_name] = activation.activations.detach().cpu()

        # Undo the rearrange to perfectly reconstruct original full activation
        # post-gather
        activation.undo_rearrange()

    # Shed non-rank0 workers if they aren't neede
    else:
        if editing_fn is None:
            return

    # If the activation was edited we need to return it. Scatter the edited
    # full activation to all ranks, then return the sharded activation
    if editing_fn is not None:
        # Scatter the output to each rank
        activation.scatter()

        return activation.layer_outputs


def opt_forward_hook_fn(
    registered_name,
    save_dict,
    aux,
    editing_fn,
    m,
    _,
    outputs,
):
    if torch.distributed.get_rank() == 0:
        if editing_fn is None:
            logger.info("Editing function is None")
        else:
            logger.info(f"Editing module {registered_name}!")
            # TODO: Debugging
            logger.info(f"Function is: {editing_fn}")

    # NOTE: Don't consider inputs, since there can be arbitrary code between
    #       module calls
    type_m = type(m)

    # every rank needs to do this
    if type_m == ColumnParallelLinear or type_m == QuantizedColumnParallelLinear:
        if not m.gather_output:
            outputs = (
                gather_from_tensor_model_parallel_region(outputs[0]),
                *outputs[1:],
            )
    elif type_m == Dropout:
        if "self_attn" in registered_name:
            # NOTE: Newest megatron supports both first and last dim
            # megatron only gathers the last dim, but we want to
            # gather on the first dim so we permute it to the end
            # and then permute it back
            if not aux:
                logger.info(
                    ("Rank {}: Required auxillary data for self-attention maps"
                    "is not present").format(torch.distributed.get_rank()))

            outputs = gather_from_tensor_model_parallel_region(
                rearrange(outputs, "(b k) s1 s2 -> s1 s2 b k", b=aux[0]))

    elif type_m == torch.nn.Linear:
        outputs = gather_from_tensor_model_parallel_region(outputs)

    # only rank 0 needs to do the rest
    if torch.distributed.get_rank() != 0:
        return

    # too scared to do isinstance checks
    if type_m == ColumnParallelLinear or type_m == QuantizedColumnParallelLinear:

        output = outputs[0]

        if m.skip_bias_add:
            output = output + outputs[1]

        layer_type = registered_name.split(".")[-1]

        # not always S x B x D, it's only when it's used in qkv_proj
        # NOTE: OPT's qkv_proj combined projection is hard to compare with HF
        if layer_type == "qkv_proj":
            output = rearrange(output, "s b d -> b s d")

        elif layer_type in ["q_proj", "k_proj", "v_proj"]:
            output = rearrange(output, "s b d -> b s d")

        elif "fc" in layer_type:
            output = rearrange(output, "(s b) d -> b s d", b=aux[0])

    elif type_m == RowParallelLinear or type_m == QuantizedRowParallelLinear:

        output = outputs[0]

        if m.skip_bias_add:
            output = output + outputs[1]

        layer_type = registered_name.split(".")[-1]

        # not always S x B x D, it's only when it's used in outproj
        if layer_type == "out_proj":
            output = rearrange(output, "s b d -> b s d")

        elif "fc" in layer_type:
            output = rearrange(output, "(s b) d -> b s d", b=aux[0])

    elif type_m in (
        ModelParallelTransformerDecoder,
        ModelParallelTransformerDecoderLayer,
    ):
        # the rest are aux info
        output = outputs[0]

    elif type_m == ModelParallelMultiheadAttention:
        # the other param is just None
        output, attn_bias = outputs[0]

        if attn_bias is not None:
            output = output + attn_bias

    elif type_m == Dropout:
        output = outputs

        if "self_attn" not in registered_name:
            output = rearrange(output, "s b d -> b s d")

        else:
            output = rearrange(output, "s1 s2 b k -> b k s1 s2")

    else:
        # VocabParallelEmbedding and final output_projection case
        output = outputs

    # some layers are always in S x B x D, permute them back
    if type_m in (
        ModelParallelTransformerDecoderLayer,
        FusedLayerNorm,
        ModelParallelMultiheadAttention,
    ):
        output = rearrange(output, "s b d -> b s d")

    # Bind the final activation to the save dict
    save_dict[registered_name] = output.detach().cpu()


def hf_forward_hook_fn(
    registered_name,
    save_dict,
    aux,
    editing_fn,
    m,
    _,
    outputs
):
    """
    Forward hook function for HuggingFace OPT model, conditioning on
    (sub)module types. This function should not be used outside of testing.
    """
    # TODO: Bring this under the generic hook
    type_m = type(m)

    # In the case of duplicate types
    layer_type = registered_name.split(".")[-1]

    if type_m == torch.nn.Embedding or type_m == OPTLearnedPositionalEmbedding:
        output = outputs

    elif type_m == OPTAttention:
        # (attn_out, attn_weights_reshaped, past_key_values)
        output = outputs[0]

    elif type_m == torch.nn.LayerNorm:
        # Having "layers" in the name means m is a per-module layer norm
        if layer_type == "final_layer_norm" and "layers" in registered_name:
            output = rearrange(outputs, "(b s) h -> b s h", b=aux[0])

        else:
            output = outputs

    elif type_m == torch.nn.Linear:
        if layer_type in ["q_proj", "k_proj", "v_proj"]:
            output = outputs

        else:
            output = rearrange(outputs, "(b s) h -> b s h", b=aux[0])
    else:
        raise NotImplementedError

    if editing_fn is not None:
        output = editing_fn(output)

    save_dict[registered_name] = output.detach().cpu()

    # Prefer not to return if we don't have to
    if editing_fn is not None:
        if type_m == torch.nn.Embedding or type_m == OPTLearnedPositionalEmbedding:
            retval = output

        elif type_m == OPTAttention:
            retval = (output, *outputs[1:])

        elif type_m == torch.nn.LayerNorm:
            # Having "layers" in the name means m is a per-module layer norm
            if layer_type == "final_layer_norm" and "layers" in registered_name:
                retval = rearrange(output, "b s h -> (b s) h")

            else:
                retval = outputs

        elif type_m == torch.nn.Linear:
            if layer_type in ["q_proj", "k_proj", "v_proj"]:
                retval = output

            else:
                retval = rearrange(output, "b s h -> (b s) h")

        else:
            raise NotImplementedError

        return retval
