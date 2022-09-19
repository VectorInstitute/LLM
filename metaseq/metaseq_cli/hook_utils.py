from contextlib import contextmanager
from functools import partial
import logging

import torch
from einops import rearrange

from megatron.mpu.mappings import gather_from_tensor_model_parallel_region
from megatron.mpu import ColumnParallelLinear, RowParallelLinear
from metaseq.modules.layer_norm import FusedLayerNorm
from metaseq.modules.dropout import Dropout
from metaseq.distributed import utils as distributed_utils
from metaseq.model_parallel.modules.transformer_layer import (
    ModelParallelTransformerDecoderLayer,
)
from metaseq.model_parallel.modules.multihead_attention import (
    ModelParallelMultiheadAttention,
)
from metaseq.model_parallel.models.transformer import (
    ModelParallelTransformerDecoder,
)



logger = logging.getLogger(__name__)


@contextmanager
def apply_forward_hook(model, hook_dict):
    """hook dict should be names of modules keyed by functions all hooks must
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


def get_activation_capture_hook_dict(model, desired_module_activations, aux=None):
    activation_dict, hook_dict = {}, {}

    desired_module_activations = set(desired_module_activations)

    #logger.info("Rank {} with aux: {}".format(torch.distributed.get_rank(), aux))

    for n, m in model.named_modules():
        if n in desired_module_activations:
            hook_dict[n] = partial(forward_hook_fn, n, activation_dict, aux)

    return hook_dict, activation_dict


def forward_hook_fn(registered_name, save_dict, aux, m, _, outputs):
    # NOTE: don't touch the inputs! THEY MIGHT BE WRONG
    # the inputs to the row parallel layers
    # are sharded, so we need to unshard them
    type_m = type(m)

    # every rank needs to do this
    if type_m == ColumnParallelLinear:
        if not m.gather_output:
            outputs = (
                gather_from_tensor_model_parallel_region(outputs[0]),
                *outputs[1:],
            )
    elif type_m == Dropout:
        if "self_attn" in registered_name:
            # TODO: Newest megatron supports both first and last dim
            # megatron only gathers the last dim, but we want to
            # gather on the first dim so we permute it to the end
            # and then permute it back

            # TODO: we need to clean this up, it depends on the fact
            # that the previous layer here is sharded
            #outputs = rearrange(outputs, "bk s1 s2 -> s1 s2 bk")
            if not aux:
                logger.info(
                    ("Rank {}: Required auxillary data for self-attention maps"
                    "is not present").format(torch.distributed.get_rank()))

            outputs = gather_from_tensor_model_parallel_region(
                rearrange(outputs, "(b k) s1 s2 -> s1 s2 b k", b=aux[0]))

    elif type_m == torch.nn.Linear:
        # TODO: This works, need to test against HF
        outputs = gather_from_tensor_model_parallel_region(outputs)

    # only rank 0 needs to do the rest
    if torch.distributed.get_rank() != 0:
        return

    # too scared to do isinstance checks
    if type_m == ColumnParallelLinear:

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

    elif type_m == RowParallelLinear:

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

    elif type_m == torch.nn.Linear:
        # Case for decoder.output_projection
        # TODO: Unhandled, gather_from_tensor_model_parallel_region(outputs) hangs
        #       Request decoder.output_projection to test
        output = outputs

    # best effort kind of thing
    else:
        # works on VocabParallelEmbedding
        output = outputs

    # some layers are always in S x B x D, permute them back
    if type_m in (
        ModelParallelTransformerDecoderLayer,
        FusedLayerNorm,
        ModelParallelMultiheadAttention,
    ):
        output = rearrange(output, "s b d -> b s d")

    # only pytorch tensors allowed for now
    save_dict[registered_name] = output.detach().cpu()
