import codecs
from contextlib import contextmanager
from functools import partial
import logging
from typing import Optional, Any, Dict, Callable, Union

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


def decode_str(obj_in_str: str) -> Any:
    """
    Given some object that has been serialized then encoded into a string,
    decode it and unserialize it back into the original object.
    """
    return cloudpickle.loads(
        codecs.decode(obj_in_str.encode("utf-8"), "base64")
    )


@contextmanager
def apply_forward_hook(
    model: torch.nn.Module,
    hook_dict: Dict[str, Callable],
) -> None:
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
    model: torch.nn.Module,
    encoded_activation_payload: Union[activation_utils.ActivationPayload, str],
    aux: Optional[tuple] = None,
    model_type: str = "opt",    # TODO: deprecate this arg
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
                    registered_name=n,
                    save_dict=activation_dict,
                    aux=aux,
                    editing_fn=editing_fn,
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
    registered_name: str,
    save_dict: Dict[str, Tensor],
    editing_fn: Callable,
    self: torch.nn.Module,
    _inputs: Any,
    outputs: Any,
    aux: Optional[tuple] = None,
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


def hf_forward_hook_fn(
    registered_name: str,
    save_dict: Dict[str, Tensor],
    editing_fn: Callable,
    self: torch.nn.Module,
    _inputs: Any,
    outputs: Any,
    aux: Optional[tuple] = None,
) -> Optional[Tensor]:

    """
    Forward hook function for HuggingFace OPT model, conditioning on
    (sub)module types. This function should not be used outside of testing.
    """
    # TODO: Bring this under the generic hook
    type_m = type(self)

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
