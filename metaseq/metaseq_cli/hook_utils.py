import torch

from megatron.mpu.mappings import gather_from_tensor_model_parallel_region

from megatron.mpu import ColumnParallelLinear, RowParallelLinear

from metaseq.model_parallel.modules.transformer_layer import (
    ModelParallelTransformerDecoderLayer,
)

from metaseq.model_parallel.modules.multihead_attention import (
    ModelParallelMultiheadAttention,
)

from metaseq.model_parallel.models.transformer import (
    ModelParallelTransformerDecoder,
)
from contextlib import contextmanager
from functools import partial


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


def get_activation_capture_hook_dict(model, desired_module_activations):
    activation_dict, hook_dict = {}, {}

    desired_module_activations = set(desired_module_activations)

    for n, m in model.named_modules():
        if n in desired_module_activations:
            hook_dict[n] = partial(forward_hook_fn, n, activation_dict)

    return hook_dict, activation_dict


def forward_hook_fn(registered_name, save_dict, m, _, outputs):
    # NOTE: don't touch the inputs! THEY MIGHT BE WRONG
    # the inputs to the row parallel layers
    # are sharded, so we need to unshard them

    # only save it on rank 0 but i think we need to call the gather in all
    # processes so this hook needs to be installed on every process
    type_m = type(m)

    # too scared to do isinstance checks
    if type_m == ColumnParallelLinear:

        if not m.gather_output:
            output = gather_from_tensor_model_parallel_region(outputs[0])

        if m.skip_bias_add:
            output = output + outputs[1]

    elif type_m == RowParallelLinear:
        output = outputs[0]

        if m.skip_bias_add:
            output = output + outputs[1]

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

    # best effort kind of thing
    else:
        # works on VocabParallelEmbedding
        output = outputs

    if torch.distributed.get_rank() == 0:
        # only pytorch tensors allowed for now
        save_dict[registered_name] = output.detach().cpu()
