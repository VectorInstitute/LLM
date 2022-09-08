import torch

from megatron.mpu.mappings import gather_from_tensor_model_parallel_region

from megatron.mpu import ColumnParallelLinear, RowParallelLinear
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

    # TODO: add the hooks one my one class by class
    if torch.distributed.get_rank() == 0:
        breakpoint()

    output = outputs

    if isinstance(m, ColumnParallelLinear):
        if not m.gather_output:
            output = gather_from_tensor_model_parallel_region(outputs[0])
            outputs = (output, *outputs[1:])

    if hasattr(m, "skip_bias_add"):
        # if skip bias add, add it back in
        output = outputs[0] + outputs[1] if m.skip_bias_add else outputs[0]

    if torch.distributed.get_rank() == 0:

        if isinstance(output, (list, tuple)):
            assert len(output) == 2 and output[1] is None
            # bias
            output = output[0][0] + output[0][1]

        # only save it on rank 0
        save_dict[registered_name] = output.detach().cpu()
