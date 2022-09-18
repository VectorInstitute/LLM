#!/usr/bin/env python3
import argparse
import logging
from contextlib import contextmanager
from functools import partial

from transformers import OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTAttention
import torch

from opt_client import Client

logger = logging.getLogger(__name__)


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def get_hf_logits(client, hf_model, prompts):
    # need to prepend 2 for start of sequence when getting the input_ids
    input_ids_list = [[2] + p for p in client.tokenize(prompts)]
    max_len = max(map(len, input_ids_list))

    # pad right
    input_ids = torch.as_tensor([i + [1] * (max_len - len(i)) for i in input_ids_list])
    with torch.no_grad():
        # slice out the start of seq
        logits_hf = hf_model(input_ids)[0]

    logits_hf_list = [
        logits_hf[i, 1 : len(toks), :] for i, toks in enumerate(input_ids_list)
    ]
    return logits_hf_list


def get_hf_activations(client, hf_model, prompts):
    # need to prepend 2 for start of sequence when getting the input_ids
    input_ids_list = [[2] + p for p in client.tokenize(prompts)]
    max_len = max(map(len, input_ids_list))

    # pad right
    input_ids = torch.as_tensor([i + [1] * (max_len - len(i)) for i in input_ids_list])

    act = hf_model(
        input_ids,
        use_cache=False,
        output_hidden_states=True,
        output_attentions=True,
    )

    return act

@contextmanager
def apply_forward_hook_hf(model, hook_dict):
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


def get_activation_capture_hook_dict_hf(model, desired_module_activations, aux=None):
    activation_dict, hook_dict = {}, {}

    desired_module_activations = set(desired_module_activations)

    for n, m in model.named_modules():
        if n in desired_module_activations:
            hook_dict[n] = partial(forward_hook_fn, n, activation_dict, aux)

    return hook_dict, activation_dict


def forward_hook_fn(registered_name, save_dict, aux, m, _, outputs):
    type_m = type(m)

    if type_m == OPTAttention:
        output = outputs[0] # (attn_out, attn_weights_reshaped, past_key_values)

    save_dict[registered_name] = output.detach().cpu()


def init_opt_hf_mappings(num_layers):
    # mappings is a dict of equivalent layer types as keys, where the values
    # are 1. A list of (sub)module names for forward hooks, or 2. A list
    # containing "custom" as the first entry, and a function which formats the
    # collection of output activations
    opt_mappings = {
        "transformer_layers": [f"decoder.layers.{i}" for i in range(num_layers - 1)] + ["decoder.layer_norm"],
        "attention_maps": [f"decoder.layers.{i}.self_attn.dropout_module" for i in range(num_layers)],
        "self_attention": [f"decoder.layers.{i}.self_attn" for i in range(num_layers)],
        "logits": ["decoder"],
    }
    hf_mappings = {
        "transformer_layers": [
            "custom",
            get_hf_activations,
            lambda output: output["hidden_states"][1:]],
        "attention_maps": [
            "custom",
            get_hf_activations,
            lambda output: output["attentions"]],
        "self_attention": [f"model.decoder.layers.{i}.self_attn" for i in range(num_layers)],
        "logits": [
            "custom",
            get_hf_logits,
            lambda output: [output]],
    }
    return opt_mappings, hf_mappings


def retrieve_opt_activations(mapping, client, prompts):
    if "custom" in mapping:
        raise NotImplementedError
    else:
        module_names = mapping

    acts = client.get_activations(prompts, module_names)

    def _format_results(activations_batched):
        result = {n: [] for n in module_names}
        for acts in activations_batched:
            for name in module_names:
                result[name].append(acts[name])

        # Can't torch.stack along new batch axis since prompt length is dynamic
        return result

    acts = _format_results(acts)
    return acts


def retrieve_hf_activations(mapping, client, model, prompts):
    # Helper functions for hooked activations
    def _retrieve_hooked_acts(client, model, prompts, module_names):
        hook_dict, acts = get_activation_capture_hook_dict_hf(
            model,
            module_names,
            aux=None,
        )
        with apply_forward_hook_hf(model, hook_dict):
            results = get_hf_logits(client, model, prompts)

        return acts

    def _format_hooked_acts(hf_results, module_names):
        results = []
        for name in module_names:
            results.append(hf_results[name])

        return results

    # Use provided retrieval and formatting fns
    if "custom" in mapping:
        module_names = None
        retrieval_fn = mapping[1]
        format_fn = mapping[2]

    # Use forward hooks with formatter
    else:
        module_names = mapping
        retrieval_fn = _retrieve_hooked_acts
        format_fn = _format_hooked_acts

    if not module_names:
        # HF has built-in retrieval
        acts = retrieval_fn(client, model, prompts)
        acts = format_fn(acts)

    elif module_names:
        # Need to hook onto HF model for retrieval
        acts = retrieval_fn(client, model, prompts, module_names)
        acts = format_fn(acts, module_names)

    else:
        raise Exception("No valid configuration for HF activation retrieval")

    return acts


def assert_activations_correctness(hf_results, opt_results, act_type="transformer_layer"):
    """
    Helper function taking HF and OPT activation collections, and
    makes sure they're allclose
    """
    def _assert_allclose(hf_acts, opt_acts):
        assert torch.allclose(
            hf_acts.cpu().float(),
            opt_acts.cpu().float(),
            atol=1e-1,
        ), "Difference in hf and opt logits detected"

    for (module_name, opt_acts), hf_acts in zip(opt_results.items(), hf_results):
        for opt_act, hf_act in zip(opt_acts, hf_acts):

            opt_act = opt_act.detach().cpu().float()
            hf_act = hf_act.detach().cpu().float()

            if act_type == "transformer_layers" or act_type == "self_attention":
                bound = slice(1, opt_act.shape[-2] + 1)  # Trim start token and padding
                hf_act = hf_act[bound]

            elif act_type == "attention_maps":
                bound = slice(1, opt_act.shape[-2] + 1)  # Trim start token and padding
                hf_act = hf_act[:, bound, bound]

            _assert_allclose(hf_act, opt_act)
                

def main(args):
    # Make client connection
    client = Client(args.host, args.port)

    # Create HF model
    hf_model = OPTForCausalLM.from_pretrained("/checkpoint/opt_test/original/OPT-125M")

    prompts = [
        "vector matrix",
        "nice working with you all :)",
        "Today is a beautiful day and I want to",
        "what is the meaning of life?",
    ]

    opt_mappings, hf_mappings = init_opt_hf_mappings(len(hf_model.model.decoder.layers))

    for (opt_type, opt_map), (hf_type, hf_map) in zip(opt_mappings.items(), hf_mappings.items()):
        assert opt_type == hf_type

        opt_acts = retrieve_opt_activations(opt_map, client, prompts)
        hf_acts = retrieve_hf_activations(hf_map, client, hf_model, prompts)

        assert_activations_correctness(hf_acts, opt_acts, act_type=opt_type)

    breakpoint()

if __name__ == "__main__":
    main(prepare_args())
