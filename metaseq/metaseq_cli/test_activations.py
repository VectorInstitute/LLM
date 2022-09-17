#!/usr/bin/env python3
import argparse
import logging

from transformers import OPTForCausalLM
import torch

from opt_client import Client

logger = logging.getLogger(__name__)


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def assert_logits_correctness(hf_logits_batched, opt_logits_batched):
    # TODO: Only works for logits, ie. module list size 1
    assert len(hf_logits_batched) == len(opt_logits_batched)

    for hf_logits, opt_logits in zip(hf_logits_batched, opt_logits_batched):
        opt_logits = next(iter(opt_logits.values()))    # Prune one-entry dict
        assert torch.allclose(
            hf_logits.cpu().float(),
            opt_logits.cpu().float(),
            atol=1e-1
        ), "Difference in hf and opt logits detected!"


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


def get_opt_logits(client, prompts):
    logits = client.get_activations(prompts, ["decoder"])
    return logits


def get_opt_activations(client, prompts, module_names):
    act = client.get_activations(prompts, module_names)
    return act


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

    # NOTE: Huggingface seems to apply the final layer norm on the last hidden state,
    # ie. decoder.layers.11, so we also have to take the output of the final
    # layer norm in OPT
    module_names_for_transformer_layer_outputs = [f"decoder.layers.{i}" for i in range(len(hf_model.model.decoder.layers) - 1)]
    module_names_for_transformer_layer_outputs += ["decoder.layer_norm"]
    module_names_for_attention_maps = [f"decoder.layers.{i}.self_attn.dropout_module" for i in range(len(hf_model.model.decoder.layers))]

    # Get logits from hf and remote opt models and test allclose
    hf_logits = get_hf_logits(client, hf_model, prompts)
    opt_logits = get_opt_logits(client, prompts)
    assert_logits_correctness(hf_logits, opt_logits)

    # Get transformer layer hidden output states/activations
    def transpose_opt_results(activations_batched, module_names):
        """
        Transposes the batched activations dictionary.

        OPT activations have batch size in outer dim and layer # as inner
        dim which mismatches HF activations
        """
        result = {n: [] for n in module_names}
        for acts in activations_batched:
            for name in module_names:
                result[name].append(acts[name])
        return result

    def assert_activations_correctness(hf_results, opt_results, act_type="transformer_layer"):
        """
        Helper function taking HF and OPT activation collections, and
        makes sure they're allclose
        """
        for (module_name, opt_acts), hf_acts in zip(opt_results.items(), hf_results):
            for opt_act, hf_act in zip(opt_acts, hf_acts):

                opt_act = opt_act.detach().cpu().float()
                hf_act = hf_act.detach().cpu().float()

                if act_type == "transformer_layer":
                    bound = slice(1, opt_act.shape[-2] + 1)  # Trim start token and padding
                    hf_act = hf_act[bound]

                elif act_type == "attention_maps":
                    bound = slice(1, opt_act.shape[-2] + 1)  # Trim start token and padding
                    hf_act = hf_act[:, bound, bound]

                else:
                    raise NotImplementedError("Activation assertion only "
                                              "supports transformer and attention "
                                              "map comparisons")

                assert torch.allclose(hf_act, opt_act, atol=1e-1), f"all-close failed on {module_name}"

    # Test output of transformer layers
    opt_acts_batched = get_opt_activations(client, prompts, module_names_for_transformer_layer_outputs)
    hf_acts_batched = get_hf_activations(client, hf_model, prompts)
    result_hf_layer_outputs = hf_acts_batched["hidden_states"][1:]  # Trim output of initial embeddings
    result_opt_layer_outputs = transpose_opt_results(opt_acts_batched, module_names_for_transformer_layer_outputs)
    assert_activations_correctness(result_hf_layer_outputs, result_opt_layer_outputs, act_type="transformer_layer")
    logger.info("Transformer layer outputs passed allclose")

    # Test attention maps for each layer
    opt_attention_maps = get_opt_activations(client, prompts, module_names_for_attention_maps)
    result_opt_attention_maps = transpose_opt_results(opt_attention_maps, module_names_for_attention_maps)
    result_hf_attention_maps = hf_acts_batched["attentions"]
    assert_activations_correctness(result_hf_attention_maps, result_opt_attention_maps, act_type="attention_maps")
    logger.info("Attention maps passed allclose")

    breakpoint()


if __name__ == "__main__":
    main(prepare_args())
