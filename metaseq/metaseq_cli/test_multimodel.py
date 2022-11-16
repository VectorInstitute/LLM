#!/usr/bin/env python3
import argparse
from functools import partial
import os

from accelerate import Accelerator
from transformers import OPTForCausalLM
import torch

from opt_client import Client
from hook_utils import get_activation_capture_hook_dict, apply_forward_hook


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def get_hf_logits(hf_model, prompts, client):
    # need to prepend 2 for start of sequence when getting the input_ids
    input_ids_list = [[2] + p for p in client.tokenize(prompts)]
    max_len = max(map(len, input_ids_list))

    # pad right
    input_ids = torch.as_tensor([i + [1] * (max_len - len(i)) for i in input_ids_list]).cuda()

    # Two forward passes since I'm not sure how HF decodes
    with torch.no_grad():
        # Existing logits
        logits_hf = hf_model(input_ids)[0]

        # New token logit for each example
        new_logits_hf = hf_model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        ).scores[0]     # (bsz, max_new_tokens, vocab_size)

    # Slice out only valid parts of sequence, then cat the newly generated
    # token logits
    logits_hf_list = []
    for i, toks in enumerate(input_ids_list):
        logits_hf_list.append(
            torch.vstack((
                logits_hf[i, 1: len(toks), :].cpu(),    # Trim out start token
                new_logits_hf[i].cpu(),
            ))
        )
    return logits_hf_list


def make_hf_model(model_name, cache_dir):
    # Create huggingface model
    accelerator = Accelerator()
    hf_model, output_loading_info = OPTForCausalLM.from_pretrained(
        "facebook/opt-125m",
        output_loading_info=True,
        low_cpu_mem_usage=True, # Prevents random init of params before load
    )
    hf_model.eval()
    assert not sum(list(output_loading_info.values()), []) # No keys randomly init
    hf_model = accelerator.prepare(hf_model)

    return hf_model


def make_hf_model_collection(model_names, cache_dirs):
    model_collection = {}
    for model_name, cache_dir in zip(model_names, cache_dirs):
        model_collection[model_name] = make_hf_model(model_name, cache_dir)

    return model_collection


def assert_activations_structure(act1, act2):
    assert len(act1) == len(act2)
    for a, b in zip(act1, act2):
        assert a.shape == b.shape


class ForwardModel:
    """
    Generic model class which simply implements a forward pass on a specific
    model, given some prompts.
    """
    def __init__(self, model, forward_fn):
        self.model = model
        self.forward_fn = forward_fn

    def forward(self, prompts):
        # Client case
        if self.model is None:
            return self.forward_fn(prompts)

        # HF case
        else:
            return self.forward_fn(self.model, prompts)


class ForwardDecoder:
    def __init__(self, models, decoding_fn_or_path, **decoding_kwargs):
        self.models = models
        for model in self.models:
            assert isinstance(model, ForwardModel), ("All model instances must "
                                                     "be ForwardModels")
        self.decoding_kwargs = decoding_kwargs

        if isinstance(decoding_fn_or_path, str):
            assert os.path.isfile(decoding_fn_or_path), (f"File: "
                                                         "{decoding_fn_or_path} "
                                                         "not found")
            self.decoding_fn = torch.load(decoding_fn_or_path)
        elif callable(decoding_fn_or_path):
            self.decoding_fn = decoding_fn_or_path
        else:
            raise Exception("Must provide valid decoding function source")

    def decode_step(self, prompts_batched):
        model_outputs = []
        for model in self.model:
            model_outputs.append(model.forward(prompts_batched))

        decoded_token_batched = self.decoding_fn(
            model_outputs,
            **self.decoding_kwargs
        )
        return decoded_token_batched


def allreduce_argmax_decoding_fn(model_outputs, **kwargs):
    reduced_logits = [0 for _ in range(len(model_outputs[0]))]

    # Reduce across model dimension
    for output_batch in model_outputs:
        for i, output in enumerate(output_batch):
            reduced_logits[i] += output

    # Softmax then argmax
    softmax_argmax_logits = [
        logits.to(torch.float32).softmax(-1).argmax(-1)
        for logits in reduced_logits
    ]
    breakpoint()


def main(args):
    # Make model references
    client = Client(args.host, args.port)

    hf_model = make_hf_model(
        "facebook/opt-125m",
        cache_dir="/checkpoint/opt_test/original/OPT-125M",
    )

    prompts = [
        "vector matrix",
        "nice working with you all :)",
        "Today is a beautiful day and I want to",
        "what is the meaning of life?",
    ]

    opt_model_6_7b = ForwardModel(None, client.forward)
    opt_model_125m = ForwardModel(hf_model, partial(get_hf_logits, client=client))

    activations_opt_6_7b = opt_model_6_7b.forward(prompts)
    activations_opt_125m = opt_model_125m.forward(prompts)

    assert_activations_structure(activations_opt_6_7b, activations_opt_125m)


if __name__ == "__main__":
    main(prepare_args())
