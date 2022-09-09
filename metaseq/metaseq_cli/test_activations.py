#!/usr/bin/env python3
from transformers import OPTForCausalLM
import torch

from opt_client import Client


prompts = [
    "Today is a beautiful day and I want to",
    "what is the meaning of life?",
    "vector matrix",
]


def main():
    client = Client("gpu076", 8888)

    # need to prepend 2 for start of sequence when getting the input_ids
    input_ids_list = [[2] + p for p in client.tokenize(prompts)]
    max_len = max(map(len, input_ids_list))
    # pad right
    input_ids = torch.as_tensor([i + [1] * (max_len - len(i)) for i in input_ids_list])

    hf_model = OPTForCausalLM.from_pretrained("125m")
    hf_logits_list = []

    with torch.no_grad():
        # slice out the start of seq
        logits_hf = hf_model(input_ids)[0]

    logits_hf_list = [
        logits_hf[i, 1 : len(toks), :] for i, toks in enumerate(input_ids_list)
    ]

    act = client.get_activations(prompts, client.module_names)

    for hf_impl, server_impl in zip(logits_hf_list, act):
        assert torch.allclose(
            hf_impl.cpu().float(),
            server_impl["decoder"].cpu().float(),
            atol=1e-1,
        )


if __name__ == "__main__":
    main()