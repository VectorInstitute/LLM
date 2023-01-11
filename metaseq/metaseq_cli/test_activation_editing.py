import argparse

import torch

from opt_client import Client


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def add10(x):
    return x + 10


def sub10(x):
    return x - 10


def perturb_1(act):
    x = act * ((torch.eye(*act.shape[-2:]).cuda() * 6.9) + 1)
    return x.to(act.dtype)


def undo_perturb_1(act):
    x = act / ((torch.eye(*act.shape[-2:]) * 6.9) + 1)
    return x.to(act.dtype)


def main(args):
    client = Client(args.host, args.port)
    prompts = [
        "Hi im Matt",
        "there are 10 goose",
    ]

    # Test retrieval
    client.get_activations(["Hi im Matt"], ["decoder"])

    # Test garbage function
    client.get_edited_activations(prompts, ["decoder"], {"foo": "bar"})

    # Compare simple edit
    original_act = client.get_activations(prompts, ["decoder"])
    edited_act = client.get_edited_activations(prompts, ["decoder"], {"decoder": add10})
    assert torch.allclose(original_act[0]["decoder"], sub10(edited_act[0]["decoder"]), atol=10)

    # Changing an activation upstream should yield different downstream results
    layers = [
        "decoder.layers.1.self_attn.dropout_module",
        "decoder.layers.24.self_attn.dropout_module",
        "decoder",
    ]
    original_logits = client.get_activations(
        prompts,
        layers,
    )
    perturbed_logits = client.get_edited_activations(
        prompts,
        layers,
        {layers[1]: perturb_1}
    )
    for original_ex, perturbed_ex in zip(original_logits, perturbed_logits):
        assert torch.allclose(
            original_ex[layers[0]],
            perturbed_ex[layers[0]],
        )
        assert torch.allclose(
            original_ex[layers[1]],
            undo_perturb_1(perturbed_ex[layers[1]]),
            atol=1e-4,
        )
        assert not torch.allclose(
            original_ex[layers[2]],
            perturbed_ex[layers[2]],
            atol=1e-1,
        )
    breakpoint()


if __name__ == "__main__":
    args = prepare_args()
    main(args)
