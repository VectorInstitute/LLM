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


def do_nothing(act):
    return act


def diag_elementwise_scaling(act):
    x = act * ((torch.eye(*act.shape[-2:]).cuda() * 2) + 1)
    out = x.to(act.dtype)
    #out = act
    assert out.shape == act.shape
    return out


def replace_with_noise(act):
    out = torch.randn(size=act.shape, dtype=act.dtype).cuda()
    return out


def replace_with_ones(act):
    out = torch.ones_like(act, dtype=act.dtype).cuda()
    return out


def replace_with_stripe(act):
    stripe = torch.linspace(-50, 50, act.shape[-1]).broadcast_to(act.shape).to(act.dtype).cuda()
    return stripe


def undo_diag_elementwise_scaling(act):
    x = act / ((torch.eye(*act.shape[-2:]) * 2) + 1)
    out = x.to(act.dtype)
    #out = act
    assert out.shape == act.shape
    return out


def main(args):
    client = Client(args.host, args.port)
    prompts = [
        "Hi im Matt",
        "there are 10 goose",
    ]

    # Test retrieval
    client.get_activations(["Hi im Matt"], ["decoder"])

    # Test garbage function
    client.get_edited_activations(
        prompts,
        desired_module_activations=["decoder"],
        activation_editing_fns={"foo": "bar"}
    )

    # Compare simple edit
    original_act = client.get_activations(prompts, ["decoder"])
    edited_act = client.get_edited_activations(
        prompts,
        desired_module_activations=["decoder"],
        activation_editing_fns={"decoder": add10}
    )
    assert torch.allclose(original_act[0]["decoder"],
                          sub10(edited_act[0]["decoder"]),
                          atol=10)

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
        desired_module_activations=layers,
        activation_editing_fns={layers[1]: diag_elementwise_scaling}
    )
    for original_ex, perturbed_ex in zip(original_logits, perturbed_logits):
        assert torch.allclose(
            original_ex[layers[0]],
            perturbed_ex[layers[0]],
        )
        assert torch.allclose(
            original_ex[layers[1]],
            undo_diag_elementwise_scaling(perturbed_ex[layers[1]]),
            atol=1e-3,
        )
        assert not torch.allclose(
            original_ex[layers[2]],
            perturbed_ex[layers[2]],
            atol=1e-1,
        )


if __name__ == "__main__":
    args = prepare_args()
    main(args)
