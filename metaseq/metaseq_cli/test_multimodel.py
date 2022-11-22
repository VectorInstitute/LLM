#!/usr/bin/env python3
import argparse

from multimodel_utils import (
    assert_activation_structure,
    ClientTokenizer,
    ClientModel,
    HuggingfaceModel,
    ModelEnsemble,
    allreduce_argmax_decoding_fn,
)


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main(args):
    # Declare prompts we will use
    prompts = [
        "there are many chickens in",
        "I'm going to the store today, where",
        "Today is a beautiful day and I want to",
        "what is the meaning of life?",
        "Chipotle is",
    ]

    # Client model implements a tokenizer
    client_model = ClientModel.build(
        client_host=args.host,
        client_port=args.port,
        tokenizer=None,
    )

    # Create a tokenizer based off of client model to be passed to other
    # models. This tokenizer will only NOT be used in the client initial
    # tokenization pipeline.
    master_tokenizer = ClientTokenizer(
        client=client_model.client,
        pad_token=1,
        start_token=2,
    )
    client_model.tokenizer = master_tokenizer

    # Instantiate locally-hosted huggingface model
    hf_model = HuggingfaceModel.build(
        model_name="facebook/opt-6.7b",
        cache_dir="/checkpoint/opt_test/original/OPT-6.7B",
        tokenizer=master_tokenizer,
    )

    # Put models into an ensemble
    model_ensemble = ModelEnsemble(
        models=(client_model, hf_model),
        tokenizer=master_tokenizer,
        decoding_fn=allreduce_argmax_decoding_fn,
    )

    # Do multiple decoding steps on the model ensemble
    model_ensemble_generation = model_ensemble.decode(
        prompts,
        num_steps=10,
        start_token=2,
        pad_token=1,
    )

    # Also do decoding on the regular OPT for comparison
    default_opt_generation = client_model.client.generate(
        prompts,
        temperature=1.0,
        response_length=10,
        top_p=1.0,
        echo=True,
    )

    # Output of model ensemble needs to be decoded by tokenizer
    import tokenizers
    tokenizer = tokenizers.ByteLevelBPETokenizer(
        "/checkpoint/opt_test/original/gpt2-vocab.json",
        "/checkpoint/opt_test/original/gpt2-merges.txt",
    )
    print(tokenizer.decode_batch(model_ensemble_generation))

    for c in default_opt_generation["choices"]:
        print(c["text"])


if __name__ == "__main__":
    main(prepare_args())
