import codecs
import random
import itertools
import pickle
from dataclasses import dataclass
from functools import cached_property

import requests


def check_response(resp):
    assert (
        resp.status_code == 200
    ), f"error in request with code {resp.status_code} resp {resp.json()}"


@dataclass
class Client:
    host: str
    port: int

    def __post_init__(self):
        self.addr = f"http://{self.host}:{self.port}/completions"
        self.encode_addr = f"http://{self.host}:{self.port}/encode"
        self.module_name_addr = f"http://{self.host}:{self.port}/module_names"

    def generate(
        self,
        prompts,
        temperature=0.7,
        response_length=32,
        top_p=0.9,
        seed=None,
        echo=False,
    ):
        return self._generate(prompts, temperature, response_length, top_p, seed, echo)

    def _generate(
        self,
        prompts,
        temperature=0.7,
        response_length=32,
        top_p=0.9,
        seed=None,
        echo=False,
        desired_module_activations=(),
    ):
        prompt_dict = {
            "prompt": prompts,
            # can tune these parameters
            "temperature": temperature,
            "max_tokens": response_length,
            "top_p": top_p,
            "seed": seed if seed is not None else random.randint(1, 20000),
            # this arg same as the semantics of
            # https://github.com/facebookresearch/metaseq/blob/689fb79b53a2441bf815ae30e64b9438dac027bd/metaseq/hub_utils.py#L568
            "echo": echo,
            "desired_module_activations": desired_module_activations,
        }

        resp = requests.post(self.addr, json=prompt_dict)

        check_response(resp)

        result = resp.json()

        return result

    def get_activations(self, prompts, desired_module_activations):
        result = self._generate(
            prompts=prompts,
            temperature=1.0,
            response_length=0,
            top_p=1.0,
            seed=0,
            echo=False,
            desired_module_activations=desired_module_activations,
        )

        def decode_str(obj_in_str):
            return pickle.loads(codecs.decode(obj_in_str.encode("utf-8"), "base64"))

        activations = [
            {k: decode_str(v) for k, v in c["activations"].items()}
            for c in result["choices"]
        ]

        return activations

    @cached_property
    def module_names(self):
        resp = requests.get(self.module_name_addr)
        check_response(resp)

        return resp.json()["module_names"]

    # TODO: figure this out
    def score(self, input_list, target_list):
        all_toks = self.encode([p for p in itertools.chain(input_list, target_list)])

        all_inputs, input_tok_lens, target_tok_lens = [], [], []

        # concatenate the input and target tokens
        for input_tok, target_tok in zip(
            all_toks[: len(input_list)], all_toks[len(input_list) :]
        ):
            all_inputs.append(input_tok + target_tok)
            # track the length of the tokens
            target_tok_lens.append(len(target_tok))

        result = self._generate(
            prompts=all_inputs,
            temperature=1.0,
            response_length=0,
            top_p=1.0,
            seed=0,
            echo=True,
        )

        tok_log_probs = [c["logprobs"]["token_logprobs"] for c in result["choices"]]

        output = []

        for target_len, tok_probs in zip(target_tok_lens, tok_log_probs):
            output.append(sum(tok_probs[-target_len:]))

        return output

    # this should just score input and targets in a zip
    def encode(
        self,
        input_list,
        target_list,
    ):

        tok_list = self.tokenize([p for p in itertools.chain(input_list, target_list)])

        all_inputs = []
        input_tok_lens = []
        target_tok_lens = []
        for input_tok, target_tok in zip(
            tok_list[: len(input_list)], tok_list[len(input_list) :]
        ):
            all_inputs.append(input_tok + target_tok)
            input_tok_lens.append(len(input_tok))
            target_tok_lens.append(len(target_tok))

        return all_inputs, input_tok_lens, target_tok_lens

    def tokenize(
        self,
        list_of_strs,
    ):
        prompt_dict = {
            "prompt": list_of_strs,
        }


        """TODO: WATCH OUT FOR THE START TOKEN HERE, MAKE SURE IT'S RIGHT
            i.e no start token
        """
        resp = requests.post(self.encode_addr, json=prompt_dict)
        check_response(resp)

        return resp.json()["tok"]


if __name__ == "__main__":
    c = Client("gpu076", 6666)
    print(c.module_names)

    act = c.get_activations(
        ["Today is a beautiful day and I want to"],
        [
            "decoder.layers.8.self_attn.qkv_proj",
        ],
    )
    breakpoint()
