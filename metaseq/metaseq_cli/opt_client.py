import codecs
import random
import itertools
import pickle
from dataclasses import dataclass
from functools import cached_property

import cloudpickle
import requests

from metaseq_cli.activation_utils import ActivationPayload


def check_response(resp):
    assert (
        resp.status_code == 200
    ), f"error in request with code {resp.status_code} resp {resp.json()}"


def encode_obj(obj):
    return codecs.encode(cloudpickle.dumps(obj), "base64").decode("utf-8")


def decode_str(obj_in_str):
    return pickle.loads(codecs.decode(obj_in_str.encode("utf-8"), "base64"))


@dataclass
class Client:
    host: str
    port: int

    def __post_init__(self):
        self.addr = f"http://{self.host}:{self.port}/completions"
        self.encode_addr = f"http://{self.host}:{self.port}/encode"
        self.module_name_addr = f"http://{self.host}:{self.port}/module_names"
        self.weight_addr = f"http://{self.host}:{self.port}/weight"

    def _generate(
        self,
        prompts,
        temperature=0.7,
        response_length=32,
        top_p=0.9,
        echo=False,
        logprobs=0,
        encoded_activation_payload=None,
    ):
        prompt_dict = {
            "prompt": prompts,
            # can tune these parameters
            "temperature": temperature,
            "max_tokens": response_length,
            "top_p": top_p,
            # this arg same as the semantics of
            # https://github.com/facebookresearch/metaseq/blob/689fb79b53a2441bf815ae30e64b9438dac027bd/metaseq/hub_utils.py#L568
            "echo": echo,
            "encoded_activation_payload": encoded_activation_payload,
            "logprobs": logprobs,
        }

        resp = requests.post(self.addr, json=prompt_dict)

        check_response(resp)

        result = resp.json()

        return result

    def generate(
        self,
        prompts,
        temperature=0.7,
        response_length=32,
        top_p=0.9,
        echo=False,
    ):
        return self._generate(prompts, temperature, response_length, top_p, echo)

    @cached_property
    def module_names(self):
        resp = requests.get(self.module_name_addr)
        check_response(resp)

        return resp.json()["module_names"]

    def weight(self, module_name):
        """
        Helper function that gives some flexibility to pinging various model
        states. This only retrieves a single rank's weights however, so do not
        use outside of debugging.
        """
        resp = requests.get(self.weight_addr, json={"module_name": module_name})

        check_response(resp)

        ret_string = resp.json()["weight"]

        return decode_str(ret_string).cpu()

    def tokenize(
        self,
        list_of_strs,
    ):
        prompt_dict = {
            "prompt": list_of_strs,
        }

        resp = requests.post(self.encode_addr, json=prompt_dict)
        check_response(resp)

        return resp.json()["tok"]

    def score(self, input_list, target_list):
        """can think of context as the input_list and the token logprobs we want as the target_list"""
        all_toks = self.tokenize([p for p in itertools.chain(input_list, target_list)])

        all_inputs, target_tok_lens = [], []

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
            echo=True,
        )

        tok_log_probs = [c["logprobs"]["token_logprobs"] for c in result["choices"]]

        output = []

        for target_len, tok_probs in zip(target_tok_lens, tok_log_probs):
            output.append(sum(tok_probs[-target_len:]))

        return output

    def get_activations(self, prompts, desired_module_activations):
        # Don't break API yet
        return self.get_edited_activations(
            prompts,
            desired_module_activations,
            activation_editing_fns=None,
        )

    def get_edited_activations(
        self,
        prompts,
        desired_module_activations,
        activation_editing_fns=None,
    ):
        assert desired_module_activations is not None
        if activation_editing_fns is None:
            activation_editing_fns = {}

        activation_payload = ActivationPayload(
            module_names_activation_retrieval=desired_module_activations,
            module_editing_fn_pairs=activation_editing_fns,
        )

        # Make sure module names are valid
        module_names = self.module_names
        desired_module_activations = set(desired_module_activations)

        for m in desired_module_activations:
            assert m in module_names, f"Module: {m} does not exist in model"

        # Encode the payload for transit over http
        encoded_activation_payload = encode_obj(activation_payload)

        result = self._generate(
            prompts=prompts,
            temperature=1.0,
            response_length=0,
            top_p=1.0,
            echo=False,
            encoded_activation_payload=encoded_activation_payload,
        )

        # Decode the return string back into tensors
        activations = [
            {k: decode_str(v) for k, v in c["activations"].items()}
            for c in result["choices"]
        ]

        return activations
