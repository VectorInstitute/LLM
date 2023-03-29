#!/usr/bin/env python3
import argparse
from typing import List, Dict, Callable, Optional, Any, Tuple

from accelerate import Accelerator
from transformers import OPTForCausalLM
import torch
from torch import Tensor

from metaseq_cli.opt_client import Client
from metaseq_cli.hook_utils import (
    get_activation_capture_hook_dict,
    apply_forward_hook,
)
from metaseq_cli.activation_utils import (
    ActivationPayload,
    replace_with_ones,
)


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# Globals
NUM_LAYERS = 32     # OPT-6.7B specific

_LayerMappingCollection = Dict[str, List[str]]

# Layer type correspondences assists with comparability between layer types in
# models
REMOTE_OPT_MAPPINGS = {
    "logits": ["decoder"],
    "embed_tokens": ["decoder.embed_tokens"],
    "embed_positions": ["decoder.embed_positions"],
    "transformer_layers": [f"decoder.layers.{i}" for i in range(NUM_LAYERS - 1)]
    + ["decoder.layer_norm"],
    "attention_maps": [
        f"decoder.layers.{i}.self_attn.dropout_module" for i in range(NUM_LAYERS)
    ],
    "self_attention": [f"decoder.layers.{i}.self_attn" for i in range(NUM_LAYERS)],
    "self_attention_layer_norm": [
        f"decoder.layers.{i}.self_attn_layer_norm" for i in range(NUM_LAYERS)
    ],
    "fc1": [f"decoder.layers.{i}.fc1" for i in range(NUM_LAYERS)],
    "fc2": [f"decoder.layers.{i}.fc2" for i in range(NUM_LAYERS)],
    "decoder_layer_norm": [
        f"decoder.layers.{i}.final_layer_norm" for i in range(NUM_LAYERS)
    ],
    "output_layer_norm": ["decoder.layer_norm"],
}
HF_OPT_MAPPINGS = {
    "logits": ["model.decoder"],
    "embed_tokens": ["model.decoder.embed_tokens"],
    "embed_positions": ["model.decoder.embed_positions"],
    "transformer_layers": [f"model.decoder.layers.{i}" for i in range(NUM_LAYERS)],
    "attention_maps": [f"model.decoder.layers.{i}.attn_map" for i in range(NUM_LAYERS)],
    "self_attention": [
        f"model.decoder.layers.{i}.self_attn" for i in range(NUM_LAYERS)
    ],
    "self_attention_layer_norm": [
        f"model.decoder.layers.{i}.self_attn_layer_norm" for i in range(NUM_LAYERS)
    ],
    "fc1": [f"model.decoder.layers.{i}.fc1" for i in range(NUM_LAYERS)],
    "fc2": [f"model.decoder.layers.{i}.fc2" for i in range(NUM_LAYERS)],
    "decoder_layer_norm": [
        f"model.decoder.layers.{i}.final_layer_norm" for i in range(NUM_LAYERS)
    ],
    "output_layer_norm": [f"model.decoder.final_layer_norm"],
}


def tree_transpose(tree: Dict[str, List[str]]) -> Dict[str, str]:
    """Helper function to transpose the layer type mappings dictionaries."""
    return {
        leaf: node
        for node, leaves in tree.items()
        for leaf in leaves
    }


def sort_modules(
    activations: Dict[str, Any],
    mappings: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Sort a dictionary of module_name activation pairs based on the order they
    appear in the layer type mappings dictionaries.
    """
    # Define an order over layer types
    order = {module: idx for idx, module in enumerate(mappings.keys())}

    # Mapping between module name and layer type
    transposed_mappings = tree_transpose(mappings)

    # Sort the module names in activations by the defined layer type
    sorted_module_names = sorted(
        activations,
        key=lambda x: order[transposed_mappings[x]],
    )

    # Want to return a dict with the new insertion order
    return {
        module_name: activations[module_name]
        for module_name in sorted_module_names
    }


class TestModel:
    """
    A base model for testing. It contains a mapping for layer
    correspondences if needed. The main feature is the activation manipulation
    function, which returns both activations for retrieval and edited
    activations.
    """
    def __init__(
        self,
        model_name: str,
        mappings: _LayerMappingCollection,
    ) -> None:
        self.model_name = model_name
        self.mappings = mappings

    def _manipulate_activations(
        self,
        prompts: List[str],
        modules: Dict[str, Optional[Callable]],
    ) -> List[Tensor]:
        raise NotImplementedError

    def manipulate_activations(
        self,
        prompts: List[str],
        modules: Dict[str, Optional[Callable]],
    ) -> List[Tensor]:
        return self._manipulate_activations(prompts, modules)


class TestRemoteOPTModel(TestModel):
    """
    Test model for remotely hosted OPT. Uses a client to make requests to the
    underlying model, hosted remotely on a server.
    """
    def __init__(
        self,
        client,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.client = client

    def _manipulate_activations(
        self,
        prompts: List[str],
        modules: Dict[str, Optional[Callable]],
    ) -> List[Tensor]:
        print(f"OPT modules: {modules}")
        editing_fns: Dict[str, Callable] = {}

        for module_name, edit_fn in modules.items():
            if edit_fn is not None:
                editing_fns[module_name] = edit_fn
                print(f"OPT editing: {module_name} - {edit_fn}")
        if len(editing_fns) == 0:
            print("OPT not editing")

        acts = self.client.get_edited_activations(
            prompts,
            desired_module_activations=list(modules.keys()),
            activation_editing_fns=editing_fns,
        )

        def _format_results(activations_batched):
            result = {n: [] for n in modules}
            for acts in activations_batched:
                for name in modules:
                    result[name].append(acts[name])

            # Can't torch.stack along new batch axis since prompt length is
            # dynamic
            return result

        acts = _format_results(acts)
        return acts


class TestHFModel(TestModel):
    """
    Test model for huggingface OPT. Builds the model locally. Provides
    functionality for interacting with the huggingface model both through
    forward hooks and specifc API gateways.
    """
    def __init__(
        self,
        hf_model_name: str = "facebook/opt-6.7b",
        hf_cache_dir: str = "/checkpoint/opt_test/original/OPT-6.7B",
        tokenizer_fn: Callable = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hf_model_name = hf_model_name
        self.hf_cache_dir = hf_cache_dir
        self.tokenizer_fn = tokenizer_fn

        accelerator = Accelerator()
        self.hf_model, output_loading_info = OPTForCausalLM.from_pretrained(
            self.hf_model_name,
            cache_dir=self.hf_cache_dir,
            output_loading_info=True,
            # Prevents random init of params before load
            low_cpu_mem_usage=True,
            # torch_dtype=torch.float32,
            # float32 gives better acc, but T4 can't load
        )
        self.hf_model.eval()
        # No keys randomly init
        assert not sum(list(output_loading_info.values()), [])
        self.hf_model = accelerator.prepare(self.hf_model)

    def _format_prompts(
        self,
        prompts: List[str],
    ) -> Tuple[Tensor, List[List[int]]]:
        """
        Given a prompt list of strings, prepend the BOS token and pad right.
        Return the formatted prompts as tokens.

        Additionally, return the unpadded token sequences. This is used
        specifically to trim activations returned by the model since they are
        batched into one tensor.
        """
        # need to prepend 2 for start of sequence when getting the input_ids
        input_ids_list = self.tokenizer_fn(prompts)

        input_ids_padded = [[2] + p for p in self.tokenizer_fn(prompts)]
        max_len = max(map(len, input_ids_padded))

        # pad right
        formatted_prompts = torch.as_tensor(
            [i + [1] * (max_len - len(i)) for i in input_ids_padded]
        ).cuda()

        return formatted_prompts, input_ids_list

    def _get_hf_logits(self, formatted_prompts: Tensor) -> Any:
        """
        Forward pass through huggingface model using formatted/tokenized
        prompts.
        """
        with torch.no_grad():
            # slice out the start of seq
            output = self.hf_model(
                formatted_prompts,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
            )

        return output

    def _slice_activation(
        self,
        module_name: str,
        activations: Tensor,
        input_ids_list: List[List[int]],
    ) -> List[Tensor]:
        """
        Huggingface return activations are full sequence padded, so we need to
        slice the pads out. Return a list of tensors, one tensor per example
        in the batch.
        """
        mappings = tree_transpose(self.mappings)

        sliced_acts = []
        for act, toks in zip(activations, input_ids_list):
            sequence_bound = slice(1, len(toks) + 1)

            if mappings[module_name] == "attention_maps":
                sliced_act = act[:, sequence_bound, sequence_bound]
            else:
                sliced_act = act[sequence_bound]
            sliced_acts.append(sliced_act)

        return sliced_acts

    def _format_hooked_acts(
        self,
        acts: Dict[str, Tensor],
        input_ids_list: List[List[int]]
    ) -> Dict[str, List[Tensor]]:
        """
        For each module activation, slice the activation to the correct
        sequence lengths. The acts are a dictionary where the keys are the
        module names, and the values singe batched tensors with padded
        sequence lengths.
        """
        results = {
            module_name: self._slice_activation(
                module_name,
                act,
                input_ids_list,
            )
            for module_name, act in acts.items()
        }

        return results

    def _manipulate_activations(
        self,
        prompts: List[str],
        modules: Dict[str, Optional[Callable]],
    ) -> List[Tensor]:
        """
        Given a prompt list of strings and a dictionary of modules to
        manipulate with either None or edit function values, ping the
        huggingface model to get the requested activations.
        """
        print(f"HF OPT modules: {modules}")
        editing_fns: Dict[str, Callable] = {}
        for module_name, edit_fn in modules.items():
            if edit_fn is not None:
                editing_fns[module_name] = edit_fn
                print(f"HF OPT editing: {module_name} - {edit_fn}")
        if len(editing_fns) == 0:
            print("HF OPT not editing")

        # Remove special-cased modules from module list
        # These are activations that HF can return through API
        # TODO: These special-cased modules also can't be edited easily, not
        # sure where to hook onto. Hence we can't actually directly edit these
        # yet
        special_modules = {}
        remove_modules = []
        for m in modules:
            if m in self.mappings["transformer_layers"]:
                layer_idx = int(m.split(".")[-1])
                special_modules[m] = lambda output: output["hidden_states"][1:][layer_idx].cpu()
                remove_modules.append(m)

            elif m in self.mappings["attention_maps"]:
                layer_idx = int(m.split(".")[-2])
                special_modules[m] = lambda output: output["attentions"][layer_idx].cpu()
                remove_modules.append(m)

            elif m in self.mappings["logits"]:
                special_modules[m] = lambda output: output["logits"].cpu()
                remove_modules.append(m)

        for m in remove_modules:
            del modules[m]

        # Define activation payload
        activation_payload = ActivationPayload(
            module_names_activation_retrieval=list(modules.keys()),
            module_editing_fn_pairs=editing_fns,
        )

        # Prepare hook dict and act dict
        hook_dict, acts = get_activation_capture_hook_dict(
            self.hf_model,
            activation_payload,
            aux=(len(prompts),),  # Just the batch size
            model_type="hf",
        )

        # Prepare inputs
        formatted_prompts, input_ids_list = self._format_prompts(prompts)

        with apply_forward_hook(self.hf_model, hook_dict):
            outputs = self._get_hf_logits(formatted_prompts)

        if len(special_modules) != 0:
            for m, fn in special_modules.items():
                acts[m] = fn(outputs)

        acts = self._format_hooked_acts(acts, input_ids_list)

        return acts


def main(args):
    # Make client connection
    client = Client(args.host, args.port)

    prompts = [
        "vector matrix",
        "nice working with you all :)",
        "Today is a beautiful day and I want to",
        "what is the meaning of life?",
    ]

    # Remote OPT module request objects
    remote_opt_modules_edited = {
        "decoder": None,
        "decoder.layers.27": None,
        "decoder.layers.27.self_attn.dropout_module": None,
        "decoder.layers.28.fc1": replace_with_ones,
        "decoder.layers.28.fc2": None,
        "decoder.layers.29.fc1": None,
    }
    remote_opt_modules_plain = {
        "decoder": None,
        "decoder.layers.27": None,
        "decoder.layers.27.self_attn.dropout_module": None,
        "decoder.layers.28.fc1": None,
        "decoder.layers.28.fc2": None,
        "decoder.layers.29.fc1": None,
    }

    # HF OPT module request objects
    hf_opt_modules_edited = {
        "model.decoder": None,
        "model.decoder.layers.27": None,
        "model.decoder.layers.27.attn_map": None,
        "model.decoder.layers.28.fc1": replace_with_ones,
        "model.decoder.layers.28.fc2": None,
        "model.decoder.layers.29.fc1": None,
    }
    hf_opt_modules_plain = {
        "model.decoder": None,
        "model.decoder.layers.27": None,
        "model.decoder.layers.27.attn_map": None,
        "model.decoder.layers.28.fc1": None,
        "model.decoder.layers.28.fc2": None,
        "model.decoder.layers.29.fc1": None,
    }

    # Get remote opt activations with edits
    test_opt_model = TestRemoteOPTModel(
        model_name="remote_opt",
        mappings=REMOTE_OPT_MAPPINGS,
        client=client,
    )
    remote_opt_acts_edited = test_opt_model.manipulate_activations(
        prompts,
        remote_opt_modules_edited,
    )
    remote_opt_acts_edited = sort_modules(
        remote_opt_acts_edited,
        REMOTE_OPT_MAPPINGS,
    )

    # Get opt activations without edits
    remote_opt_acts_plain = test_opt_model.manipulate_activations(
        prompts,
        remote_opt_modules_plain,
    )
    remote_opt_acts_plain = sort_modules(
        remote_opt_acts_plain,
        REMOTE_OPT_MAPPINGS,
    )

    # Get huggingface opt activations with edits
    test_hf_model = TestHFModel(
        model_name="hf_opt",
        mappings=HF_OPT_MAPPINGS,
        hf_model_name="facebook/opt-6.7b",
        hf_cache_dir="/checkpoint/opt_test/original/OPT-6.7B",
        tokenizer_fn=lambda prompts: client.tokenize(prompts),
    )
    hf_opt_acts_edited = test_hf_model.manipulate_activations(
        prompts,
        hf_opt_modules_edited,
    )
    hf_opt_acts_edited = sort_modules(hf_opt_acts_edited, HF_OPT_MAPPINGS)

    # Get huggingface activations without edits
    hf_opt_acts_plain = test_hf_model.manipulate_activations(
        prompts,
        hf_opt_modules_plain,
    )
    hf_opt_acts_plain = sort_modules(hf_opt_acts_plain, HF_OPT_MAPPINGS)

    # Transpose the mapping constants for ease of use
    remote_opt_mappings = tree_transpose(REMOTE_OPT_MAPPINGS)
    hf_opt_mappings = tree_transpose(HF_OPT_MAPPINGS)

    def compare_activations(group1, group2, group1_mappings, group2_mappings):
        # For each module and for each example in the module batch, do a torch
        # allclose comparison
        for (
            (group1_module, group1_batch),
            (group2_module, group2_batch),
        ) in zip(
            group1.items(),
            group2.items()
        ):
            assert group1_mappings[group1_module] == group2_mappings[group2_module]
            for group1_ex, group2_ex in zip(group1_batch, group2_batch):
                if not torch.allclose(group1_ex, group2_ex, atol=1e-1):
                    print(f"Modules {group1_module}: {group2_module} failed "
                          f"allclose with max diff "
                          f"{torch.abs(group1_ex - group2_ex).max()}")

    print("OPT Edited vs HF Edited")
    compare_activations(
        remote_opt_acts_edited,
        hf_opt_acts_edited,
        remote_opt_mappings,
        hf_opt_mappings
    )

    print("OPT Plain vs HF Plain")
    compare_activations(
        remote_opt_acts_plain,
        hf_opt_acts_plain,
        remote_opt_mappings,
        hf_opt_mappings
    )

    print("OPT Plain vs OPT Edited")
    compare_activations(
        remote_opt_acts_plain,
        remote_opt_acts_edited,
        remote_opt_mappings,
        remote_opt_mappings
    )

    print("HF Plain vs HF Edited")
    compare_activations(
        hf_opt_acts_plain,
        hf_opt_acts_edited,
        hf_opt_mappings,
        hf_opt_mappings
    )


if __name__ == "__main__":
    main(prepare_args())
