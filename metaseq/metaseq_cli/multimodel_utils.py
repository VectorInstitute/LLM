from functools import partial
from typing import List, Tuple, Optional, Union, Any, Sequence, Callable

from accelerate import Accelerator
from transformers import OPTForCausalLM
import torch
from torch import Tensor

from opt_client import Client


HuggingfaceTransformersModel = Any
HuggingfaceTokenizer = Any


def assert_activation_structure(act1: Tensor, act2: Tensor) -> None:
    assert len(act1) == len(act2)
    for a, b in zip(act1, act2):
        assert a.shape == b.shape


def _single_model_argmax_decoding_fn(
    model_outputs: Sequence[Tensor],
    **kwargs
) -> Sequence[Tensor]:
    softmax_argmax_logits = [
        logits.to(torch.float32).log_softmax(-1).argmax(-1)
        for logits in model_outputs[0]  # Just use the first model
    ]

    output = tuple([
        logits[-1].unsqueeze(0)
        for logits in softmax_argmax_logits
    ])

    return output


def sample(lprobs: Tensor) -> Tuple[Tensor, Tensor]:
    """Pulled directly from `metaseq.sequence_generator`."""
    sampling_topp = 1.0

    probs = torch.softmax(lprobs, dim=-1)
    sprobs, sinds = probs.sort(dim=-1, descending=True)
    mask = (sprobs.cumsum(dim=-1) - sprobs) >= sampling_topp
    trunc_sprobs = sprobs.detach().clone()
    trunc_sprobs[mask] = 0
    trunc_sprobs.div_(trunc_sprobs.sum(dim=-1).unsqueeze(-1))
    choices = torch.multinomial(trunc_sprobs, 1)[0]
    # hyp_ids = torch.arange(lprobs.size(0)).to(lprobs.device)
    tok_ids = sinds[choices]
    scores = sprobs[choices].log()
    return scores, tok_ids


def allreduce_argmax_decoding_fn(
    model_outputs: Sequence[Tensor],
    **kwargs
) -> Sequence[Tensor]:
    """
    Decoding function which reduces along the model dimension, then does a
    softmax argmax to predict the next token.
    """
    assert_activation_structure(*model_outputs)

    next_ids = []
    for i, tuple_of_examples in enumerate(zip(*model_outputs)):
        assert len(set(map(len, tuple_of_examples)))

        tuple_of_logprobs = tuple(
            map(lambda x: x[-1, :].to(torch.float32).log_softmax(-1), tuple_of_examples)
        )
        # Average logits
        cum_logprobs = sum(tuple_of_logprobs) / len(tuple_of_logprobs)

        _scores, samples = sample(cum_logprobs)

        next_ids.append(samples.unsqueeze(0))

    return next_ids


class ClientTokenizer:
    """
    Tokenizer which is implemented by the client. This tokenizer object should
    primarily be used when other clients or models require a tokenizer from
    another client. The complement to this tokenizer object would be something
    like a huggingface Tokenizer.
    """

    def __init__(
        self,
        client: Client,
        pad_token: int = 1,
        start_token: int = 2,
    ) -> None:
        self.client = client
        self.pad_token = pad_token
        self.start_token = start_token

    def _get_tokenized_lengths(
        self,
        tokenized_prompts: List[List[int]],
    ) -> List[int]:
        """Return the lengths of each tokenized example."""
        return list(map(len, tokenized_prompts))

    def tokenize(self, prompts: List[str]) -> List[List[int]]:
        """
        Tokenize a list of string prompts using client-provided
        tokenizer.
        """
        return self.client.tokenize(prompts)

    def prepend_start_token(
        self,
        tokenized_prompts: List[Union[Tensor, List[int]]],
    ) -> List[Union[Tensor, List[int]]]:
        """Prepends a start token to each tokenized prompt in a list."""
        # Return the prompts in the same container as the input
        if isinstance(tokenized_prompts[0], Tensor):
            return [
                torch.cat((torch.LongTensor([self.start_token]), p))
                for p in tokenized_prompts
            ]
        else:
            return [[self.start_token] + p for p in tokenized_prompts]

    def pad_right(
        self,
        tokenized_prompts: Union[List[Tensor], List[List[int]]]
    ) -> Tensor:
        """
        Right-pad a list of tensors, then convert the entire list to one large
        tensor.
        """
        def _tensor_pad_fn(x):
            return torch.nn.functional.pad(
                x, pad=(0, max_len - len(x)), value=self.pad_token)

        def _list_pad_fn(x):
            return x + [self.pad_token] * (max_len - len(x))

        max_len = max(map(len, tokenized_prompts))
        if isinstance(tokenized_prompts[0], Tensor):
            pad_fn = _tensor_pad_fn
            combine_fn = torch.stack
        else:
            pad_fn = _list_pad_fn
            combine_fn = torch.as_tensor

        padded_prompts = [pad_fn(p) for p in tokenized_prompts]
        return combine_fn(padded_prompts)

    def tokenize_prompts(
        self,
        prompts: List[Union[str, List[int]]],
        output_lengths: bool = True
    ) -> Tuple[Tensor, Optional[List[int]]]:
        """
        Tokenize a list of string prompts using the client-provided tokenizer.
        Do additional pre-processing like prepending a padding token and
        pad right for generation.
        """
        # need to prepend 2 for start of sequence when getting the input_ids
        # Case where prompts are already pre-tokenized
        if isinstance(prompts[0], list) and isinstance(prompts[0][0], int):
            input_ids = prompts
        else:
            input_ids = self.tokenize(prompts)

        if output_lengths:
            tokenized_lengths = self._get_tokenized_lengths(input_ids)
        else:
            tokenized_lengths = None

        input_ids_with_start = self.prepend_start_token(input_ids)
        input_ids_with_pad = self.pad_right(input_ids_with_start)

        return input_ids_with_pad, tokenized_lengths

    def prepare_autoregressive_input(
        self,
        list_of_next_tokens: List[Tensor],
        prompts: List[Union[str, List[int]]],
        output_tokens_in_list: bool = True,
    ) -> Union[List[List[int]], List[Tensor]]:
        """
        Tokenize the prompts already used as input to the models and append the
        newly generated tokens to the corresponding batch examples.
        """
        assert len(prompts) == len(list_of_next_tokens)

        # Case where prompts are already pre-tokenized
        if isinstance(prompts[0], list) and isinstance(prompts[0][0], int):
            input_ids = prompts
        elif isinstance(prompts[0], str):
            input_ids = self.tokenize(prompts)
        else:
            raise ValueError("Prompts are in an incorrect format.")

        autoreg_input = []
        for tokenized_prompt, next_token in zip(input_ids, list_of_next_tokens):
            autoreg_input.append(
                torch.cat((torch.LongTensor(tokenized_prompt), next_token), dim=-1)
            )

        if output_tokens_in_list:
            return [p.tolist() for p in autoreg_input]

        return autoreg_input


class ForwardModel:
    """
    Base class which wraps some model or client with a tokenizer. Implements a
    simple forward pass, which generates the next step prediction logits.
    """

    def __init__(
        self,
        model_or_client: Union[HuggingfaceTransformersModel, Client, Any],
        tokenizer: Union[ClientTokenizer, HuggingfaceTokenizer, Any],
    ) -> None:
        self.model_or_client = model_or_client
        self.tokenizer = tokenizer

    def generation_logic(
        self,
        prompts: List[Union[str, List[int]]],
    ) -> List[Tensor]:
        raise NotImplementedError(
            "Child class does not implement the generation_logic function."
        )

    def forward(self, prompts: List[Union[str, List[int]]]) -> List[Tensor]:
        """
        Pass prompts through the generation logic, which will return a list
        of logits.
        """
        logits = self.generation_logic(prompts)
        return logits


class ClientModel(ForwardModel):
    def __init__(
        self,
        client: Client,
        tokenizer: Union[ClientTokenizer, HuggingfaceTokenizer, Any],
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer

    @classmethod
    def build(
        cls,
        client_host: int,
        client_port: int,
        tokenizer: Union[ClientTokenizer, HuggingfaceTokenizer, Any],
    ):
        client = Client(client_host, client_port)
        return cls(client, tokenizer)

    def generation_logic(
        self,
        prompts: List[Union[str, List[int]]]
    ) -> List[Tensor]:
        # If a tokenizer was passed in, then the client does not do
        # tokenization itself
        if self.tokenizer is not None:
            input_ids = self.tokenizer.tokenize(prompts)
            return self.client.forward(input_ids)
        else:
            # `interactive_hosted_updated` can handle List[List[int]]
            assert hasattr(self.client, "tokenize")
            return self.client.forward(prompts)


class HuggingfaceModel(ForwardModel):
    def __init__(
        self,
        hf_model: HuggingfaceTransformersModel,
        tokenizer: Union[ClientTokenizer, HuggingfaceTokenizer, Any],
    ) -> None:
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    @classmethod
    def build(
        cls,
        model_name: str,
        cache_dir: str,
        tokenizer: Union[ClientTokenizer, Any],
    ):
        # Create huggingface model
        accelerator = Accelerator()
        hf_model, output_loading_info = OPTForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir,
            output_loading_info=True,
            # Prevents random init of params before load
            low_cpu_mem_usage=True,
        )
        hf_model.eval()

        # No keys randomly init
        assert not sum(list(output_loading_info.values()), [])

        hf_model = accelerator.prepare(hf_model)

        # Return generic huggingface model wrapper, implementing logit
        # retrieval
        return cls(hf_model, tokenizer)

    def _forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass through huggingface model."""
        with torch.no_grad():
            # Second ele in tuple is for caching I think
            logits_hf = self.hf_model(input_ids)[0]
        return logits_hf

    def _generate(self, input_ids: Tensor) -> Tensor:
        """Brute force the next token logits without sampling from them."""
        with torch.no_grad():
            new_logits_hf = self.hf_model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            ).scores[
                0
            ]  # (bsz, max_new_tokens, vocab_size)
        return new_logits_hf

    def prepare_prompts(
        self, prompts: List[Union[str, List[int]]],
    ) -> Tuple[Tensor, List[int]]:
        input_ids, prompt_lengths = self.tokenizer.tokenize_prompts(
            prompts,
            output_lengths=True,
        )
        input_ids = input_ids.cuda()

        return input_ids, prompt_lengths

    def generation_logic(
        self, prompts: List[Union[str, List[int]]],
    ) -> List[Tensor]:
        """
        Given some prompts, tokenize and preprocess them and give them to
        some huggingface model to obtain the logits. These logits include the
        predictions for only the immediate next token.
        """
        input_ids, prompt_lengths = self.prepare_prompts(prompts)

        # Two forward passes since I'm not sure how huggingface decodes. We
        # just brute force what we want through their API (see _generate fn).
        logits = self._forward(input_ids)
        new_logits = self._generate(input_ids)

        # Slice out only valid parts of sequence, then cat the newly generated
        # token logits
        logits_list = []
        for i, p_length in enumerate(prompt_lengths):
            logits_list.append(
                torch.vstack(
                    (
                        # Trim out start token
                        logits[i, 1: p_length + 1, :].cpu(),
                        new_logits[i].cpu(),
                    )
                )
            )
        return logits_list


class ModelEnsemble:
    def __init__(
        self,
        models: Sequence[Union[Client, HuggingfaceModel]],
        tokenizer: Union[ClientTokenizer, HuggingfaceTokenizer, Any],
        decoding_fn: Callable,
    ) -> None:
        # Tokenizers are already wrapped in the models
        self.models = models
        self.tokenizer = tokenizer
        self.decoding_fn = decoding_fn

    def decode_step(
        self,
        prompts: List[Union[str, List[int]]],
        **kwargs
    ) -> List[Tensor]:
        """
        Do forward step on every single model in the collection, apply the
        aggregation and token decoding function given all of the generated
        model logits, then return the decoded next token prediction.
        """
        model_outputs = []
        for model in self.models:
            model_outputs.append(model.forward(prompts))

        decoded_token_batched = self.decoding_fn(
            model_outputs,
            **kwargs,
        )
        return decoded_token_batched

    def decode(
        self,
        prompts: List[str],
        num_steps: int,
        **kwargs
    ) -> List[List[int]]:
        """
        Decode num_steps times.
        """
        prepare_next_ids_fn = partial(
            self.tokenizer.prepare_autoregressive_input,
            output_tokens_in_list=True,
        )

        step_next_ids = self.decode_step(prompts, **kwargs)
        input_ids = prepare_next_ids_fn(step_next_ids, prompts)
        # Prompts starts out as List[str], but due to tokenization constraints
        # we feed in input_ids:List[List[int]] for steps [1:].
        from tqdm import tqdm

        for i in tqdm(range(1, num_steps)):
            step_next_ids = self.decode_step(input_ids, **kwargs)
            input_ids = prepare_next_ids_fn(step_next_ids, input_ids)
            print(input_ids[0])

        return prepare_next_ids_fn(step_next_ids, input_ids)


