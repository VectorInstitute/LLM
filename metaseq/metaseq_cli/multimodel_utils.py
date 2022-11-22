from typing import List, Tuple, Optional, Union, Any, Sequence, Callable
import math

from accelerate import Accelerator
from transformers import OPTForCausalLM
import torch
from torch import Tensor
from tqdm import tqdm

from opt_client import Client


HuggingfaceTransformersModel = Any


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
) -> List[int]:
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

        start_token = kwargs["start_token"]
        pad_token = kwargs["pad_token"]
        cum_logprobs[start_token] = -math.inf
        cum_logprobs[pad_token] = -math.inf

        _scores, samples = sample(cum_logprobs)

        next_ids.append(samples.item())

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

    def tensorize_prompts(self, prompts: List[List[int]]) -> Tensor:
        return torch.as_tensor(prompts)

    def untensorize_prompts(self, prompts: Tensor) -> List[List[int]]:
        assert prompts.ndim == 2, "Prompts have a 3rd dimension."
        return prompts.tolist()

    def get_tokenized_lengths(
        self,
        tokenized_prompts: List[List[int]],
    ) -> List[int]:
        """Return the lengths of each tokenized example."""
        return list(map(len, tokenized_prompts))

    def prepend_start_token(self, prompts: List[List[int]]) -> List[List[int]]:
        """Prepend start token to each prompt in batch."""
        return [[self.start_token] + p for p in prompts]

    def remove_start_token(self, prompts: List[List[int]]) -> List[List[int]]:
        """Remove start token from each prompt in batch."""
        return [[t for t in p if t != self.start_token] for p in prompts]

    def pad_right(self, prompts: List[List[int]]) -> List[List[int]]:
        """
        Pad right to each prompt in batch, such that the list of lists is
        now square. The inner dimension is the same size as the longest prompt.
        """
        max_len = max(map(len, prompts))

        def pad_fn(x):
            return x + [self.pad_token] * (max_len - len(x))

        return [pad_fn(p) for p in prompts]

    def remove_pad_right(self, prompts: List[List[int]]) -> List[List[int]]:
        """Remove the pad tokens from each prompt in batch."""
        return [[t for t in p if t != self.pad_token] for p in prompts]

    def append_next_ids(
        self,
        prompts: List[List[int]],
        next_ids: List[int]
    ) -> List[List[int]]:
        """
        Given prompts and next tokens, return the prompts with the next
        tokens append to the end of the prompts.

        Note: Will remove any padding tokens if they exist.
        """
        prompts = self.remove_pad_right(prompts)
        return [p + [id_] for p, id_ in zip(prompts, next_ids)]

    def _tokenize(self, prompts: List[str]) -> List[List[int]]:
        """
        Tokenize a list of string prompts using client-provided tokenizer.
        """
        return self.client.tokenize(prompts)

    def _preprocess_prompts(
        self,
        prompts: List[List[int]],
        pad_right: bool = True,
        output_lengths: bool = True,
    ) -> Tuple[List[List[int]], Optional[List[int]]]:
        """
        Tokenize a list of string prompts using the client-provided tokenizer.
        Do additional pre-processing like prepending a padding token and
        pad right for generation.

        NOTE: Prompts are only ever in List[str] format in the first decoding
            step.
        """
        if output_lengths:
            # Always remove pad token and start tokens
            prompt_lengths = self.get_tokenized_lengths(prompts)
        else:
            prompt_lengths = None

        if pad_right:
            input_ids = self.pad_right(prompts)

        return input_ids, prompt_lengths

    def _prepare_ar_input_generic(
        self,
        prompts: List[List[int]],
        output_lengths: bool = True
    ) -> Tuple[List[List[int]], Optional[List[int]]]:
        input_ids, prompt_lengths = self._preprocess_prompts(
            prompts=prompts,
            pad_right=True,
            output_lengths=output_lengths,
        )
        return input_ids, prompt_lengths

    def prepare_ar_input_client(
        self,
        prompts: List[List[int]],
    ) -> List[List[int]]:
        """
        Given prompts and next tokens, combine them for input to the next
        decoding step in a ClientModel.
        """
        input_ids, _ = self._prepare_ar_input_generic(
            prompts,
            output_lengths=False,
        )
        return input_ids

    def prepare_ar_input_hf(
        self,
        prompts: List[List[int]],
    ) -> Tuple[Tensor, List[int]]:
        """
        Given prompts and next tokens, combine them for input to the next
        decoding step in a HuggingfaceModel.
        """
        input_ids, prompt_lengths = self._prepare_ar_input_generic(
            prompts,
            output_lengths=True,
        )
        tensor_ids = self.tensorize_prompts(input_ids).cuda()

        return tensor_ids, prompt_lengths


class ForwardModel:
    """
    Base class which wraps some model or client with a tokenizer. Implements a
    simple forward pass, which generates the next step prediction logits.
    """

    def __init__(
        self,
        model_or_client: Union[HuggingfaceTransformersModel, Client, Any],
        tokenizer: Union[ClientTokenizer],
    ) -> None:
        self.model_or_client = model_or_client
        self.tokenizer = tokenizer

    def prepare_prompts(self, prompts: List[List[int]]) -> Any:
        """
        Given a list of tokenized prompts, format/preprocess them as
        necessary for input to the `generation_logic` function.
        """
        raise NotImplementedError(
            "Child class does not implement the prepare_prompts function."
        )

    def generation_logic(
        self,
        prompts: List[List[int]],
    ) -> List[Tensor]:
        """
        Given a list of tokenized prompts, pre-process them and pass them
        through the model to generate logits for current and next tokens.
        """
        raise NotImplementedError(
            "Child class does not implement the generation_logic function."
        )

    def forward(
        self,
        prompts: List[List[int]],
    ) -> List[Tensor]:
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
        tokenizer: Union[ClientTokenizer],
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer

    @classmethod
    def build(
        cls,
        client_host: int,
        client_port: int,
        tokenizer: Union[ClientTokenizer],
    ):
        client = Client(client_host, client_port)
        return cls(client, tokenizer)

    def prepare_prompts(self, prompts: List[List[int]]) -> List[List[int]]:
        return self.tokenizer.prepare_ar_input_client(prompts=prompts)

    def generation_logic(
        self,
        prompts: List[List[int]],
    ) -> List[Tensor]:
        prepared_prompts = self.prepare_prompts(prompts)
        return self.client.forward(prepared_prompts)


class HuggingfaceModel(ForwardModel):
    def __init__(
        self,
        hf_model: HuggingfaceTransformersModel,
        tokenizer: Union[ClientTokenizer],
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

    def _generate(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Brute force the next token logits without sampling from them."""
        with torch.no_grad():
            # Get the logprobs for the input context
            logits = self.hf_model(input_ids)[0]

            # Get the lobprobs for the newly generated tokens
            new_logits = self.hf_model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            ).scores[0]  # (bsz, max_new_tokens, vocab_size)

        return logits, new_logits

    def prepare_prompts(
        self,
        prompts: List[List[int]]
    ) -> Tuple[Tensor, List[int]]:
        return self.tokenizer.prepare_ar_input_hf(prompts=prompts)

    def generation_logic(
        self,
        prompts: List[List[int]],
    ) -> List[Tensor]:
        """
        Given some prompts, tokenize and preprocess them and give them to
        some huggingface model to obtain the logits. These logits include the
        predictions for only the immediate next token.
        """
        prepared_prompts, prompt_lengths = self.prepare_prompts(prompts)

        # Two forward passes since I'm not sure how huggingface decodes. We
        # just brute force what we want through their API (see _generate fn).
        logits, new_logits = self._generate(prepared_prompts)

        # Slice out only valid parts of sequence, then cat the newly generated
        # token logits
        logits_list = []
        for i, p_length in enumerate(prompt_lengths):
            logits_list.append(
                torch.vstack(
                    (
                        # Trim out start token
                        logits[i, 1: p_length, :].cpu(),
                        new_logits[i].cpu(),
                    )
                )
            )
        return logits_list


class ModelEnsemble:
    def __init__(
        self,
        models: Sequence[Union[Client, HuggingfaceModel]],
        tokenizer: Union[ClientTokenizer],
        decoding_fn: Callable,
    ) -> None:
        # Tokenizers are already wrapped in the models
        self.models = models
        self.tokenizer = tokenizer
        self.decoding_fn = decoding_fn

    def decode_step(
        self,
        prompts: List[List[int]],
        **kwargs
    ) -> List[List[int]]:
        """
        Do forward step on every single model in the collection, apply the
        aggregation and token decoding function given all of the generated
        model logits, then return the decoded next token prediction.
        """
        # Each model formats the prompt itself, then forward pass to generate
        # logits per model
        model_outputs = []
        for model in self.models:
            logits = model.forward(prompts)
            model_outputs.append(logits)

        # Get next tokens batch
        decoded_token_batched = self.decoding_fn(
            model_outputs,
            **kwargs,
        )

        # Update original prompts and return them
        updated_prompts = self.tokenizer.append_next_ids(
            prompts, decoded_token_batched)

        return updated_prompts

    def decode(
        self,
        initial_prompts: List[str],
        num_steps: int,
        **kwargs
    ) -> List[List[int]]:
        """
        Decode num_steps times.
        """
        # Do preprocessing that just needs to happen once
        prompts = self.tokenizer.prepend_start_token(
            self.tokenizer._tokenize(initial_prompts),
        )

        for i in tqdm(range(0, num_steps)):
            prompts = self.decode_step(prompts, **kwargs)

        return self.tokenizer.remove_pad_right(
            self.tokenizer.remove_start_token(prompts)
        )
