import math

import torch

from megatron.mpu.layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.mpu.utils import divide
from megatron.mpu.initialize import get_tensor_model_parallel_world_size

from metaseq.models.glm_forward_defaults import (
    standard_attention,
    attention_forward_default,
    cross_attention_forward_default,
    mlp_forward_default,
    word_embedding_forward_default,
    position_embedding_forward_default,
    final_forward_default,
    layer_forward_default,
)

# Import fused layer norm with fallback to regular layer norm
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm

    class LayerNorm(FusedLayerNorm):
        def __init__(self, *args, pb_relax=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.pb_relax = pb_relax

        def forward(self, x):
            if not self.pb_relax:
                return super().forward(x)
            return super().forward(x / (x.abs().max().detach() / 8))

except ModuleNotFoundError:
    print("Install apex to use fused_layer_norm. Fallback: torch.nn.LayerNorm")
    from torch.nn import LayerNorm


# TODO (mchoi): Run formatter and change operation calls (ie. matmul to einsum)
# TODO (mchoi): Convert this to a registry instead
HOOKS_DEFAULT = {
    "attention_fn": standard_attention,
    "attention_forward": attention_forward_default,
    "cross_attention_forward": cross_attention_forward_default,
    "mlp_forward": mlp_forward_default,
    "word_embedding_forward": word_embedding_forward_default,
    "position_embedding_forward": position_embedding_forward_default,
    "final_forward": final_forward_default,
    "layer_forward": layer_forward_default,
}


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor, **kwargs):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor, **kwargs):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


@torch.jit.script
def gelu(x):
    """OpenAI's gelu implementation."""
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        init_method,
        layer_id,
        hidden_size_per_attention_head=None,
        output_layer_init_method=None,
        bias=True,
        hooks={},
        transformer_pointer=None,
        params_dtype=torch.float,
        skip_init=False,
        device=torch.device("cpu"),
    ):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size

        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(
                hidden_size, num_attention_heads
            )

        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head

        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.inner_hidden_size = (
            num_attention_heads * self.hidden_size_per_attention_head
        )
        self.hidden_size_per_partition = (
            self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        )

        # Strided linear layer.
        # NOTE: All ColumnParallelLinear layers are conformant to metaseq
        #       megatron fork
        self.query_key_value = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * self.inner_hidden_size,
            bias=bias,
            gather_output=False,
            init_method=init_method,
            stride=3,
            dtype=params_dtype,
        )
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)
        # NOTE: All RowParallelLinear layers are conformant to metaseq
        #       megatron fork
        self.dense = RowParallelLinear(
            input_size=self.inner_hidden_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            dtype=params_dtype,
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        object.__setattr__(self, "transformer", transformer_pointer)
        assert transformer_pointer is not None

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask, *args, **kw_args):
        if "attention_forward" in self.hooks:
            return self.hooks["attention_forward"](hidden_states, mask, **kw_args)

        else:
            return HOOKS_DEFAULT["attention_forward"](
                self, hidden_states, mask, **kw_args
            )


class CrossAttention(torch.nn.Module):
    """Parallel cross-attention layer for Transformer"""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        init_method,
        layer_id,
        hidden_size_per_attention_head=None,
        output_layer_init_method=None,
        bias=True,
        hooks={},
        transformer_pointer=None,
        params_dtype=torch.float,
        skip_init=False,
        device=torch.device("cpu"),
    ):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(
                hidden_size, num_attention_heads
            )
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.inner_hidden_size = (
            num_attention_heads * self.hidden_size_per_attention_head
        )
        self.hidden_size_per_partition = (
            self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        )
        # Strided linear layer.
        self.query = ColumnParallelLinear(
            hidden_size,
            self.inner_hidden_size,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="query",
            skip_init=skip_init,
            device=device,
        )
        self.key_value = ColumnParallelLinear(
            hidden_size,
            2 * self.inner_hidden_size,
            stride=2,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="key_value",
            skip_init=skip_init,
            device=device,
        )
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)
        # Output
        self.dense = RowParallelLinear(
            self.inner_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            params_dtype=params_dtype,
            module=self,
            name="dense",
            skip_init=skip_init,
            device=device,
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        object.__setattr__(self, "transformer", transformer_pointer)
        assert transformer_pointer is not None

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
        # hidden_states: [b, s, h]
        if "cross_attention_forward" in self.hooks:
            return self.hooks["cross_attention_forward"](
                hidden_states, cross_attention_mask, encoder_outputs, **kw_args
            )
        else:
            return HOOKS_DEFAULT["cross_attention_forward"](
                self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args
            )


class MLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        output_dropout_prob,
        init_method,
        inner_hidden_size=None,
        output_layer_init_method=None,
        layer_id=None,
        hooks={},
        bias=True,
        activation_func=gelu,
        transformer_pointer=None,
        params_dtype=torch.float,
        skip_init=False,
        device=torch.device("cpu"),
    ):
        super(MLP, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.inner_hidden_size,
            bias=bias,
            gather_output=False,
            init_method=init_method,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            input_size=self.inner_hidden_size,
            output_size=self.hidden_size,
            bias=bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            dtype=params_dtype,
        )
        self.dropout = torch.nn.Dropout(output_dropout_prob)
        object.__setattr__(self, "transformer", transformer_pointer)
        assert transformer_pointer is not None

    def forward(self, hidden_states, **kw_args):
        if "mlp_forward" in self.hooks:
            output = self.hooks["mlp_forward"](hidden_states, **kw_args)
        else:
            output = HOOKS_DEFAULT["mlp_forward"](self, hidden_states, **kw_args)

        if self.training:
            output = self.dropout(output)
        return output


class BaseTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        layernorm_epsilon,
        init_method,
        layer_id,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        output_layer_init_method=None,
        layernorm_order="pre",
        layernorm=LayerNorm,
        is_decoder=False,
        use_bias=True,
        activation_func=gelu,
        hooks={},
        transformer_pointer=None,
        params_dtype=torch.float,
        skip_init=False,
        device=torch.device("cpu"),
    ):
        super(BaseTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.is_decoder = is_decoder
        self.layernorm_order = layernorm_order
        self.hooks = hooks
        object.__setattr__(self, "transformer", transformer_pointer)
        assert transformer_pointer is not None

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        if self.layernorm_order == "sandwich":
            self.third_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Cross attention.
        if self.is_decoder:
            self.cross_attention = CrossAttention(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                init_method,
                layer_id,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=output_layer_init_method,
                bias=use_bias,
                hooks=hooks,
                transformer_pointer=transformer_pointer,
                params_dtype=params_dtype,
            )
            self.post_cross_attention_layernorm = layernorm(
                hidden_size, eps=layernorm_epsilon
            )
        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            inner_hidden_size=inner_hidden_size,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            layer_id=layer_id,
            activation_func=activation_func,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device,
        )

    def forward(self, hidden_states, mask, *args, **kw_args):
        return HOOKS_DEFAULT["layer_forward"](
            self, hidden_states, mask, *args, **kw_args
        )
