from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from metaseq import utils
from metaseq.modules import gelu, MultiheadAttention
from metaseq.modules.dropout import Dropout
from metaseq.modules.fused_bias_gelu import (
    fused_bias_gelu,
    has_fused_bias_gelu,
    load_megatron_fused_kernel,
)
from metaseq.modules.layer_norm import LayerNorm
from metaseq.modules.linear import Linear


def _linear(x, weight, bias=None):
    return F.linear(x, weight, bias)


def _ffn(x, fc1, activation_fn, fc2, dropout_module):
    pass


class FeedForwardNetwork(nn.Module):
    pass


class SwissArmyTransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pass

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention()

    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x, encoder_padding_mask, attn_mask):
        pass


class SwissArmyTransformerDecoderLayer(nn.Module):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        pass

    def _get_model_init_dtype(self):
        pass

    def build_fc1(self, input_dim, output_dim, initialize_params_on_gpu=False, **unused_args):
        pass

    def build_fc2(self, input_dim, output_dim, initialize_params_on_gpu=False, **unused_args):
        pass

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        pass

    def build_encoder_attention(self, embed_dim, args):
        pass

    def residual_connection(self, x, residual):
        return residual + x

    def forward_attention(
        self,
        query,
        key,
        value,
        residual,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=False,
        attn_mas=None,
    ):
        pass
