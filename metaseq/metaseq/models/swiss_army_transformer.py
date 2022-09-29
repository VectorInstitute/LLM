import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from metaseq.dataclass.constants import UNSPECIFIED_DOC_SEP

from metaseq import utils
from metaseq.distributed import utils as distributed_utils, fsdp_wrap
from metaseq.models import BaseEncoder, IncrementalDecoder
from metaseq.modules.swiss_army_transformer_layer import (
    #Dropout,
    #LayerNorm,
    #PositionalEmbedding,
    SwissArmyTransformerDecoderLayer,
    SwissArmyTransformerEncoderLayer,
)
from metaseq.modules.checkpoint_activations import checkpoint_wrapper

logger = logging.getLogger(__name__)


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

class TransformerEncoder(BaseEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)

        # TODO (mchoi): Init embeds, encoder layers, etc.
        pass

    def build_encoder_layer(self, args):
        # Do stuff
        layer = SwissArmyTransformerEncoderLayer(args)
        pass

    def forward_embedding(self, src_tokens, token_embedding):
        # Do stuff
        pass

    def forward(self, src_tokens, src_lengths, token_embeddings):
        return self.forward_scriptable(src_tokens, src_lengths, token_embeddings)

    def forward_scriptable(self, src_tokens, src_lengths, token_embeddings):
        # Do stuff
        pass

    def max_positions(self):
        # Do stuff
        pass


class SwissArmyTransformerDecoder(IncrementalDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args=args
        super().__init__(dictionary)
        # Do stuff
        pass

    def build_base_decoder_layer(self, args, no_encoder_attn=False):
        pass

    def build_decoder_layer(self, args, no_encoder_attn=False):
        pass

    def forward_embedding(self, tokens, token_embedding, incremental_state):
        pass

    def forward(
        self,
        prev_output_tokens,
        encoder_out,
        incremental_state,
        features,
        src_lengths,
        token_embeddings,
        self_attn_padding_mask,
    ):
        pass

    def extract_features(
        prev_output_tokens,
        encoder_out,
        incremental_state,
        token_embeddings,
        self_attn_padding_mask,
    ):
        pass

    def extract_features_scriptable(
        prev_output_tokens,
        encoder_out,
        incremental_state,
        token_embeddings,
        self_attn_padding_mask,
    ):
        pass

    def output_layer(self, features):
        pass

    def max_positions(self):
        pass

    def buffered_future_mask(self, tensor, input_tokens=None):
        pass


def Embedding(
    num_embeddings, embedding_dim, padding_idx, initialize_params_on_gpu=False
):
    pass


def Linear(in_features, out_features, bias=True):
    pass
