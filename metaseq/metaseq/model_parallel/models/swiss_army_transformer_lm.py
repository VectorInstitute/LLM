import torch
import torch.nn as nn

from metaseq.models import register_model, register_model_architecture


@register_model("model_parallel_swiss_army_transformer_lm")
class ModelParallelSwissArmyTransformerLM(SwissArmyTransformerLanguageModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance"""
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        raise NotImplementedError

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        raise NotImplementedError
