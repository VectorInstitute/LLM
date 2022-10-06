import torch
from torch import nn, Tensor


class BaseMixin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
