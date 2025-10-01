import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np


# @torch.compile
def silu_activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

# @torch.compile
def softmax_activation(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    max_x: torch.Tensor = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp((x - max_x) / temperature)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)