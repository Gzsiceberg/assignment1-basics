import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def extra_repr(self) -> str:
        return "SiLU Activation Function"


class Sofmtax(nn.Module):
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_x: torch.Tensor = torch.max(x, dim=self.dim, keepdim=True).values
        exp_x = torch.exp(x - max_x)
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)

    def extra_repr(self) -> str:
        return f"softmax dim={self.dim}"
