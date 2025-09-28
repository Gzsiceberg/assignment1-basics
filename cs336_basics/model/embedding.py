import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np


class Embedding(nn.Module):
    num_embeddings: int
    embedding_dim: int

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        """
        Parameters: num_embeddings * embedding_dim
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # make parameters in the correct device and dtype
        init_tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(init_tensor, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(init_tensor)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"