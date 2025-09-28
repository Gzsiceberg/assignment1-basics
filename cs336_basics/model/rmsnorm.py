import torch
from torch import nn
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


class RMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        """
        Parameters: dim
        """
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim)
        scale: (dim,)
        Flops: 2 * dim * ... (batch size and other dimensions) for power, sum, division, sqrt, multiplication
        Output: (..., dim)
        """
        assert x.shape[-1] == self.dim, f"Input dimension {x.shape[-1]} does not match RMSNorm dimension {self.dim}"
        in_dtype = x.dtype
        x = x.to(torch.float32)
        sqr_x = x.pow(2).sum(dim=-1, keepdim=True) / self.dim + self.eps
        rms_x = torch.sqrt(sqr_x)
        x = x / rms_x
        x = x * self.scale
        return x.to(in_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# todo: try to implement https://arxiv.org/abs/2503.10622
