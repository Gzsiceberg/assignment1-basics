import torch
from torch import Tensor, nn
from einops import rearrange, einsum
import numpy as np
from cs336_basics.model.activation import SiLU
from cs336_basics.model.linear import Linear
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


class SwiGLU(nn.Module):
    dim: int
    dim_ff: int
    silu = SiLU()

    def __init__(self, dim: int, dim_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        """
        Parameters: 
            - fc1: (dim_ff, dim)
            - fc2: (dim, dim_ff)
            - fc3: (dim_ff, dim)
            - Total Parameters: 3 * dim * dim_ff
        """
        self.dim = dim
        if dim_ff > 0:
            self.dim_ff = dim_ff
        else:
            # make dim_ff 8/3 of dim rounded to a multiple of 64
            self.dim_ff = int(np.ceil(8 * self.dim / 3 / 64)) * 64
        self.fc1 = Linear(self.dim, self.dim_ff, device=device, dtype=dtype)
        self.fc3 = Linear(self.dim, self.dim_ff, device=device, dtype=dtype)
        self.fc2 = Linear(self.dim_ff, self.dim, device=device, dtype=dtype)


    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... in_features"]:
        """
        x: (..., dim)
        fc1: (dim_ff, dim)
        fc3: (dim_ff, dim)
        fc2: (dim, dim_ff)
        Flops_fc1: 2 * dim * dim_ff * ... (batch size and other dimensions)
        Flops_fc3: 2 * dim * dim_ff * ... (batch size and other dimensions)
        Flops_fc2: 2 * dim * dim_ff * ... (batch size and other dimensions)
        Total Flops: (6 * dim * dim_ff) * ... (batch size and other dimensions)
        Output: (..., dim)
        """
        a = self.silu(self.fc1(x))
        b = self.fc3(x)
        c = a * b
        out = self.fc2(c)
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, dim_ff={self.dim_ff}"
