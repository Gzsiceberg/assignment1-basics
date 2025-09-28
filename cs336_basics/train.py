from types import NoneType
from typing import Callable, Iterable, Optional, overload
import torch
from torch import nn
import numpy as np
import random
from einops import rearrange, einsum
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker
from cs336_basics.model.linear import Linear
from cs336_basics.model.common import get_device
from cs336_basics.model.optimizer import SGD, AdaGrad

is_main_file = __name__ == "__main__"

# Torch
seed = 0
torch.manual_seed(seed)
# NumPy
np.random.seed(seed)
# Python
random.seed(seed)


class LinearModel(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Linear(dim, dim))
        self.final_layer = Linear(dim, 1)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: Float[torch.Tensor, "batch dim"]) -> Float[torch.Tensor, "batch"]:
        B, D = x.shape
        assert (
            D == self.layers[0].in_features
        ), f"Input dimension {D} does not match model dimension {self.layers[0].in_features}"
        for layer in self.layers:
            x = layer(x)

        x = self.final_layer(x)
        assert x.shape == (B, 1), f"Output shape {x.shape} does not match expected shape {(B, 1)}"

        x = x.squeeze(-1)
        assert x.shape == (B,), f"Squeezed output shape {x.shape} does not match expected shape {(B,)}"
        return x


if is_main_file:
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() 
