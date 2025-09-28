import torch
from torch import nn, Tensor, softmax
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def log_softmax(x: Float[torch.Tensor, "batch num_classes"], dim: int = -1) -> Float[torch.Tensor, "batch num_classes"]:
    """
    x: (batch, num_classes)
    FLOPS: batch * num_classes
    Output: (batch, num_classes)
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    x_exp_sum = torch.exp(x).sum(dim=dim, keepdim=True)
    log_probs = x - torch.log(x_exp_sum)
    return log_probs


@jaxtyped(typechecker=typechecker)
def cross_entropy(
    logits: Float[torch.Tensor, "batch num_classes"],
    targets: Int[torch.Tensor, "batch"],
) -> Float[torch.Tensor, ""]:
    """
    logits: (batch, num_classes)
    targets: (batch,) with values in [0, num_classes-1]
    FLOPS: batch * num_classes
    Output: scalar loss
    """
    B = logits.shape[0]
    log_probs = log_softmax(logits, dim=-1)
    loss = -log_probs[torch.arange(B), targets].mean()
    return loss
