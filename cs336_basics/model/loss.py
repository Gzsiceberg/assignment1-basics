import torch
from torch import nn, Tensor, softmax
from einops import rearrange, einsum
import numpy as np
from jaxtyping import Float, Int, jaxtyped, Bool
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def log_softmax(x: Float[torch.Tensor, "... num_classes"], dim: int = -1) -> Float[torch.Tensor, "... num_classes"]:
    """
    x: (..., num_classes)
    Output: (..., num_classes)
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    x_exp_sum = torch.exp(x).sum(dim=dim, keepdim=True)
    log_probs = x - torch.log(x_exp_sum)
    return log_probs


@jaxtyped(typechecker=typechecker)
def cross_entropy(
    logits: Float[torch.Tensor, "... num_classes"],
    targets: Int[torch.Tensor, "..."],
) -> Float[torch.Tensor, ""]:
    """
    logits: (..., num_classes)
    targets: (...,) with values in [0, num_classes-1]
    Output: scalar loss
    """
    log_probs = log_softmax(logits, dim=-1)
    loss = -log_probs[..., targets].mean()
    return loss


@jaxtyped(typechecker=typechecker)
def perplexity(loss: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
    """
    loss: scalar loss
    FLOPS: 1
    Output: scalar perplexity
    """
    ppl = torch.exp(loss)
    return ppl
